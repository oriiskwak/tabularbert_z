# extract_z_cv.py  (seed별 pretrain→finetune→test, group-aware split, no leakage)

import os, random, hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier

from tabularbert import TabularBERTTrainer
from tabularbert.utils.data import QuantileDiscretize
from tabularbert.utils.metrics import ClassificationError

# =========================
# 경로 / 하이퍼파라미터
# =========================
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PRETRAIN_CKPT = './pretraining/version0/model_checkpoint.pt'  # ✨ 사전학습 ckpt 경로
NUM_BINS = 50
SAVE_DIR = "/home/labhosik4609/DG/tabularbert/embedding_xgboost"
os.makedirs(SAVE_DIR, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]
N_SPLITS = 5  # SGKF에서 test≈20%

# =========================
# 유틸
# =========================
def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def extract_Z(model, bin_ids_np, device: torch.device, pool: str = 'mean'):
    b = torch.tensor(bin_ids_np, dtype=torch.long, device=device)
    emb = model.embedding(b)         # (N, F+1, D) [CLS] 포함
    contextual = model.bert(emb)     # (N, F+1, D)
    if pool == 'cls':
        z = contextual[:, 0, :]
    else:
        z = contextual[:, 1:, :].mean(dim=1)  # CLS 제외 평균
    return z.detach().cpu().numpy()

def run_xgb(Xtr, Xte, ytr, yte, name: str, seed: int):
    clf = XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        n_jobs=-1, eval_metric='logloss'
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    roc = roc_auc_score(yte, proba)
    pr  = average_precision_score(yte, proba)
    f1  = f1_score(yte, pred)
    print(f"[{name}] seed={seed} | ROC-AUC={roc:.4f} | PR-AUC={pr:.4f} | F1@0.5={f1:.4f}")
    return clf, proba, {'Seed': seed, 'Variant': name, 'ROC_AUC': roc, 'PR_AUC': pr, 'F1@0.5': f1}

def get_threshold_table(y_true, proba, name, n_steps=21):
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    roc_auc = roc_auc_score(y_true, proba)
    pr_auc  = average_precision_score(y_true, proba)

    thresholds = np.linspace(0.0, 1.0, n_steps)
    rows = []
    P = max(int((y_true == 1).sum()), 1)
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        TP = int(((y_pred == 1) & (y_true == 1)).sum())
        PP = max(int((y_pred == 1).sum()), 1)
        rows.append({
            'Variant': name,
            'Threshold': float(t),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'Precision': TP / PP,
            'Recall': TP / P,
        })
    th_df = pd.DataFrame(rows)
    best_idx = th_df['F1'].idxmax()
    summary_row = {
        'Variant': name,
        'ROC_AUC': float(roc_auc),
        'PR_AUC': float(pr_auc),
        'Best_F1': float(th_df.loc[best_idx, 'F1']),
        'Best_Threshold': float(th_df.loc[best_idx, 'Threshold']),
    }
    roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    pr_df  = pd.DataFrame({'Recall': rec, 'Precision': prec})
    return summary_row, th_df, roc_df, pr_df

# =========================
# 데이터 로드 + 그룹 키 생성
# =========================
df = pd.read_csv('./datasets/hitcall.csv').drop_duplicates()  # 완전중복 제거 권장
X_all = df.iloc[:, :-1].to_numpy()
y_all = pd.Categorical(df.iloc[:, -1]).codes.astype(int)

# 근접중복까지 묶기 위해 반올림 후 해시
X_round = np.round(X_all.astype(np.float64), 6)
group_ids = np.array([hashlib.md5(r.tobytes()).hexdigest() for r in X_round])

# (선택) 라벨 충돌 그룹 제거: 같은 feature 그룹에 서로 다른 y가 섞여 있으면 드롭
tmp = pd.DataFrame({'g': group_ids, 'y': y_all})
conflict = set(tmp.groupby('g')['y'].nunique()[lambda s: s > 1].index)
if len(conflict) > 0:
    print(f"[WARN] label-conflict groups: {len(conflict)} -> dropping them")
    keep = ~np.isin(group_ids, list(conflict))
    X_all, y_all, X_round, group_ids = X_all[keep], y_all[keep], X_round[keep], group_ids[keep]

# =========================
# 실험 루프
# =========================
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
xlsx_path = os.path.join(SAVE_DIR, f"tabbert_xgb_cv_{ts}.xlsx")
print("Excel will be saved to ->", xlsx_path)

all_rows = []
per_seed_tables = []

for idx, seed in enumerate(SEEDS, start=1):
    print(f"\n========== RUN {idx}/{len(SEEDS)} | seed={seed} ==========")
    set_global_seed(seed)

    # 1) 그룹 보존 + 레이블 유지 분할 (test ≈ 20%)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(X_round, y_all, groups=group_ids))
    assert len(set(group_ids[train_idx]) & set(group_ids[test_idx])) == 0, "Group leakage!"

    X_tr, X_te = X_all[train_idx], X_all[test_idx]
    y_tr, y_te = y_all[train_idx], y_all[test_idx]

    # 2) RAW 스케일러: train-only fit
    raw_scaler = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=min(1000, len(X_tr)),
        random_state=seed
    )
    raw_scaler.fit(X_tr)
    Xtr_raw = raw_scaler.transform(X_tr)
    Xte_raw = raw_scaler.transform(X_te)

    # 3) 사전학습 ckpt 로드 → seed별로 새로 finetune (train-only)
    trainer = TabularBERTTrainer.from_pretrained(
        save_path=PRETRAIN_CKPT,
        device=DEVICE
    )
    seed_dir = f'./fine-tuning_cv/seed_{seed}'
    os.makedirs(seed_dir, exist_ok=True)

    trainer.setup_directories_and_logging(
    save_dir='./fine-tuning_cv',          # 루트만
    phase='fine-tuning',
    project_name=f'FT_seed_{seed}',       # seed로 구분
    use_wandb=False )
    # inner validation split (train 내부에서만)
    X_tr_tr, X_tr_val, y_tr_tr, y_tr_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr
    )
    # 파인튜닝 (API는 레포에 맞춰 필요시 조정)
    trainer.finetune(
        x=X_tr_tr, y=y_tr_tr,
        valid_x=X_tr_val, valid_y=y_tr_val,
        epochs=200, batch_size=256,
        criterion=nn.CrossEntropyLoss(),
        metric=ClassificationError(ignore_index=-100),
        num_workers=0
    )
    trainer.eval()

    # 4) binning: train-only fit → B_tr/B_te
    encoding_info = trainer.encoding_info  # ckpt에 저장된 인코딩 정보
    disc = QuantileDiscretize(num_bins=NUM_BINS, encoding_info=encoding_info)
    disc.fit(X_tr)
    B_tr = disc.discretize(X_tr)
    B_te = disc.discretize(X_te)

    # 5) Z 추출(이 seed에서 방금 finetune된 모델로!)
    Ztr = extract_Z(trainer.model, B_tr, device=DEVICE, pool='mean')
    Zte = extract_Z(trainer.model, B_te, device=DEVICE, pool='mean')

    # 6) XGB 비교: RAW vs Z
    _, proba_raw, row_raw = run_xgb(Xtr_raw, Xte_raw, y_tr, y_te, 'RAW', seed)
    _, proba_z,   row_z   = run_xgb(Ztr,     Zte,    y_tr, y_te, 'TAB_BERT_Z(mean)', seed)
    all_rows += [row_raw, row_z]

    # 7) 시드별 세부 표
    sum_raw, th_raw, roc_raw, pr_raw = get_threshold_table(y_te, proba_raw, 'RAW')
    sum_z,   th_z,   roc_z,  pr_z   = get_threshold_table(y_te, proba_z,   'TAB_BERT_Z(mean)')

    per_seed_tables.append((f'summary_seed{seed}', pd.DataFrame([sum_raw, sum_z])))
    per_seed_tables.append((f'th_raw_{seed}', th_raw))
    per_seed_tables.append((f'th_z_{seed}',   th_z))
    per_seed_tables.append((f'roc_raw_{seed}', roc_raw))
    per_seed_tables.append((f'roc_z_{seed}',   roc_z))
    per_seed_tables.append((f'pr_raw_{seed}',  pr_raw))
    per_seed_tables.append((f'pr_z_{seed}',    pr_z))

# 8) 전체 요약
summary_df = pd.DataFrame(all_rows)
pivot_mean = summary_df.groupby('Variant', as_index=False)[['ROC_AUC','PR_AUC','F1@0.5']].mean()
pivot_std  = summary_df.groupby('Variant', as_index=False)[['ROC_AUC','PR_AUC','F1@0.5']].std()

meta = {
    'pretrain_ckpt': os.path.abspath(PRETRAIN_CKPT),
    'device': str(DEVICE),
    'n_samples_total': int(len(X_all)),
    'n_features': int(X_all.shape[1]),
    'num_bins': int(NUM_BINS),
    'seeds': str(SEEDS),
    'n_splits': int(N_SPLITS),
}

# 9) 엑셀 저장 (단 한 번)
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    for sheet_name, df in per_seed_tables:
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    summary_df.to_excel(writer, sheet_name='all_runs', index=False)
    pivot_mean.to_excel(writer, sheet_name='mean_by_variant', index=False)
    pivot_std.to_excel(writer,  sheet_name='std_by_variant',  index=False)
    pd.DataFrame([meta]).to_excel(writer, sheet_name='meta', index=False)

print("Saved Excel ->", xlsx_path)
