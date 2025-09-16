# x_vs_z_xgb_dedup.py
import os, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             roc_curve, precision_recall_curve)
from xgboost import XGBClassifier

# -------- 설정 --------
DATA_CSV = "./datasets/hitcall.csv"
Z_DIR    = "./z_out"  # Z_tr.npy, Z_va.npy, Z_te.npy, y_*.npy 들이 여기에 있다고 가정
ROUND_DECIMALS = 6     # 근접중복 판단용 반올림 자릿수
SEED = 0               # Z 저장 시 썼던 split과 동일해야 함
# ----------------------

def best_f1(y_true, proba, steps=201):
    thr = np.linspace(0,1,steps)
    f1s = [f1_score(y_true, (proba>=t).astype(int), zero_division=0) for t in thr]
    i = int(np.argmax(f1s))
    return f1s[i], float(thr[i])

def run_xgb(*, Xtr, Xte, y_tr, y_te, name, seed=42):
    clf = XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        n_jobs=-1, eval_metric='logloss'
    )
    clf.fit(Xtr, y_tr)
    proba = clf.predict_proba(Xte)[:,1]
    roc = roc_auc_score(y_te, proba)
    pr  = average_precision_score(y_te, proba)
    f1_05 = f1_score(y_te, (proba>=0.5).astype(int), zero_division=0)
    bf1, bthr = best_f1(y_te, proba)
    print(f"[{name}] ROC={roc:.4f} | PR={pr:.4f} | F1@0.5={f1_05:.4f} | BestF1={bf1:.4f} @ thr={bthr:.3f}")
    return proba, roc, pr

def main():
    # 1) 원본 데이터 & 동일 split 복원 (Z를 만들 때와 동일)
    df = pd.read_csv(DATA_CSV)
    X = df.iloc[:, :-1].to_numpy()
    y = pd.Categorical(df.iloc[:, -1]).codes.astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size=0.8, random_state=SEED, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, train_size=0.8, random_state=SEED, stratify=y_tr
    )

    # 2) train(=train+valid) vs test 사이 완전/근접 중복 탐지 → test에서 제거
    # 2) train(=train+valid) vs test 사이 완전/근접 중복 탐지 → test에서 제거
    X_train_all = np.vstack([X_tr, X_va])
    X_test_all  = X_te.copy()

    # hashable tuple 로 변환
    X_train_hash = {tuple(np.round(row, ROUND_DECIMALS)) for row in X_train_all.astype(np.float64)}
    test_hash    = [tuple(np.round(r, ROUND_DECIMALS)) for r in X_test_all.astype(np.float64)]

    dup_mask = np.array([h in X_train_hash for h in test_hash])   # True면 중복
    n_dup = int(dup_mask.sum())
    print(f"[INFO] exact/rounded duplicates between train and test: {n_dup}")


    keep = ~dup_mask
    # 3) X-baseline용 스케일러(train만 fit) + dedup된 test 적용
    scaler = QuantileTransformer(output_distribution='normal',
                                 n_quantiles=min(1000, len(X_train_all)),
                                 random_state=SEED)
    scaler.fit(X_train_all)
    Xtr_s = scaler.transform(X_train_all)
    Xte_s = scaler.transform(X_test_all)[keep]
    yte_x = y_te[keep]

    # 4) 저장된 Z와 y 로드 → 같은 방식으로 train/valid concat 후 test를 keep 마스크로 필터
    Z_tr = np.load(os.path.join(Z_DIR, "Z_tr.npy"))
    Z_va = np.load(os.path.join(Z_DIR, "Z_va.npy"))
    Z_te = np.load(os.path.join(Z_DIR, "Z_te.npy"))
    y_tr_z = np.load(os.path.join(Z_DIR, "y_tr.npy"))
    y_va_z = np.load(os.path.join(Z_DIR, "y_va.npy"))
    y_te_z = np.load(os.path.join(Z_DIR, "y_te.npy"))

    Z_train = np.vstack([Z_tr, Z_va])
    y_train = np.concatenate([y_tr_z, y_va_z])
    Z_test  = Z_te[keep]
    yte_z   = y_te_z[keep]

    # sanity check
    assert len(yte_x) == len(yte_z) == Z_test.shape[0] == Xte_s.shape[0], "dedup 크기 불일치"

    # 5) X vs Z 비교 (동일한 XGB 세팅)
    y_tr_all = np.concatenate([y_tr, y_va])   # train+valid 라벨

    _ = run_xgb(Xtr=Xtr_s, Xte=Xte_s, y_tr=y_tr_all, y_te=yte_x,
                name="RawX+XGB (dedup)", seed=SEED)

    _ = run_xgb(Xtr=Z_train, Xte=Z_test, y_tr=y_train, y_te=yte_z,
                name="CLS-Z+XGB (dedup)", seed=SEED)

    # 6) (선택) 셔플 베이스라인 – dedup된 test에서 label 섞기
    rng = np.random.default_rng(42)
    y_shuf = yte_z.copy()
    rng.shuffle(y_shuf)
    print(f"[Shuffle] ROC≈{roc_auc_score(y_shuf, np.ones_like(y_shuf)*y_shuf.mean()):.3f}  "
          f"PR≈{average_precision_score(y_shuf, np.ones_like(y_shuf)*y_shuf.mean()):.3f}  "
          f"(just sanity; not a proper model)")

if __name__ == "__main__":
    main()
