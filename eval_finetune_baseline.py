import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    roc_curve, precision_recall_curve
)
from tabularbert import TabularBERTTrainer
from tabularbert.utils.data import QuantileDiscretize
import pickle

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT   = "./fine-tuning/version0/model_checkpoint.pt"   
DATA   = "./datasets/hitcall.csv"
SCALER_PKL = "./preprocessing/preprocessing_quantile.pkl" 
NUM_BINS = 50                                          
OUT_DIR = "./eval_finetuned"
os.makedirs(OUT_DIR, exist_ok=True)


# 데이터 

df = pd.read_csv(DATA)
X = df.iloc[:, :-1].to_numpy()
y = pd.Categorical(df.iloc[:, -1]).codes.astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, train_size=0.8, random_state=0, stratify=y
)

# 스케일러: 학습 때와 동일하게
if os.path.exists(SCALER_PKL):
    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)
else:
    scaler = QuantileTransformer(
        n_quantiles=min(10000, len(X_tr)),
        output_distribution="uniform",
        subsample=None,
        random_state=0
    )
    scaler.fit(X_tr)

X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)


# 파인튜닝 모델 로드
model = TabularBERTTrainer.from_finetuned(CKPT, device=DEVICE)
model.eval()  # 중요!


# Binning: 반드시 train으로만 fit 후 test 변환
disc = QuantileDiscretize(num_bins=NUM_BINS, encoding_info=model.encoding_info)
disc.fit(X_tr_s)
B_tr = disc.discretize(X_tr_s)
B_te = disc.discretize(X_te_s)


# 예측 확률 계산 (로짓→소프트맥스)
@torch.no_grad()
def predict_proba(m, bin_ids_np):
    b = torch.tensor(bin_ids_np, dtype=torch.long, device=DEVICE)
    logits = m(b)                       # DownstreamModel.forward -> (N, C)
    probs = F.softmax(logits, dim=-1)   # 2-class 기준
    return probs[:, 1].cpu().numpy()    # positive 클래스 확률

proba_te = predict_proba(model, B_te)

# 지표표
roc = roc_auc_score(y_te, proba_te)
pr  = average_precision_score(y_te, proba_te)
f1  = f1_score(y_te, (proba_te >= 0.5).astype(int))

print(f"[Finetuned TabBERT] ROC-AUC={roc:.4f} | PR-AUC={pr:.4f} | F1@0.5={f1:.4f}")

# 저장
fpr, tpr, _ = roc_curve(y_te, proba_te)
prec, rec, _ = precision_recall_curve(y_te, proba_te)
pd.DataFrame({"FPR": fpr, "TPR": tpr}).to_csv(os.path.join(OUT_DIR, "roc_curve.csv"), index=False)
pd.DataFrame({"Recall": rec, "Precision": prec}).to_csv(os.path.join(OUT_DIR, "pr_curve.csv"), index=False)
meta = {
    "ckpt": os.path.abspath(CKPT),
    "data": os.path.abspath(DATA),
    "num_bins": NUM_BINS,
    "split_random_state": 0,
    "device": str(DEVICE),
    "n_train": int(len(X_tr)),
    "n_test": int(len(X_te)),
    "roc_auc": float(roc),
    "pr_auc": float(pr),
    "f1@0.5": float(f1),
}
pd.DataFrame([meta]).to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
print(f"Saved curves & summary -> {OUT_DIR}")

# 1) 라벨 셔플 테스트
rng = np.random.default_rng(0)
y_te_shuf = rng.permutation(y_te)
from sklearn.metrics import roc_auc_score, average_precision_score
print("[Shuffle] ROC≈", roc_auc_score(y_te_shuf, proba_te),
      " PR≈", average_precision_score(y_te_shuf, proba_te))

# 2) train↔test 완전 중복 여부
import hashlib
def row_hash(a): return hashlib.md5(np.round(a.astype(np.float64),6).tobytes()).hexdigest()
tr_set = set(row_hash(r) for r in X_tr)
te_set = set(row_hash(r) for r in X_te)
dup = tr_set & te_set
print("exact duplicates between train/test:", len(dup))

# 3) train 성능도 확인(너무 높으면 과적합 경향)
proba_tr = predict_proba(model, B_tr)
from sklearn.metrics import f1_score
print("[Train] ROC=", roc_auc_score(y_tr, proba_tr),
      " PR=", average_precision_score(y_tr, proba_tr),
      " F1@0.5=", f1_score(y_tr, (proba_tr>=0.5).astype(int)))

# 4) 양성비(= PR baseline) 확인
pos_rate = y_te.mean()
print("test positive rate (baseline AP) =", float(pos_rate))
