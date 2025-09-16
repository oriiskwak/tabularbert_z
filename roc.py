import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tabularbert import TabularBERTTrainer
from tabularbert.utils.data import QuantileDiscretize

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CKPT = "./fine-tuning/version0/model_checkpoint.pt"

# 1) data & split (학습 때와 동일하게)
df = pd.read_csv("./datasets/hitcall.csv")
X = df.iloc[:, :-1].to_numpy()
y = pd.Categorical(df.iloc[:, -1]).codes.astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, train_size=0.8, random_state=0, stratify=y
)

# 2) ckpt 로드 → 모델/인코딩 정보 얻기
obj = TabularBERTTrainer.from_finetuned(CKPT, device=DEVICE)
model = getattr(obj, "model", obj)
if hasattr(model, "eval"):
    model.eval()
enc_info = getattr(obj, "encoding_info", getattr(model, "encoding_info", None))

# 3) binning: ckpt의 encoding_info가 있으면 그대로 사용(절대 refit X)
#    없으면 fallback으로 train-only fit
def make_bins(X_tr, X_te):
    # num_bins 추출
    num_bins = 50
    try:
        if isinstance(enc_info, dict) and "num_bins" in enc_info:
            num_bins = int(enc_info["num_bins"])
        elif hasattr(enc_info, "num_bins"):
            num_bins = int(enc_info.num_bins)
        elif hasattr(model, "num_bins"):
            num_bins = int(model.num_bins)
    except Exception:
        pass

    disc = QuantileDiscretize(num_bins=num_bins, encoding_info=enc_info)

    if enc_info is None:
        # ckpt에 경계가 없으면 train-only fit
        disc.fit(X_tr)

    B_tr = disc.discretize(X_tr)
    B_te = disc.discretize(X_te)
    return B_tr, B_te, num_bins

B_tr, B_te, nb = make_bins(X_tr, X_te)

@torch.no_grad()
def predict_logits(model, bin_ids_np, pool="cls"):
    b = torch.tensor(bin_ids_np, dtype=torch.long, device=DEVICE)
    emb = model.embedding(b)          # (N, F+1, D)
    contextual = model.bert(emb)      # (N, F+1, D)
    if pool == "cls":
        feat = contextual[:, 0, :]
    else:  # mean pooling over feature tokens
        feat = contextual[:, 1:, :].mean(dim=1)
    logits = model.head(feat)         # (N, C)
    return logits

def auc_of(logits, y_true, pos_col):
    probs = F.softmax(logits, dim=-1)[:, pos_col].detach().cpu().numpy()
    return roc_auc_score(y_true, probs)

# 4) 네 가지 조합 점검
for pool in ["cls", "mean"]:
    logits = predict_logits(model, B_te, pool=pool)
    for pos_col in [0, 1]:
        auc = auc_of(logits, y_te, pos_col=pos_col)
        print(f"[CHECK] pool={pool:4s} | pos_col={pos_col} | ROC-AUC={auc:.4f}")
