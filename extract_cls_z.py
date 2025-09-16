import os
import numpy as np, torch, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from tabularbert import TabularBERTTrainer
from tabularbert.utils.data import QuantileDiscretize

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CKPT   = "./fine-tuning/version0/model_checkpoint.pt"
NUM_BINS = 50
OUT_DIR = "./z_out"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) 데이터 & 네가 학습 때 쓰던 동일 split/전처리
df = pd.read_csv("./datasets/hitcall.csv")
X = df.iloc[:, :-1].to_numpy()
y = pd.Categorical(df.iloc[:, -1]).codes.astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.8, random_state=0, stratify=y)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, train_size=0.8, random_state=0, stratify=y_tr)

scaler = QuantileTransformer(n_quantiles=10000, output_distribution='uniform', subsample=None)
scaler.fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_va_s = scaler.transform(X_va)
X_te_s = scaler.transform(X_te)

# 2) 파인튜닝된 모델 로드
obj = TabularBERTTrainer.from_finetuned(CKPT, device=DEVICE)
model = getattr(obj, "model", obj)
model.eval()

# 3) binning: 반드시 train만 fit → 동일 규칙으로 토큰화
disc = QuantileDiscretize(num_bins=NUM_BINS, encoding_info=None)
disc.fit(X_tr_s)
B_tr = disc.discretize(X_tr_s)
B_va = disc.discretize(X_va_s)
B_te = disc.discretize(X_te_s)

@torch.no_grad()
def extract_z(model, bins_np, pool="cls"):
    b = torch.tensor(bins_np, dtype=torch.long, device=DEVICE)
    emb = model.embedding(b)          # (N, F+1, D)  [CLS] 포함
    h   = model.bert(emb)             # (N, F+1, D)
    if pool == "cls":
        return h[:, 0, :].detach().cpu().numpy()
    else:
        return h[:, 1:, :].mean(dim=1).detach().cpu().numpy()

Z_tr = extract_z(model, B_tr, pool="cls")   # 또는 pool="mean"
Z_va = extract_z(model, B_va, pool="cls")
Z_te = extract_z(model, B_te, pool="cls")

print("Z shapes:", Z_tr.shape, Z_va.shape, Z_te.shape)



# --------------------
# 6) 저장
# --------------------
np.save(os.path.join(OUT_DIR, "Z_tr.npy"), Z_tr)
np.save(os.path.join(OUT_DIR, "Z_va.npy"), Z_va)
np.save(os.path.join(OUT_DIR, "Z_te.npy"), Z_te)

np.save(os.path.join(OUT_DIR, "y_tr.npy"), y_tr)
np.save(os.path.join(OUT_DIR, "y_va.npy"), y_va)
np.save(os.path.join(OUT_DIR, "y_te.npy"), y_te)

print(f"Z and labels saved in {OUT_DIR}")


# h = BERT encoder가 학습해서 만든 문맥화된 hidden states (임베딩 공간의 표현)
# 그 중 h[:,0,:] = CLS hidden vector = 전체 feature들의 집약 표현
# hidden space 안에서 CLS 토큰 벡터만 뽑아서 downstream task(XGBoost 등)에 쓰는 거.

# Z는 2차원 행렬 ( N * D ) 