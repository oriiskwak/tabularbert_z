import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    roc_curve, precision_recall_curve)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


DATA_CSV = "./datasets/hitcall.csv"
Z_DIR    = "./z_out"  
OUT_DIR  = "./eval_x_vs_z"
os.makedirs(OUT_DIR, exist_ok=True)

def run_xgb(Xtr, Xte, ytr, yte, name, seed=0):
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

    # threshold 탐색
    thresholds = np.linspace(0, 1, 501)
    f1s = []
    for t in thresholds:
        f1s.append(f1_score(yte, (proba >= t).astype(int), zero_division=0))
    best_idx = int(np.argmax(f1s))
    best_f1, best_thr = float(f1s[best_idx]), float(thresholds[best_idx])

    print(f"[{name}] ROC={roc:.4f} | PR={pr:.4f} | F1@0.5={f1:.4f} | BestF1={best_f1:.4f} @ thr={best_thr:.3f}")

    # 커브
    fpr, tpr, _ = roc_curve(yte, proba)
    prec, rec, _ = precision_recall_curve(yte, proba)

    summary = {
        "Variant": name,
        "ROC_AUC": roc,
        "PR_AUC": pr,
        "F1@0.5": f1,
        "Best_F1": best_f1,
        "Best_Threshold": best_thr
    }
    curves = {"fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec, "proba": proba}
    return summary, curves

def main():
    df = pd.read_csv(DATA_CSV)
    X = df.iloc[:, :-1].to_numpy()
    y = pd.Categorical(df.iloc[:, -1]).codes.astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size=0.8, random_state=0, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, train_size=0.8, random_state=0, stratify=y_tr
    )

    # 원본 피처 X 파이프라인 (train-only fit)
    Xtr_s = X_tr
    Xva_s = X_va
    Xte_s = X_te

    # 뽑아놓은 Z (CLS 임베딩)
    Z_tr = np.load(os.path.join(Z_DIR, "Z_tr.npy"))
    Z_va = np.load(os.path.join(Z_DIR, "Z_va.npy"))
    Z_te = np.load(os.path.join(Z_DIR, "Z_te.npy"))
    y_tr_z = np.load(os.path.join(Z_DIR, "y_tr.npy"))
    y_va_z = np.load(os.path.join(Z_DIR, "y_va.npy"))
    y_te_z = np.load(os.path.join(Z_DIR, "y_te.npy"))

    # sanity check
    assert len(y_tr) == len(y_tr_z) and len(y_va) == len(y_va_z) and len(y_te) == len(y_te_z), \
        "Saved Z splits and re-split labels mismatch. Ensure random_state/stratify identical."
    y_tr, y_va, y_te = y_tr_z, y_va_z, y_te_z

    # X vs Z → XGBoost
    sum_x, curves_x = run_xgb(Xtr_s, Xte_s, y_tr, y_te, name="X+XGB", seed=0)
    sum_z, curves_z = run_xgb(Z_tr,   Z_te,   y_tr, y_te, name="Z+XGB", seed=0)

    # ROC/PR 커브 저장 
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    roc_png = os.path.join(OUT_DIR, f"roc_compare_{ts}.png")
    pr_png  = os.path.join(OUT_DIR, f"pr_compare_{ts}.png")

    # ROC
    plt.figure()
    plt.plot(curves_x["fpr"], curves_x["tpr"], label=f"RawX+XGB (ROC={sum_x['ROC_AUC']:.3f})")
    plt.plot(curves_z["fpr"], curves_z["tpr"], label=f"CLS-Z+XGB (ROC={sum_z['ROC_AUC']:.3f})")
    plt.plot([0,1],[0,1],'--',alpha=0.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve: X vs Z")
    plt.legend(); plt.tight_layout(); plt.savefig(roc_png, dpi=150)

    # PR
    plt.figure()
    plt.plot(curves_x["rec"], curves_x["prec"], label=f"RawX+XGB (PR={sum_x['PR_AUC']:.3f})")
    plt.plot(curves_z["rec"], curves_z["prec"], label=f"CLS-Z+XGB (PR={sum_z['PR_AUC']:.3f})")
    base_ap = (y_te == 1).mean()
    plt.hlines(base_ap, 0, 1, linestyles='--', alpha=0.5, label=f"Baseline AP={base_ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve: X vs Z")
    plt.legend(); plt.tight_layout(); plt.savefig(pr_png, dpi=150)

    # 저장 (csv)
    summary = pd.DataFrame([sum_x, sum_z])
    csv_path = os.path.join(OUT_DIR, f"summary_{ts}.csv")
    xlsx_path = os.path.join(OUT_DIR, f"summary_{ts}.xlsx")
    summary.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        pd.DataFrame({"FPR": curves_x["fpr"], "TPR": curves_x["tpr"]}).to_excel(w, sheet_name="roc_rawx", index=False)
        pd.DataFrame({"FPR": curves_z["fpr"], "TPR": curves_z["tpr"]}).to_excel(w, sheet_name="roc_z", index=False)
        pd.DataFrame({"Recall": curves_x["rec"], "Precision": curves_x["prec"]}).to_excel(w, sheet_name="pr_rawx", index=False)
        pd.DataFrame({"Recall": curves_z["rec"], "Precision": curves_z["prec"]}).to_excel(w, sheet_name="pr_z", index=False)

    print("Saved:")
    print(" -", roc_png)
    print(" -", pr_png)
    print(" -", csv_path)
    print(" -", xlsx_path)

if __name__ == "__main__":
    main()
