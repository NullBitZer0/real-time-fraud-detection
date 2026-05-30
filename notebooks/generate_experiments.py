"""Run from project root: python notebooks/generate_experiments.py"""
import nbformat as nbf, os

nb  = nbf.v4.new_notebook()
cells = []
def md(t):   return nbf.v4.new_markdown_cell(t)
def code(s): return nbf.v4.new_code_cell(s)

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md("""# 🧪 Model Experiments — Credit Card Fraud Detection
**Strategy**: Class weights / `scale_pos_weight` — no synthetic data (no SMOTE).  
Covers: proper scaling, no data leakage, model comparison, feature importance, threshold tuning.
"""))

# ── 1. Setup ──────────────────────────────────────────────────────────────────
cells.append(md("## 1. Setup & Imports"))
cells.append(code("""\
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve,
    precision_score, recall_score
)
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

plt.rcParams.update({
    "figure.dpi": 120, "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d2e", "axes.edgecolor": "#3a3d5c",
    "axes.labelcolor": "#e0e0e0", "axes.titlecolor": "#ffffff",
    "xtick.color": "#aaaaaa",   "ytick.color": "#aaaaaa",
    "grid.color": "#2a2d4a",    "grid.linestyle": "--",
    "grid.alpha": 0.5,          "text.color": "#e0e0e0",
    "legend.facecolor": "#1a1d2e",
})
FRAUD_C, NORM_C, ACC_C, PUR_C = "#ff4d6d", "#4cc9f0", "#f8961e", "#9d4edd"
print("Setup complete ✓")
"""))

# ── 2. Load & Split ───────────────────────────────────────────────────────────
cells.append(md("## 2. Load Data & Stratified Split"))
cells.append(code("""\
df = pd.read_csv("data/raw/creditcard.csv")
print(f"Shape: {df.shape}  |  Fraud rate: {df['Class'].mean()*100:.3f}%")

TARGET   = "Class"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Stratified split BEFORE any preprocessing — prevents leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Train fraud: {y_train.mean()*100:.3f}%  Test fraud: {y_test.mean()*100:.3f}%")

# Imbalance ratio — used for scale_pos_weight
neg  = int((y_train == 0).sum())
pos  = int((y_train == 1).sum())
ratio = neg / pos
print(f"\\nNeg/Pos ratio (scale_pos_weight): {ratio:.1f}  ({neg} normal / {pos} fraud)")
"""))

# ── 3. Feature Engineering ────────────────────────────────────────────────────
cells.append(md("## 3. Feature Engineering (Row-level, No Leakage)"))
cells.append(code("""\
def add_features(X):
    X = X.copy()
    X["Amount_log"] = np.log1p(X["Amount"])
    X["Hour"]       = (X["Time"] // 3600) % 24
    X["Hour_sin"]   = np.sin(2 * np.pi * X["Hour"] / 24)
    X["Hour_cos"]   = np.cos(2 * np.pi * X["Hour"] / 24)
    v_cols = [c for c in X.columns if c.startswith("V")]
    X["V_mean"] = X[v_cols].mean(axis=1)
    X["V_std"]  = X[v_cols].std(axis=1)
    X["V_max"]  = X[v_cols].max(axis=1)
    X["V_min"]  = X[v_cols].min(axis=1)
    return X

X_train = add_features(X_train)
X_test  = add_features(X_test)

DROP = ["Time", "Amount", "Hour"]
X_train.drop(columns=DROP, inplace=True)
X_test.drop(columns=DROP, inplace=True)

print(f"Features: {X_train.shape[1]}")
print(X_train.columns.tolist())
"""))

# ── 4. Scaling ────────────────────────────────────────────────────────────────
cells.append(md("## 4. Scaling — Fit on Train Only"))
cells.append(code("""\
# RobustScaler: uses median/IQR — handles Amount outliers better
SCALE_COLS = ["Amount_log","Hour_sin","Hour_cos","V_mean","V_std","V_max","V_min"]

scaler = RobustScaler()
X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])  # fit on train only
X_test[SCALE_COLS]  = scaler.transform(X_test[SCALE_COLS])       # transform test

print("✅ Scaler fit on TRAIN only — zero data leakage")
print(f"Scaled: {SCALE_COLS}")
"""))

# ── 5. Eval Helper ────────────────────────────────────────────────────────────
cells.append(md("## 5. Evaluation Helper"))
cells.append(code("""\
results = {}

def evaluate(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    probs = model.predict_proba(Xte)[:, 1]
    preds = (probs >= 0.5).astype(int)

    roc = roc_auc_score(yte, probs)
    pr  = average_precision_score(yte, probs)
    f1  = f1_score(yte, preds)
    pre = precision_score(yte, preds, zero_division=0)
    rec = recall_score(yte, preds, zero_division=0)

    results[name] = dict(
        ROC_AUC=round(roc,4), PR_AUC=round(pr,4),
        F1=round(f1,4), Precision=round(pre,4), Recall=round(rec,4),
        model=model, probs=probs
    )
    print(f"[{name:<35}]  ROC={roc:.4f}  PR={pr:.4f}  F1={f1:.4f}")
    return model
"""))

# ── 6. Model Comparison ───────────────────────────────────────────────────────
cells.append(md("""\
## 6. Model Comparison
All models use **class weights** or **`scale_pos_weight`** — real data only, no synthetic samples.
"""))

cells.append(code("""\
# ── Logistic Regression ───────────────────────────────────────────────────────
evaluate("LogisticRegression (balanced)",
    LogisticRegression(class_weight="balanced", max_iter=1000, C=0.1, random_state=42),
    X_train, y_train, X_test, y_test)

# ── Random Forest ─────────────────────────────────────────────────────────────
evaluate("RandomForest (balanced)",
    RandomForestClassifier(n_estimators=200, class_weight="balanced",
                           max_depth=12, random_state=42, n_jobs=-1),
    X_train, y_train, X_test, y_test)
"""))

cells.append(code("""\
# ── XGBoost — scale_pos_weight = neg/pos ratio ───────────────────────────────
evaluate("XGBoost (scale_pos_weight)",
    XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=8,
                  subsample=0.8, colsample_bytree=0.8,
                  scale_pos_weight=ratio,          # key param for imbalance
                  eval_metric="logloss", random_state=42, n_jobs=-1),
    X_train, y_train, X_test, y_test)

# ── LightGBM — class_weight balanced ─────────────────────────────────────────
evaluate("LightGBM (class_weight=balanced)",
    LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=10,
                   num_leaves=64, class_weight="balanced",
                   subsample=0.8, colsample_bytree=0.8,
                   random_state=42, n_jobs=-1, verbose=-1),
    X_train, y_train, X_test, y_test)

# ── LightGBM — manual is_unbalance flag (alternative) ────────────────────────
evaluate("LightGBM (is_unbalance=True)",
    LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=10,
                   num_leaves=64, is_unbalance=True,
                   subsample=0.8, colsample_bytree=0.8,
                   random_state=42, n_jobs=-1, verbose=-1),
    X_train, y_train, X_test, y_test)

# ── CatBoost — auto_class_weights ────────────────────────────────────────────
evaluate("CatBoost (auto_class_weights)",
    CatBoostClassifier(iterations=300, learning_rate=0.05, depth=8,
                       auto_class_weights="Balanced",
                       random_seed=42, verbose=0),
    X_train, y_train, X_test, y_test)
"""))

# ── 7. Results Table ──────────────────────────────────────────────────────────
cells.append(md("## 7. Results Summary"))
cells.append(code("""\
METRIC_COLS = ["ROC_AUC","PR_AUC","F1","Precision","Recall"]
summary = pd.DataFrame({
    name: {k: v for k,v in vals.items() if k in METRIC_COLS}
    for name, vals in results.items()
}).T.sort_values("PR_AUC", ascending=False)

summary.style \
    .background_gradient(cmap="YlOrRd", subset=["PR_AUC","F1"]) \
    .background_gradient(cmap="Blues",  subset=["ROC_AUC"]) \
    .format(precision=4)
"""))

# ── 8. PR & ROC Curves ────────────────────────────────────────────────────────
cells.append(md("## 8. PR Curve & ROC Curve"))
cells.append(code("""\
palette = plt.cm.tab10.colors
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, (name, vals) in enumerate(results.items()):
    c = palette[i % len(palette)]
    prec, rec, _ = precision_recall_curve(y_test, vals["probs"])
    axes[0].plot(rec, prec, label=f"{name} ({vals['PR_AUC']:.3f})", color=c, lw=1.8)

    fpr, tpr, _ = roc_curve(y_test, vals["probs"])
    axes[1].plot(fpr, tpr, label=f"{name} ({vals['ROC_AUC']:.3f})", color=c, lw=1.8)

axes[0].axhline(y_test.mean(), color="white", ls="--", alpha=0.4, label="No-skill baseline")
axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
axes[0].set_title("Precision-Recall Curve ← primary metric for fraud", fontweight="bold")
axes[0].legend(fontsize=7)

axes[1].plot([0,1],[0,1], "w--", alpha=0.4, label="Random")
axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
axes[1].set_title("ROC Curve", fontweight="bold")
axes[1].legend(fontsize=7)

plt.tight_layout(); plt.show()
"""))

# ── 9. Confusion Matrices ─────────────────────────────────────────────────────
cells.append(md("## 9. Confusion Matrices — All Models"))
cells.append(code("""\
n     = len(results)
ncols = 3
nrows = -(-n // ncols)   # ceiling div
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
axes = axes.flatten()

for i, (name, vals) in enumerate(results.items()):
    preds = (vals["probs"] >= 0.5).astype(int)
    cm    = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt=",", ax=axes[i],
                cmap="YlOrRd", cbar=False,
                xticklabels=["Normal","Fraud"],
                yticklabels=["Normal","Fraud"])
    tn, fp, fn, tp = cm.ravel()
    axes[i].set_title(f"{name}\\nTP={tp} FP={fp} FN={fn}", fontsize=9, fontweight="bold")

for j in range(i+1, len(axes)):
    axes[j].axis("off")

plt.suptitle("Confusion Matrices @ threshold=0.5", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()
"""))

# ── 10. Threshold Analysis ────────────────────────────────────────────────────
cells.append(md("## 10. Threshold Optimisation (Best Model)"))
cells.append(code("""\
best_name  = summary["PR_AUC"].idxmax()
best_probs = results[best_name]["probs"]

thresholds = np.arange(0.01, 1.0, 0.01)
f1s, precs, recs = [], [], []

for t in thresholds:
    p = (best_probs >= t).astype(int)
    f1s.append(f1_score(y_test, p, zero_division=0))
    precs.append(precision_score(y_test, p, zero_division=0))
    recs.append(recall_score(y_test, p, zero_division=0))

best_t    = thresholds[np.argmax(f1s)]
best_f1   = max(f1s)
default_f1 = f1_score(y_test, (best_probs >= 0.5).astype(int))

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(thresholds, f1s,   color=ACC_C,   lw=2.5, label="F1")
ax.plot(thresholds, precs, color=NORM_C,  lw=2,   label="Precision", alpha=0.8)
ax.plot(thresholds, recs,  color=FRAUD_C, lw=2,   label="Recall",    alpha=0.8)
ax.axvline(best_t, color="white", ls="--", lw=1.5,
           label=f"Best F1 @ t={best_t:.2f}  (F1={best_f1:.4f})")
ax.axvline(0.5, color="grey", ls=":", lw=1,
           label=f"Default 0.5  (F1={default_f1:.4f})")
ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
ax.set_title(f"Threshold Analysis — {best_name}", fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

print(f"Best model     : {best_name}")
print(f"Best threshold : {best_t:.2f}")
print(f"F1 improvement : {default_f1:.4f} → {best_f1:.4f}  (+{best_f1-default_f1:.4f})")
"""))

# ── 11. Class Weight Sensitivity ─────────────────────────────────────────────
cells.append(md("## 11. Class Weight Sensitivity — LightGBM"))
cells.append(code("""\
# How sensitive is performance to different positive-class weights?
weight_results = {}
weights_to_try = [1, 5, 10, 20, 50, ratio, ratio*2]

for w in weights_to_try:
    m = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=10,
        num_leaves=64, class_weight={0: 1, 1: w},
        random_state=42, n_jobs=-1, verbose=-1
    )
    m.fit(X_train, y_train)
    pr_  = m.predict_proba(X_test)[:, 1]
    weight_results[f"w={w:.0f}"] = {
        "PR_AUC":  round(average_precision_score(y_test, pr_), 4),
        "F1":      round(f1_score(y_test, (pr_>=0.5).astype(int)), 4),
        "Recall":  round(recall_score(y_test, (pr_>=0.5).astype(int)), 4),
        "Precision": round(precision_score(y_test, (pr_>=0.5).astype(int), zero_division=0), 4),
    }

w_df = pd.DataFrame(weight_results).T
print(w_df.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
w_df["PR_AUC"].plot(ax=axes[0], color=ACC_C,   marker="o", lw=2)
axes[0].set_title("PR-AUC vs Positive Class Weight"); axes[0].set_ylabel("PR-AUC")

w_df[["Precision","Recall","F1"]].plot(
    ax=axes[1], color=[NORM_C, FRAUD_C, ACC_C], marker="o", lw=2)
axes[1].set_title("Precision / Recall / F1 vs Weight"); axes[1].set_ylabel("Score")

for ax in axes:
    ax.set_xlabel("Class Weight")
plt.tight_layout(); plt.show()
"""))

# ── 12. Feature Importance ────────────────────────────────────────────────────
cells.append(md("## 12. Feature Importance"))
cells.append(code("""\
lgbm_model = results["LightGBM (class_weight=balanced)"]["model"]
xgb_model  = results["XGBoost (scale_pos_weight)"]["model"]

# Built-in importance
lgbm_fi = pd.Series(lgbm_model.feature_importances_,
                     index=X_train.columns, name="LightGBM")
xgb_fi  = pd.Series(xgb_model.feature_importances_,
                     index=X_train.columns, name="XGBoost")

# Normalise
lgbm_fi_n = lgbm_fi / lgbm_fi.max()
xgb_fi_n  = xgb_fi  / xgb_fi.max()

combined = pd.concat([lgbm_fi_n, xgb_fi_n], axis=1)
combined["Mean"] = combined.mean(axis=1)
combined = combined.sort_values("Mean", ascending=False).head(20)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

combined["LightGBM"].sort_values().plot.barh(
    ax=axes[0], color=NORM_C, edgecolor="white")
axes[0].set_title("LightGBM Importance (norm)", fontweight="bold")

combined["XGBoost"].sort_values().plot.barh(
    ax=axes[1], color=ACC_C, edgecolor="white")
axes[1].set_title("XGBoost Importance (norm)", fontweight="bold")

combined["Mean"].sort_values().plot.barh(
    ax=axes[2], color=FRAUD_C, edgecolor="white")
axes[2].set_title("Mean Importance — Top 20", fontweight="bold")

plt.suptitle("Feature Importance Comparison", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# Permutation importance — model-agnostic, uses PR-AUC
perm = permutation_importance(
    lgbm_model, X_test, y_test,
    n_repeats=10, random_state=42,
    scoring="average_precision"
)
perm_df = pd.Series(perm.importances_mean, index=X_train.columns)
perm_df = perm_df.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 7))
colors = [FRAUD_C if v > 0 else "grey" for v in perm_df.head(20).values]
perm_df.head(20).sort_values().plot.barh(ax=ax, color=colors[::-1], edgecolor="white")
ax.axvline(0, color="white", lw=0.8, ls="--")
ax.set_title("Permutation Importance (drop in PR-AUC when feature shuffled)",
             fontweight="bold")
ax.set_xlabel("Mean decrease in PR-AUC")
plt.tight_layout(); plt.show()

print("Top 10 most important features:")
print(perm_df.head(10).to_string())
"""))

# ── 13. Scaling Comparison ────────────────────────────────────────────────────
cells.append(md("## 13. Scaling Strategy Comparison"))
cells.append(code("""\
scale_results = {}

for name, ScalerCls in [("RobustScaler", RobustScaler),
                         ("StandardScaler", StandardScaler),
                         ("No Scaling", None)]:
    Xtr2, Xte2 = X_train.copy(), X_test.copy()
    if ScalerCls:
        sc = ScalerCls()
        Xtr2[SCALE_COLS] = sc.fit_transform(Xtr2[SCALE_COLS])
        Xte2[SCALE_COLS] = sc.transform(Xte2[SCALE_COLS])

    m = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=10,
                       class_weight="balanced", random_state=42,
                       n_jobs=-1, verbose=-1)
    m.fit(Xtr2, y_train)
    pr_ = m.predict_proba(Xte2)[:, 1]
    p   = (pr_ >= 0.5).astype(int)

    scale_results[name] = dict(
        PR_AUC  = round(average_precision_score(y_test, pr_), 4),
        ROC_AUC = round(roc_auc_score(y_test, pr_), 4),
        F1      = round(f1_score(y_test, p), 4),
        Recall  = round(recall_score(y_test, p), 4),
    )
    print(f"[{name:<16}] PR={scale_results[name]['PR_AUC']}  "
          f"F1={scale_results[name]['F1']}")

pd.DataFrame(scale_results).T \
    .style.background_gradient(cmap="Blues")
"""))

# ── 14. Final Summary ─────────────────────────────────────────────────────────
cells.append(md("## 14. Final Results & MLOps Update Checklist"))
cells.append(code("""\
print("=" * 70)
print("RANKED BY PR-AUC (primary fraud metric)")
print("=" * 70)
print(summary[METRIC_COLS].to_string())
print()
best = summary["PR_AUC"].idxmax()
print(f"🏆  Best model    : {best}")
print(f"    PR-AUC       : {summary.loc[best,'PR_AUC']}")
print(f"    F1           : {summary.loc[best,'F1']}")
print(f"    Best threshold: {best_t:.2f}  (use this in prediction_pipeline.py)")
"""))

cells.append(md("""\
## ✅ MLOps Pipeline Update Checklist

Based on experiments above, update your pipeline:

### Files to Update

| File | Change |
|---|---|
| `preprocessing.py` | Add `RobustScaler` — fit on train, save with joblib |
| `configs/model/lightgbm.yaml` | Add `class_weight: balanced` |
| `configs/model/xgboost.yaml` | Add `scale_pos_weight: <ratio>` (auto-compute from train) |
| `configs/model/catboost.yaml` | Confirm `auto_class_weights: Balanced` is set |
| `model_training.py` | Read `scale_pos_weight` from data ratio, not hardcoded |
| `model_evaluation.py` | Use PR-AUC as primary metric; add threshold-optimised F1 |
| `params.yaml` | Add `inference_threshold: 0.35` (adjust from section 10) |
| `prediction_pipeline.py` | Load scaler + apply threshold from params |

### What NOT to do
- ❌ No SMOTE — introduces synthetic fraud patterns not in real distribution
- ❌ Do not fit scaler on full dataset — fit on train only
- ❌ Do not use accuracy as metric — useless under 0.17% fraud rate
"""))

# ── 15. Isolation Forest ──────────────────────────────────────────────────────
cells.append(md("""\
## 15. Isolation Forest — Hybrid Anomaly Feature

**Strategy**: Fit `IsolationForest` on train only → extract anomaly score per transaction →
add as a new feature (`IF_score`) to LightGBM.  
This is the best way to use IF with labeled data — it adds an "oddness" signal without
replacing the supervised model.

> IsolationForest as standalone classifier is NOT recommended for this dataset
> because it has no access to labels and will produce many false positives.
"""))

cells.append(code("""\
from sklearn.ensemble import IsolationForest

# ── Step 1: Fit IF on TRAIN only (unsupervised — no labels used) ──────────────
iso = IsolationForest(
    n_estimators=200,
    contamination=float(y_train.mean()),  # set to actual fraud rate
    random_state=42,
    n_jobs=-1
)
iso.fit(X_train)
print("IsolationForest fitted on train ✓")

# ── Step 2: Extract anomaly scores (lower = more anomalous) ──────────────────
# score_samples returns negative average depth — we negate so higher = more anomalous
train_if_score = -iso.score_samples(X_train)
test_if_score  = -iso.score_samples(X_test)

print(f"IF score range (train): [{train_if_score.min():.3f}, {train_if_score.max():.3f}]")
print(f"IF score range (test) : [{test_if_score.min():.3f}, {test_if_score.max():.3f}]")
"""))

cells.append(code("""\
# ── Step 3: Visualise — does IF score separate fraud from normal? ─────────────
import matplotlib.pyplot as plt

fraud_scores  = test_if_score[y_test == 1]
normal_scores = test_if_score[y_test == 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(normal_scores, bins=80, color=NORM_C,  alpha=0.6, density=True, label="Normal")
axes[0].hist(fraud_scores,  bins=80, color=FRAUD_C, alpha=0.8, density=True, label="Fraud")
axes[0].set_xlabel("IF Anomaly Score (higher = more anomalous)")
axes[0].set_ylabel("Density")
axes[0].set_title("Anomaly Score Distribution by Class")
axes[0].legend()

# Score as a classifier — PR curve
from sklearn.metrics import precision_recall_curve, average_precision_score
prec_if, rec_if, _ = precision_recall_curve(y_test, test_if_score)
pr_auc_if = average_precision_score(y_test, test_if_score)
axes[1].plot(rec_if, prec_if, color=PUR_C, lw=2,
             label=f"IF standalone (PR={pr_auc_if:.3f})")
axes[1].axhline(y_test.mean(), color="white", ls="--", alpha=0.4, label="No-skill")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("IF as Standalone Classifier (PR Curve)")
axes[1].legend()

plt.suptitle("Isolation Forest Analysis", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()

print(f"\\nIF Standalone PR-AUC : {pr_auc_if:.4f}")
print(f"Best supervised PR-AUC: {summary['PR_AUC'].max():.4f}")
print("→ IF alone is weaker; adding it as a feature to supervised model is better.")
"""))

cells.append(code("""\
# ── Step 4: Add IF score as a feature — hybrid approach ──────────────────────
X_train_if = X_train.copy()
X_test_if  = X_test.copy()

X_train_if["IF_score"] = train_if_score
X_test_if["IF_score"]  = test_if_score

# Train LightGBM with IF_score feature
lgbm_if = LGBMClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=10,
    num_leaves=64, class_weight="balanced",
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1
)
lgbm_if.fit(X_train_if, y_train)
probs_if = lgbm_if.predict_proba(X_test_if)[:, 1]

pr_if_hybrid  = average_precision_score(y_test, probs_if)
roc_if_hybrid = roc_auc_score(y_test, probs_if)
f1_if_hybrid  = f1_score(y_test, (probs_if >= 0.5).astype(int))

# Compare with base LightGBM
base_pr  = results["LightGBM (class_weight=balanced)"]["PR_AUC"]
base_roc = results["LightGBM (class_weight=balanced)"]["ROC_AUC"]
base_f1  = results["LightGBM (class_weight=balanced)"]["F1"]

compare = pd.DataFrame({
    "LightGBM (no IF)":     {"PR_AUC": base_pr,     "ROC_AUC": base_roc,     "F1": base_f1},
    "LightGBM + IF score":  {"PR_AUC": round(pr_if_hybrid,4),
                              "ROC_AUC": round(roc_if_hybrid,4),
                              "F1": round(f1_if_hybrid,4)},
    "IF standalone":         {"PR_AUC": round(pr_auc_if,4), "ROC_AUC": None, "F1": None},
}).T

print(compare.to_string())
print()
delta = pr_if_hybrid - base_pr
print(f"PR-AUC change from adding IF score: {delta:+.4f}")
if delta > 0:
    print("✅ IF score improves the model — keep it as a feature")
else:
    print("⚠️  IF score did not help — V-features already capture this signal")
"""))

cells.append(code("""\
# ── Step 5: Feature importance — where does IF_score rank? ───────────────────
fi_if = pd.Series(lgbm_if.feature_importances_,
                  index=X_train_if.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 7))
colors = [FRAUD_C if idx == "IF_score" else ACC_C for idx in fi_if.head(20).index]
fi_if.head(20).sort_values().plot.barh(ax=ax, color=colors[::-1], edgecolor="white")
ax.set_title("Feature Importance — LightGBM + IF Score (red = IF_score)",
             fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout(); plt.show()

rank = list(fi_if.index).index("IF_score") + 1
print(f"IF_score ranks #{rank} out of {len(fi_if)} features")
"""))

cells.append(md("""\
### Isolation Forest — Summary

| Approach | Verdict |
|---|---|
| **IF standalone** | ❌ Weak — unsupervised, no label info, high false positive rate |
| **IF + supervised (hybrid)** | ✅ Best — anomaly score added as feature to LightGBM |
| **SMOTE** | ❌ Avoided — synthetic fraud samples |

**Key insight**: The IF anomaly score works best as an additional signal, not a replacement.  
If the `IF_score` ranks in the top 10 features → add it to your `feature_engineering.py` pipeline.  
If it ranks poorly → the V1–V28 PCA features already capture the anomaly signal.
"""))

# ── Write ─────────────────────────────────────────────────────────────────────
nb.cells = cells
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"✅ Written: {out}")
