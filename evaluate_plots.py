"""Generates evaluation figures and a results table for the report.
Run AFTER train_pipeline.py. Uses the same split (seed 42) as the pipeline, so hold out numbers match metrics.json."""
import os, json, warnings, numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.calibration import calibration_curve
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); PL = os.path.join(HERE, "plots"); os.makedirs(PL, exist_ok=True)
SEED = 42; T = ["CAD", "LAD", "LCX", "RCA"]
M = joblib.load(os.path.join(HERE, "model_weights.joblib")); F = M["features"]
res = json.load(open(os.path.join(HERE, "metrics.json")))
d = pd.read_csv(os.path.join(HERE, "processed_dataset.csv"))
X = d[F].clip(M["clip_lo"], M["clip_hi"], axis=1)

def lr(t):
    s = res["lr_setting"][t]
    return make_pipeline(StandardScaler(), LogisticRegression(C=s["C"], penalty=s["penalty"], solver="liblinear",
                         max_iter=5000, class_weight="balanced", random_state=SEED))
def models(t):
    return {"Logistic regression (deployed)": lr(t),
            "Random forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1),
            "Gradient boosting": GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED)}

def boot_auc(y, p, n=2000):
    r = np.random.RandomState(SEED); a = []
    y = np.asarray(y)
    for _ in range(n):
        i = r.randint(0, len(y), len(y))
        if y[i].min() != y[i].max(): a.append(roc_auc_score(y[i], p[i]))
    return np.percentile(a, [2.5, 97.5])

fig_roc, ax_roc = plt.subplots(1, 4, figsize=(17, 4.2))
fig_cm, ax_cm = plt.subplots(1, 4, figsize=(15, 3.8))
fig_cal, ax_cal = plt.subplots(1, 4, figsize=(15, 3.8))
rows = []
for k, t in enumerate(T):
    y = d[t]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    for name, m in models(t).items():
        m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, p); auc = roc_auc_score(yte, p)
        ax_roc[k].plot(fpr, tpr, lw=2.4 if "deployed" in name else 1.3, label=f"{name.split(' (')[0]} ({auc:.2f})")
        if "deployed" in name:
            lo, hi = boot_auc(yte, p); yh = (p >= 0.5).astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(yte, yh, average="binary")
            ConfusionMatrixDisplay(confusion_matrix(yte, yh), display_labels=["No", "Yes"]).plot(ax=ax_cm[k], colorbar=False, cmap="Blues")
            ax_cm[k].set_title(f"{t} (n={len(yte)})")
            rows.append({"Target": t, "Test n": len(yte), "Positive rate": round(y.mean(), 2), "ROC-AUC": round(auc, 3),
                         "AUC 95% CI": f"{lo:.2f} to {hi:.2f}", "Accuracy": round((yh == yte).mean(), 3),
                         "Precision": round(pr, 3), "Recall": round(rc, 3), "F1": round(f1, 3)})
    ax_roc[k].plot([0, 1], [0, 1], "k--", lw=.8); ax_roc[k].set_title(f"{t} ROC (hold out)")
    ax_roc[k].set_xlabel("False positive rate"); ax_roc[k].set_ylabel("True positive rate"); ax_roc[k].legend(fontsize=8, loc="lower right")
    # calibration with out of fold predictions on all 303 patients (more stable than 61 test patients)
    oof = cross_val_predict(lr(t), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    fp, mp = calibration_curve(y, oof, n_bins=6, strategy="quantile")
    ax_cal[k].plot(mp, fp, "o-", label="Logistic regression"); ax_cal[k].plot([0, 1], [0, 1], "k--", lw=.8, label="Perfect")
    ax_cal[k].set_title(f"{t} calibration (5 fold out of fold)"); ax_cal[k].set_xlabel("Mean predicted score"); ax_cal[k].set_ylabel("Observed fraction positive")
    ax_cal[k].legend(fontsize=8)
for f, n in [(fig_roc, "roc_curves.png"), (fig_cm, "confusion_matrices.png"), (fig_cal, "calibration.png")]:
    f.tight_layout(); f.savefig(os.path.join(PL, n), dpi=150)

# global feature importance (mean absolute SHAP value of the linear model) for each target
P = json.load(open(os.path.join(HERE, "model_params.json")))
fig, ax = plt.subplots(1, 4, figsize=(18, 5.2))
for k, t in enumerate(T):
    imp = pd.Series(P["targets"][t]["global_importance"], index=F).sort_values().tail(12)
    ax[k].barh(imp.index, imp.values, color="#c0392b"); ax[k].set_title(f"{t}: mean |SHAP|"); ax[k].tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(os.path.join(PL, "feature_importance.png"), dpi=150)

tab = pd.DataFrame(rows); tab.to_csv(os.path.join(HERE, "results_table.csv"), index=False)
open(os.path.join(HERE, "results_table.md"), "w").write(tab.to_markdown(index=False) if hasattr(tab, "to_markdown") else tab.to_string(index=False))
print(tab.to_string(index=False))
