"""
Track A: Cardiovascular Risk Prediction pipeline
Predicts overall CAD (Cath) and LAD / LCX / RCA stenosis.
Target leakage control: LAD, LCX, RCA and Cath are NEVER used as input features.
Outputs: metrics.json, model_params.json (consumed by the 3D viewer), model_weights.joblib
"""
import json, warnings
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
warnings.filterwarnings("ignore")

SEED = 42
import os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "extention_of_Z-Alizadeh_sani_dataset.xlsx")
OUT = HERE + os.sep

# ---------------------------------------------------------------- load
df = pd.read_excel(DATA, sheet_name=0)
df.columns = [c.strip() for c in df.columns]

# ---------------------------------------------------------------- targets
TARGETS = {"CAD": ("Cath", "CAD"), "LAD": ("LAD", "Stenotic"),
           "LCX": ("LCX", "Stenotic"), "RCA": ("RCA", "Stenotic")}
Y = {k: (df[c] == pos).astype(int) for k, (c, pos) in TARGETS.items()}

# ---------------------------------------------------------------- features
LEAK = ["LAD", "LCX", "RCA", "Cath"]
X = df.drop(columns=LEAK).copy()
X["Sex"] = (X["Sex"].str.strip().str.lower() == "male").astype(int)   # 1 = male
vhd_map = {"N": 0, "mild": 1, "Moderate": 2, "Severe": 3}
X["VHD"] = X["VHD"].str.strip().map(lambda s: vhd_map.get(s, vhd_map.get(s.capitalize(), 0)))
for c in X.columns:
    if X[c].dtype == object or str(X[c].dtype).startswith("str"):
        X[c] = (X[c].astype(str).str.strip().str.upper() == "Y").astype(int)   # Y/N -> 1/0
X["Region RWMA"] = pd.to_numeric(X["Region RWMA"], errors="coerce").fillna(0)
X = X.drop(columns=["Weight", "Length"])          # BMI already encodes both (UI computes BMI)
X = X.astype(float)
FEATURES = list(X.columns)
assert not set(LEAK) & set(FEATURES)
assert X.isna().sum().sum() == 0
print(f"{len(FEATURES)} features, {len(X)} patients")

# robust clipping bounds (1st-99th percentile) for outlier-safe inference
lo, hi = X.quantile(0.01), X.quantile(0.99)
Xc = X.clip(lo, hi, axis=1)

# ---------------------------------------------------------------- models
def models(C=0.1, pen="l2"):
    return {
        "LogisticRegression": make_pipeline(StandardScaler(),
            LogisticRegression(C=C, penalty=pen, solver="liblinear", max_iter=5000, class_weight="balanced", random_state=SEED)),
        "RandomForest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
            class_weight="balanced", random_state=SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=150, max_depth=2,
            learning_rate=0.05, subsample=0.8, random_state=SEED),
    }

scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc"}
results = {"cross_validation": {}, "holdout": {}}
final = {}

for tname, y in Y.items():
    print(f"\n=== {tname}  (positive rate {y.mean():.2f}) ===")
    Xtr, Xte, ytr, yte = train_test_split(Xc, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results["cross_validation"][tname], results["holdout"][tname] = {}, {}
    # tune regularisation of the deployable logistic model on the TRAIN split only (nested away from test)
    best, bestauc = (0.1, "l2"), -1
    for pen in ("l1", "l2"):
        for C in (0.01, 0.03, 0.1, 0.3, 1.0):
            a = cross_validate(models(C, pen)["LogisticRegression"], Xtr, ytr, cv=cv, scoring="roc_auc")["test_score"].mean()
            if a > bestauc: best, bestauc = (C, pen), a
    print("best LR setting:", best, round(bestauc, 3))
    results.setdefault("lr_setting", {})[tname] = {"C": best[0], "penalty": best[1]}
    for mname, m in models(*best).items():
        # 5-fold stratified CV on the training split only (no test peeking)
        cvr = cross_validate(m, Xtr, ytr, cv=cv, scoring=scoring)
        results["cross_validation"][tname][mname] = {k: round(float(np.mean(cvr["test_" + k])), 4) for k in scoring} | \
            {"roc_auc_std": round(float(np.std(cvr["test_roc_auc"])), 4)}
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]; yh = (p >= 0.5).astype(int)
        results["holdout"][tname][mname] = {
            "accuracy": round(accuracy_score(yte, yh), 4), "precision": round(precision_score(yte, yh), 4),
            "recall": round(recall_score(yte, yh), 4), "f1": round(f1_score(yte, yh), 4),
            "roc_auc": round(roc_auc_score(yte, p), 4)}
        print(f"{mname:19s} CV AUC {results['cross_validation'][tname][mname]['roc_auc']:.3f} | "
              f"holdout AUC {results['holdout'][tname][mname]['roc_auc']:.3f} "
              f"acc {results['holdout'][tname][mname]['accuracy']:.3f} f1 {results['holdout'][tname][mname]['f1']:.3f}")
    # Deployment model: logistic regression (exactly exportable to the browser, exact SHAP-style contributions)
    lr = models(*results["lr_setting"][tname].values())["LogisticRegression"].fit(Xc, y)       # refit on all data for deployment
    final[tname] = lr

# ---------------------------------------------------------------- export
joblib.dump({"models": final, "features": FEATURES, "clip_lo": lo, "clip_hi": hi}, OUT + "model_weights.joblib")

params = {"features": FEATURES, "clip_lo": lo.round(4).tolist(), "clip_hi": hi.round(4).tolist(),
          "feature_mean_raw": Xc.mean().round(4).tolist(), "targets": {}}
for t, pipe in final.items():
    sc, lr = pipe[0], pipe[1]
    z = sc.transform(Xc)
    contrib = z * lr.coef_[0]
    params["targets"][t] = {
        "mean": sc.mean_.round(6).tolist(), "scale": sc.scale_.round(6).tolist(),
        "coef": lr.coef_[0].round(6).tolist(), "intercept": round(float(lr.intercept_[0]), 6),
        "global_importance": np.abs(contrib).mean(0).round(5).tolist(),
        "base_rate": round(float(Y[t].mean()), 4)}
params["metrics"] = results
json.dump(params, open(OUT + "model_params.json", "w"))
json.dump(results, open(OUT + "metrics.json", "w"), indent=2)

# example patients (for the demo presets)
X.assign(**{k: v for k, v in Y.items()}).to_csv(OUT + "processed_dataset.csv", index=False)
print("\nexported.")
