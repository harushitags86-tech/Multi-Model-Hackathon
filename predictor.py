"""Prediction service logic (framework independent, unit testable).
Runs the trained logistic regression models from model_params.json (plain numbers), so it needs only numpy and pandas:
no scikit-learn or pickle files, and no dependence on the Python version used for training.
Returns probabilities plus exact SHAP values.
For a linear model with a mean background, the SHAP value of a feature is coef * standardised value,
so no extra library is required and the numbers match shap.LinearExplainer."""
import os, json
import numpy as np, pandas as pd

TARGETS = ["CAD", "LAD", "LCX", "RCA"]
GROUPS = [
    ("Demographics", ["Age", "Sex", "BMI"]),
    ("History and risk factors", ["DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Obesity", "CRF", "CVA",
                                  "Airway disease", "Thyroid Disease", "CHF", "DLP"]),
    ("Physical examination", ["BP", "PR", "Edema", "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur"]),
    ("Symptoms", ["Typical Chest Pain", "Dyspnea", "Function Class", "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang"]),
    ("ECG", ["Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression", "BBB"]),
    ("Laboratory", ["FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT"]),
    ("Echocardiography", ["EF-TTE", "Region RWMA", "VHD"]),
]
LABEL = {"Typical Chest Pain": "Typical chest pain", "EF-TTE": "Ejection fraction (echo)", "Region RWMA": "Region with wall motion abnormality (RWMA)",
 "St Depression": "ST depression", "St Elevation": "ST elevation", "Tinversion": "T inversion", "Current Smoker": "Current smoker",
 "EX-Smoker": "Ex smoker", "FH": "Family history of CAD", "DM": "Diabetes mellitus", "HTN": "Hypertension", "BP": "Systolic blood pressure",
 "PR": "Pulse rate", "FBS": "Fasting blood sugar", "LDL": "LDL cholesterol", "HDL": "HDL cholesterol", "TG": "Triglycerides", "ESR": "ESR",
 "Function Class": "Functional class", "Q Wave": "Q wave", "BMI": "Body mass index", "Sex": "Sex", "DLP": "Dyslipidemia", "CHF": "Congestive heart failure",
 "CRF": "Chronic renal failure", "LVH": "Left ventricular hypertrophy", "VHD": "Valvular heart disease", "WBC": "White blood cells", "PLT": "Platelets",
 "HB": "Hemoglobin", "CR": "Creatinine", "BUN": "Blood urea nitrogen", "Na": "Sodium", "K": "Potassium", "Lymph": "Lymphocytes %", "Neut": "Neutrophils %",
 "LowTH Ang": "Low threshold angina", "Exertional CP": "Exertional chest pain", "Atypical": "Atypical chest pain", "Nonanginal": "Non anginal chest pain",
 "CVA": "Stroke history", "BBB": "Bundle branch block", "Airway disease": "Airway disease", "Thyroid Disease": "Thyroid disease", "Weak Peripheral Pulse": "Weak peripheral pulse",
 "Lung rales": "Lung rales", "Systolic Murmur": "Systolic murmur", "Diastolic Murmur": "Diastolic murmur", "Poor R Progression": "Poor R wave progression"}
# measured physiological values shown in the "measurements vs contribution" table: unit and general adult reference (orientation only)
PHYSIO = {"Age": ("years", None, None), "BMI": ("kg/m2", 18.5, 24.9), "BP": ("mmHg", 90, 120), "PR": ("bpm", 60, 100),
 "FBS": ("mg/dL", 70, 100), "LDL": ("mg/dL", None, 130), "HDL": ("mg/dL", 40, None), "TG": ("mg/dL", None, 150), "EF-TTE": ("%", 55, 70),
 "CR": ("mg/dL", 0.6, 1.3), "BUN": ("mg/dL", 7, 20), "ESR": ("mm/h", None, 20), "HB": ("g/dL", 12, 17.5), "K": ("mEq/L", 3.5, 5.0),
 "Na": ("mEq/L", 135, 145), "WBC": ("cells/uL", 4000, 11000), "Lymph": ("%", 20, 40), "Neut": ("%", 40, 75), "PLT": ("x1000/uL", 150, 450)}
ORDINAL = {"VHD": ["None", "Mild", "Moderate", "Severe"], "Function Class": ["0", "1", "2", "3"], "Region RWMA": ["0", "1", "2", "3", "4"]}


class ValidationError(ValueError):
    pass


class Predictor:
    def __init__(self, folder):
        self.folder = folder
        self.params = json.load(open(os.path.join(folder, "model_params.json")))
        self.features = self.params["features"]
        self.lo = pd.Series(self.params["clip_lo"], index=self.features)
        self.hi = pd.Series(self.params["clip_hi"], index=self.features)
        self.T = {t: {k: np.array(self.params["targets"][t][k], dtype=float) for k in ("mean", "scale", "coef")} |
                     {"intercept": float(self.params["targets"][t]["intercept"])} for t in TARGETS}
        self.metrics = json.load(open(os.path.join(folder, "metrics.json")))
        self.data = pd.read_csv(os.path.join(folder, "processed_dataset.csv"))
        flat = [f for _, fs in GROUPS for f in fs]
        assert sorted(flat) == sorted(self.features), "feature groups do not match the trained model"
        d = self.data[self.features]
        self.dmin, self.dmax, self.default = d.min(), d.max(), d.median()
        self.binary = {f for f in self.features if set(d[f].unique()) <= {0, 1}}

    # ------------------------------------------------------------------ metadata
    def schema(self):
        groups = []
        for g, fs in GROUPS:
            items = []
            for f in fs:
                kind = "ordinal" if f in ORDINAL else ("binary" if f in self.binary else "number")
                it = {"name": f, "label": LABEL.get(f, f), "kind": kind, "default": float(self.default[f]),
                      "min": float(self.dmin[f]), "max": float(self.dmax[f])}
                if f == "Sex":
                    it["options"] = [{"value": 1, "label": "Male"}, {"value": 0, "label": "Female"}]; it["kind"] = "ordinal"; it["default"] = 1
                if f in ORDINAL:
                    it["options"] = [{"value": i, "label": l} for i, l in enumerate(ORDINAL[f])]
                if f in PHYSIO:
                    u, a, b = PHYSIO[f]; it["physio"] = True; it["unit"] = u; it["ref_low"] = a; it["ref_high"] = b
                items.append(it)
            groups.append({"group": g, "features": items})
        return {"groups": groups, "targets": TARGETS}

    def samples(self, n=30, seed=7):
        r = np.random.RandomState(seed)
        d = self.data
        idx = list(r.choice(d.index[d.CAD == 0], n // 3, replace=False)) + list(r.choice(d.index[d.CAD == 1], n - n // 3, replace=False))
        out = []
        for i in idx:
            out.append({"id": int(i), "features": {f: float(d.loc[i, f]) for f in self.features},
                        "truth": {t: int(d.loc[i, t]) for t in TARGETS}})
        return out

    def metrics_summary(self):
        gi = {}
        for t in TARGETS:
            imp = sorted(zip(self.features, self.params["targets"][t]["global_importance"]), key=lambda x: -x[1])[:12]
            gi[t] = [{"feature": f, "label": LABEL.get(f, f), "importance": v} for f, v in imp]
        return {"holdout": self.metrics["holdout"], "cross_validation": self.metrics["cross_validation"],
                "lr_setting": self.metrics.get("lr_setting"), "global_importance": gi,
                "deployed_model": "Class balanced logistic regression (per target)",
                "notes": ["Scores are risk scores, not calibrated probabilities (balanced class weights).",
                          "Hold out set is small (61 patients); LCX performance is weak."]}

    # ------------------------------------------------------------------ prediction
    def _coerce(self, name, v):
        if isinstance(v, bool):
            return float(v)
        if name == "Sex" and isinstance(v, str):
            s = v.strip().lower()
            if s in ("male", "m", "1"): return 1.0
            if s in ("female", "fmale", "f", "0"): return 0.0
            raise ValidationError(f"Sex must be Male or Female, got '{v}'")
        if isinstance(v, str) and name in self.binary:
            s = v.strip().lower()
            if s in ("y", "yes", "true", "1"): return 1.0
            if s in ("n", "no", "false", "0"): return 0.0
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise ValidationError(f"'{name}' must be a number, got {v!r}")
        if not np.isfinite(x):
            raise ValidationError(f"'{name}' must be finite")
        if name in self.binary and x not in (0.0, 1.0):
            raise ValidationError(f"'{name}' must be 0 or 1, got {v!r}")
        return x

    def predict(self, features):
        if not isinstance(features, dict):
            raise ValidationError("'features' must be an object of feature name to value")
        unknown = [k for k in features if k not in self.features]
        if unknown:
            raise ValidationError("Unknown feature(s): " + ", ".join(unknown[:8]))
        row, filled, warnings = {}, [], []
        for f in self.features:
            if f in features and features[f] not in (None, ""):
                row[f] = self._coerce(f, features[f])
                if row[f] < self.dmin[f] or row[f] > self.dmax[f]:
                    warnings.append(f"{LABEL.get(f, f)} = {row[f]:g} is outside the training range ({self.dmin[f]:g} to {self.dmax[f]:g}); value is clipped")
            else:
                row[f] = float(self.default[f]); filled.append(f)
        X = pd.DataFrame([row])[self.features]
        Xc = X.clip(self.lo, self.hi, axis=1)
        res = {"probabilities": {}, "targets": {}}
        for t in TARGETS:
            T = self.T[t]
            z = (Xc.values[0] - T["mean"]) / T["scale"]; c = z * T["coef"]
            logit = float(T["intercept"] + c.sum()); p = float(1 / (1 + np.exp(-logit)))
            tot = float(np.abs(c).sum()) or 1.0
            contrib = [{"feature": f, "label": LABEL.get(f, f), "value": row[f], "contribution": float(c[i]),
                        "share_percent": float(abs(c[i]) / tot * 100)} for i, f in enumerate(self.features)]
            contrib.sort(key=lambda d: -abs(d["contribution"]))
            res["probabilities"][t] = p
            res["targets"][t] = {"probability": p, "category": category(p), "logit": logit,
                                 "baseline_logit": T["intercept"], "contributions": contrib}
        res["filled_with_defaults"] = filled; res["warnings"] = warnings
        res["disclaimer"] = "Predictions are for decision support and educational purposes only and are not a substitute for formal diagnostic imaging."
        return res


def category(p):
    return "Low" if p < 0.33 else ("Moderate" if p < 0.66 else "High")
