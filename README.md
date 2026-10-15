# CAD Risk 3D: Coronary Artery Disease Prediction on an Interactive 3D Heart

Multimodal AI Hackathon 2026, **Track A: Cardiovascular Risk Visualization & Prediction**.

Predicts overall CAD and stenosis of the **LAD, LCX and RCA** from clinical, ECG, lab and echo data
(Z-Alizadeh Sani extension, 303 patients), explains each prediction with exact SHAP values, and colours
the matching coronary arteries on an interactive 3D heart (BodyParts3D meshes, Three.js).

> **Clinical safety:** for decision support and educational purposes only. Not a substitute for formal diagnostic imaging or clinical judgement.


## Problem Statement — Track A: Cardiovascular Risk Visualization & Prediction

### Challenge

Build an interactive 3D visualization system that predicts coronary artery disease and overall cardiac risk from patient physiological and clinical data, mapping predictions onto an interactive 3D human anatomical model.

### Track A Requirements Addressed

Our project, **CAD Risk 3D**, addresses the Track A challenge through the following features:

- **Predictive Modeling:** Predicts overall Coronary Artery Disease (CAD) status and vessel-specific stenosis status for the LAD, LCX, and RCA using clinical, demographic, laboratory, ECG, and echocardiographic features.
- **3D Visual Mapping:** Displays an interactive 3D heart with coronary arteries color-coded according to their predicted risk, with rotation, zoom, and vessel selection.
- **Clinical Dashboard:** Presents risk predictions, SHAP-based explanations, and physiological measurements alongside the visualization.
- **Clinical Safety:** Clearly states that predictions are intended for educational and decision-support purposes and are not a substitute for formal diagnostic imaging.

The system uses the Extension of Z-Alizadeh Sani dataset, containing 303 patient records, and open-source anatomical meshes from BodyParts3D. It is designed to help users explore model predictions intuitively while acknowledging that predicted vessel risk does not identify the location or severity of an anatomical lesion.

## Features
- 4 binary classifiers (CAD, LAD, LCX, RCA); `LAD, LCX, RCA, Cath` excluded from inputs (asserted in code) to prevent target leakage
- 53 input features; 1st/99th percentile clipping applied identically at train and inference
- Exact SHAP contributions (logistic regression: coefficient x standardised value; sums to the logit, unit-tested)
- 3D viewer: rotate, zoom, click LAD / LCX / RCA / heart; colour and glow follow predicted probability
- Dashboard: probabilities, top-10 contributions, measurements table with relative contribution, metrics
- Runs without a dedicated GPU; backend needs only Flask, NumPy, pandas

## Results (80/20 stratified hold-out, n=61, seed 42, threshold 0.5)

| Target | ROC-AUC (95% CI) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| CAD | 0.897 (0.80-0.97) | 0.820 | 0.900 | 0.837 | 0.867 |
| LAD | 0.854 (0.75-0.94) | 0.705 | 0.821 | 0.639 | 0.719 |
| LCX | 0.660 (0.52-0.79) | 0.607 | 0.500 | 0.625 | 0.556 |
| RCA | 0.788 (0.66-0.90) | 0.705 | 0.586 | 0.739 | 0.654 |

Tuning and 5-fold CV use the training split only. Deployed weights are refit on all 303 patients.
Scores are class-balanced risk scores, not calibrated probabilities. LCX is the weakest target.

## Quick start (Python 3.10+)
```bash
git clone https://github.com/harushitags86-tech/Multi-Model-Hackathon.git
cd Multi-Model-Hackathon
pip install -r requirements.txt
python test_api.py        # backend self-test
python app.py             # open http://127.0.0.1:8000
```
Retrain (optional): `pip install -r requirements-train.txt && python train_pipeline.py && python evaluate_plots.py`

Standalone viewer: open `cad_3d_viewer_standalone.html` in a browser (needs internet for Three.js, GLTFLoader and JSZip from CDNs).

## API
| Endpoint | Purpose |
|---|---|
| `POST /api/predict` | `{"features": {...}}` -> probabilities, risk band, SHAP contributions (missing values use training median; invalid input returns 400) |
| `GET /api/schema` | 53 grouped features with type, range, default, unit |
| `GET /api/samples` | 30 real dataset patients with true labels |
| `GET /api/metrics` | hold-out and CV metrics, global importance |
| `GET /api/health` | liveness |

## Structure
```
app.py, predictor.py        Flask API + inference
train_pipeline.py           preprocessing, CV, hold-out, export
evaluate_plots.py           ROC, confusion, calibration, importance
extract_bp3d.py             builds mesh set from BodyParts3D
static/dashboard.html       clinical dashboard
static/models/bp3d/         .obj meshes + manifest.json (file -> role mapping)
cad_3d_viewer_standalone.html
data/                       Z-Alizadeh Sani extension (xlsx)
plots/, metrics.json, model_weights.joblib, model_params.json
docs/                       project documentation, demo script
```

## Extending
New anatomy: list files in `static/models/bp3d/manifest.json`. New features/models: update the schema, rerun `train_pipeline.py`.

## Limitations
Single-centre data (303 patients); weak LCX model; small test set; no lesion location in the dataset, so the 3D view shows model risk per vessel, not a measured lesion; generic, not patient-specific, anatomy.

## License
Code: MIT (see `LICENSE`). Anatomy and dataset: see `NOTICE.md`.

Demo video: https://www.youtube.com/watch?v=PixuO_2qoM4&t=10s
