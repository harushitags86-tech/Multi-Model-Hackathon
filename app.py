"""Backend API + static hosting for the dashboard and 3D viewer.
Run:  python app.py      then open http://127.0.0.1:8000"""
import os
from flask import Flask, jsonify, request, send_from_directory
from predictor import Predictor, ValidationError

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
pred = Predictor(HERE)
SCHEMA, SAMPLES, METRICS = pred.schema(), pred.samples(), pred.metrics_summary()


@app.after_request
def cors(resp):  # lets the dashboard also work when opened from another origin or file://
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/api/predict", methods=["OPTIONS"])
def preflight():
    return ("", 204)


@app.get("/api/health")
def health():
    return jsonify(status="ok", n_features=len(pred.features), targets=["CAD", "LAD", "LCX", "RCA"])


@app.get("/api/schema")
def schema():
    return jsonify(SCHEMA)


@app.get("/api/samples")
def samples():
    return jsonify(SAMPLES)


@app.get("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Send a JSON body such as {\"features\": {\"Age\": 60}}"), 400
    feats = body["features"] if "features" in body else body
    try:
        return jsonify(pred.predict(feats))
    except ValidationError as e:
        return jsonify(error=str(e)), 400


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "dashboard.html")


@app.get("/viewer")
def viewer():
    return send_from_directory(HERE, "cad_3d_viewer_standalone.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), debug=False)
