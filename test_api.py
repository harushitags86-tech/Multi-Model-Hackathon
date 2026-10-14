"""Quick self test of the backend:  python test_api.py"""
from app import app, SAMPLES

c = app.test_client()
assert c.get("/api/health").get_json()["status"] == "ok"
assert sum(len(g["features"]) for g in c.get("/api/schema").get_json()["groups"]) == 53
s = SAMPLES[0]
r = c.post("/api/predict", json={"features": s["features"]}).get_json()
assert set(r["probabilities"]) == {"CAD", "LAD", "LCX", "RCA"} and all(0 <= p <= 1 for p in r["probabilities"].values())
t = r["targets"]["LAD"]
assert abs(sum(x["contribution"] for x in t["contributions"]) + t["baseline_logit"] - t["logit"]) < 1e-9   # SHAP additivity
assert c.post("/api/predict", json={"features": {"NotAFeature": 1}}).status_code == 400
assert c.post("/api/predict", json={"features": {"Age": "abc"}}).status_code == 400
assert c.post("/api/predict", json={"features": {"DM": 2}}).status_code == 400
v=c.get("/viewer"); assert v.status_code==200 and b"BP3D_EMBED" in v.data   # the served viewer contains the real meshes
print("all backend tests passed")
