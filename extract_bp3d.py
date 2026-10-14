"""Prepare real BodyParts3D anatomy for the 3D viewer.

Usage
  python extract_bp3d.py --download            # downloads partof_BP3D_4.0_obj_99.zip (about 62 MB) from the official archive, then extracts
  python extract_bp3d.py path/to/partof_BP3D_4.0_obj_99.zip

Writes the needed .obj files to static/models/bp3d/ plus manifest.json (roles per file). The viewer loads them automatically at start.
Warning: this replaces the manifest, so the viewer then uses the official 4.0 meshes instead of the FJ meshes that ship with the project.
Attribution is required. Use the license text stated in the header of the files you use and on
https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html (the older FJ files state CC Attribution-Share Alike 2.1 Japan)."""
import os, re, sys, json, zipfile, argparse, urllib.request

URL = "https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/partof_BP3D_4.0_obj_99.zip"
NEEDED = {  # representation id -> what it is (ids taken from the official partof_parts_list_e.txt)
    "BP9305": "heart", "BP9548": "LAD (anterior interventricular branch of left coronary artery)",
    "BP9745": "LCX (circumflex branch of left coronary artery)", "BP9299": "RCA (right coronary artery)",
    "BP10074": "trunk of left coronary artery", "BP10408": "ascending aorta", "BP10404": "arch of aorta",
    "BP9912": "pulmonary trunk", "BP10386": "superior vena cava", "BP9359": "right lung", "BP9417": "left lung", "BP9393": "rib cage"}
REQUIRED = ["BP9305", "BP9548", "BP9745", "BP9299"]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "static", "models", "bp3d")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("zip", nargs="?"); ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    path = a.zip
    if a.download:
        path = os.path.join(HERE, "data", "partof_BP3D_4.0_obj_99.zip"); os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            print("Downloading", URL); urllib.request.urlretrieve(URL, path)
    if not path or not os.path.exists(path):
        sys.exit("Give the zip path or use --download")
    os.makedirs(OUT, exist_ok=True); found = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            m = re.search(r"(BP\d+)\.obj$", name, re.I)
            if m and m.group(1).upper() in NEEDED:
                bid = m.group(1).upper()
                with z.open(name) as src, open(os.path.join(OUT, bid + ".obj"), "wb") as dst:
                    dst.write(src.read())
                found.append(bid); print("extracted", bid, "-", NEEDED[bid])
    ROLE = {"BP9305": "heart", "BP9548": "LAD", "BP9745": "LCX", "BP9299": "RCA", "BP10074": "main", "BP10408": "aortaA", "BP10404": "aortaArch",
            "BP9912": "pulm", "BP10386": "svc", "BP9359": "lungR", "BP9417": "lungL", "BP9393": "ribs"}
    roles = {}
    for bid in sorted(found):
        roles.setdefault(ROLE[bid], []).append(bid + ".obj")
    json.dump({"roles": roles, "license": "BodyParts3D, (c) The Database Center for Life Science; see the license stated in the files"},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    miss = [b for b in REQUIRED if b not in found]
    print("Missing required parts:", miss if miss else "none", "\nDone. Start the app (python app.py); the viewer loads these files automatically.")


if __name__ == "__main__":
    main()
