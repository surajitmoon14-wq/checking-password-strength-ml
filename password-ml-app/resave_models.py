from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]

pkl_files = list(ROOT.glob("**/*.pkl"))
print(f"Found {len(pkl_files)} pickle files to attempt resave")

for p in pkl_files:
    # skip already resaved files
    if p.stem.endswith("_resaved"):
        continue
    out = p.with_name(p.stem + "_resaved.pkl")
    try:
        obj = joblib.load(p)
        joblib.dump(obj, out)
        print("resaved:", p, '->', out)
    except Exception as e:
        print("skip:", p, "error:", e)
