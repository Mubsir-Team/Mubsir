"""
Record SHA-256 of every existing project file (outside this study folder and the virtual environment).
Usage: python hash_artifacts.py before|after     -> hashes_before.json / hashes_after.json (+ comparison on 'after')
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent.parent  # studies/<name>/ -> Model/ (moved 2026-11-26)
SKIP = {".venv", "__pycache__", HERE.name, "user_risk"}


def files():
    for p in sorted(MODEL_DIR.rglob("*")):
        rel = p.relative_to(MODEL_DIR)
        if p.is_file() and not (set(rel.parts) & SKIP):
            yield rel.as_posix(), p


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


which = sys.argv[1]
hashes = {rel: digest(p) for rel, p in files()}
(HERE / f"hashes_{which}.json").write_text(json.dumps(hashes, indent=1), encoding="utf-8")
print(f"{which}: {len(hashes)} files hashed")
if which == "after":
    before = json.loads((HERE / "hashes_before.json").read_text(encoding="utf-8"))
    changed = [k for k in before if k in hashes and before[k] != hashes[k]]
    removed = [k for k in before if k not in hashes]
    added = [k for k in hashes if k not in before]
    key = ["model/lgbm_y10.txt", "model/manifest.json", "work/test/test_scores.parquet", "data/model_dataset_995.parquet",
           "config/final_split.csv", "config/selection_rules.json", "results/pretest/freeze.json", "work/oof/lightgbm_h10.parquet"]
    out = {"files_compared": len(before), "changed": changed, "removed": removed, "added_outside_study": added,
           "key_artifacts": {k: {"before": before.get(k), "after": hashes.get(k), "unchanged": before.get(k) == hashes.get(k)}
                             for k in key}}
    (HERE / "hashes_comparison.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "key_artifacts"}, indent=1))
    print("all key artifacts unchanged:", all(v["unchanged"] for v in out["key_artifacts"].values()))
