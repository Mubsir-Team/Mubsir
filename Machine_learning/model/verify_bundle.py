"""Standard-library-only integrity check of the MUBSIR bundle: python verify_bundle.py [bundle_dir]"""

import hashlib
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent)
bad, n = [], 0
for line in (d / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    digest, name = line.split(maxsplit=1)
    name = name.strip().lstrip("*")
    n += 1
    f = d / name
    if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != digest:
        bad.append(name)
print(f"{n - len(bad)}/{n} files OK" + (f"; FAILED: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
