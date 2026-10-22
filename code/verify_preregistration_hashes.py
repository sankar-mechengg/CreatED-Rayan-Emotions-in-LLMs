"""Check the frozen preregistration files against the recorded SHA-256 hashes.

Run from anywhere:

    python code/verify_preregistration_hashes.py

The runnable scripts under code/ were given repository-relative paths so they
can be executed on this machine. Those edits change the file bytes, so the
hashes are checked against preregistration/hashed_files_as_frozen/, which
keeps the files named in preregistration/HASHES.txt and
preregistration/HASHES_DEVIATION_1.txt.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "preregistration" / "hashed_files_as_frozen"
LISTS = [
    ROOT / "preregistration" / "HASHES.txt",
    ROOT / "preregistration" / "HASHES_DEVIATION_1.txt",
]
# The supplementary archive's analyse_experiments.py is the script that wrote
# the released Table 3 files. It is not the byte string hashed at the
# 2026-10-05T20:36:36Z freeze. The pre-freeze source was not in the project
# files. See preregistration/README.md.
KNOWN_LATER_REVISION = {
    "analyse_experiments.py": "784b5ad06b0e75db817d57eb3c5ac375a56664ebd8531e862c629181bf82553e",
}


def _entries(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            yield parts[0], parts[1]


def main() -> int:
    ok = True
    checked = 0
    for listing in LISTS:
        print(listing.name)
        for digest, rel in _entries(listing):
            fp = FROZEN / rel
            checked += 1
            if not fp.is_file():
                print(f"  MISSING  {rel}")
                ok = False
                continue
            found = hashlib.sha256(fp.read_bytes()).hexdigest()
            if found == digest:
                status = "OK"
            elif KNOWN_LATER_REVISION.get(rel) == found:
                status = "LATER"
                print(f"  {status:8}  {rel}")
                print("           Recorded at the freeze, not matched by the released script.")
                print(f"           Freeze SHA-256  {digest}")
                print(f"           Archive SHA-256 {found}")
                continue
            else:
                status = "MISMATCH"
                ok = False
            print(f"  {status:8}  {rel}")
    print(f"{checked} files checked")
    if not ok:
        print("One or more frozen files do not match the preregistration record.")
        return 1
    print("Every recorded file matches the freeze, except analyse_experiments.py,")
    print("which is the later archive revision named above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
