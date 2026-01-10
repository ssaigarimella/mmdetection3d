#!/usr/bin/env python3
import argparse, pickle
from pathlib import Path

def read_ids(p: Path):
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

def norm6(x):
    # normalize int/str sample_idx to 6-digit string
    if isinstance(x, int):
        return f"{x:06d}"
    s = str(x).strip()
    if s.isdigit():
        return f"{int(s):06d}"
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="KITTI root containing ImageSets/ and kitti_infos_trainval.pkl")
    args = ap.parse_args()
    root = Path(args.root)

    test_ids = set(read_ids(root / "ImageSets" / "test.txt"))
    test_ids6 = {norm6(x) for x in test_ids}

    src = root / "kitti_infos_trainval.pkl"
    dst = root / "kitti_infos_test.pkl"

    obj = pickle.load(open(src, "rb"))
    if not isinstance(obj, dict) or "data_list" not in obj:
        raise RuntimeError(f"{src} is not the expected dict format with data_list")

    keep = []
    for info in obj["data_list"]:
        sid = norm6(info.get("sample_idx", ""))
        if sid in test_ids6:
            keep.append(info)

    out = dict(obj)
    out["data_list"] = keep

    with open(dst, "wb") as f:
        pickle.dump(out, f)

    print(f"Wrote labeled {dst} with {len(keep)} frames (from {src})")

if __name__ == "__main__":
    main()
