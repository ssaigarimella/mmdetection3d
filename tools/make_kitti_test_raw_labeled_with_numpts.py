#!/usr/bin/env python3
import argparse, pickle
from pathlib import Path
from tools.dataset_converters.kitti_data_utils import get_kitti_image_info

def read_ids_as_int(p: Path):
    out = []
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if s:
            out.append(int(s))  # handles "000862"
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="KITTI root (has training/, ImageSets/)")
    ap.add_argument("--out", default="kitti_infos_test_raw_labeled.pkl")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    ids = read_ids_as_int(root / "ImageSets" / "test.txt")

    infos = list(get_kitti_image_info(
        path=str(root),
        training=True,      # read labels from training/label_2
        label_info=True,
        velodyne=True,
        calib=True,
        image_ids=ids,
        relative_path=True
    ))

    # Add missing field required by update_infos_to_v2
    for it in infos:
        ann = it.get("annos", None)
        if not ann:
            continue
        n = len(ann.get("name", []))
        if "num_points_in_gt" not in ann:
            ann["num_points_in_gt"] = [0] * n

    out_pkl = root / args.out
    with open(out_pkl, "wb") as f:
        pickle.dump(infos, f)
    print(f"Wrote {out_pkl} frames {len(infos)}")

if __name__ == "__main__":
    main()
