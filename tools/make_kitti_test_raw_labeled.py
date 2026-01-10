#!/usr/bin/env python3
import argparse, pickle
from pathlib import Path
from tools.dataset_converters.kitti_data_utils import get_kitti_image_info

def read_ids_as_int(p: Path):
    out = []
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if not s:
            continue
        out.append(int(s))  # handles "000862"
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="KITTI root (has training/, ImageSets/)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    ids = read_ids_as_int(root / "ImageSets" / "test.txt")

    infos = get_kitti_image_info(
        path=str(root),
        training=True,      # IMPORTANT: read labels from training/label_2
        label_info=True,    # IMPORTANT: actually parse labels
        velodyne=True,
        calib=True,
        image_ids=ids,
        relative_path=True
    )

    out_pkl = root / "kitti_infos_test_raw_labeled.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(list(infos), f)

    print("Wrote", out_pkl, "frames", len(ids))

if __name__ == "__main__":
    main()
