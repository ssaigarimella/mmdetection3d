#!/usr/bin/env python3
import argparse, pickle
from pathlib import Path

from tools.dataset_converters.kitti_data_utils import get_kitti_image_info

def read_ids_int(p: Path):
    ids = []
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if not s:
            continue
        # handles "000862" correctly
        ids.append(int(s))
    return ids

def dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def build(root: Path, split: str, out_name: str):
    ids = read_ids_int(root / "ImageSets" / f"{split}.txt")
    if not ids:
        raise RuntimeError(f"No ids in ImageSets/{split}.txt under {root}")

    # training=True + label_info=True forces label parsing from training/label_2
    infos = get_kitti_image_info(
        path=str(root),
        training=True,
        label_info=True,
        velodyne=True,
        calib=True,
        image_ids=ids,
        relative_path=True
    )
    dump(infos, root / out_name)
    print(f"Wrote {root/out_name} ({len(infos)} frames)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="KITTI root (has training/, ImageSets/)")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    build(root, "train", "kitti_infos_train.pkl")
    build(root, "val", "kitti_infos_val.pkl")
    build(root, "trainval", "kitti_infos_trainval.pkl")
    build(root, "test", "kitti_infos_test.pkl")  # labeled test subset

if __name__ == "__main__":
    main()
