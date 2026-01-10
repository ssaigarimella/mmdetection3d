#!/usr/bin/env python3
import argparse, pickle
from pathlib import Path

from tools.dataset_converters.kitti_data_utils import get_kitti_image_info

def read_ids_as_ints(p: Path):
    ids = []
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if not s:
            continue
        ids.append(int(s))  # handles "000862" etc
    return ids

def dump_mmengine_dict(data_list, out_path: Path, classes):
    out = {
        "metainfo": {"classes": classes},
        "data_list": list(data_list),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

def build(root: Path, split: str, out_name: str, classes):
    ids_txt = root / "ImageSets" / f"{split}.txt"
    ids = read_ids_as_ints(ids_txt)
    if not ids:
        raise RuntimeError(f"No ids in {ids_txt}")

    # IMPORTANT:
    # training=True + label_info=True forces parsing labels from training/label_2
    infos = get_kitti_image_info(
        path=str(root),
        training=True,
        label_info=True,
        velodyne=True,
        calib=True,
        image_ids=ids,
        relative_path=True
    )

    out_path = root / out_name
    dump_mmengine_dict(infos, out_path, classes)
    print(f"Wrote {out_path} ({len(ids)} frames)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--classes", nargs="+", default=["Pedestrian", "Cyclist", "Car"])
    args = ap.parse_args()
    root = Path(args.root).resolve()

    build(root, "train",    "kitti_infos_train.pkl",    args.classes)
    build(root, "val",      "kitti_infos_val.pkl",      args.classes)
    build(root, "trainval", "kitti_infos_trainval.pkl", args.classes)
    build(root, "test",     "kitti_infos_test.pkl",     args.classes)

if __name__ == "__main__":
    main()
