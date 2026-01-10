#!/usr/bin/env python3
import argparse, pickle
from pathlib import Path

def norm_id(x):
    # sample_idx might be int (e.g., 862) or string (e.g., '000862')
    if isinstance(x, int):
        return f"{x:06d}"
    s = str(x)
    return s.zfill(6) if s.isdigit() else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. data/kitti or data/kitti_infra")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--src", default="kitti_infos_trainval.pkl",
                    help="source labeled infos pkl inside root (default: kitti_infos_trainval.pkl)")
    ap.add_argument("--out", default=None, help="output pkl filename inside root")
    args = ap.parse_args()

    root = Path(args.root)
    ids = [ln.strip() for ln in (root/"ImageSets"/f"{args.split}.txt").read_text().splitlines() if ln.strip()]
    idset = set(ids)

    src_pkl = root/args.src
    data = pickle.loads(src_pkl.read_bytes())

    # MMDet3D v1.x KITTI infos are typically: {"metainfo":..., "data_list":[...]}
    meta = data.get("metainfo", {})
    dl = data.get("data_list", data)  # fallback if older format is a list

    new_dl = []
    for info in dl:
        sid = norm_id(info.get("sample_idx", info.get("image_idx", "")))
        if sid in idset:
            new_dl.append(info)

    out_name = args.out or f"kitti_infos_{args.split}_gt.pkl"
    out = {"metainfo": meta, "data_list": new_dl}
    (root/out_name).write_bytes(pickle.dumps(out))
    print(f"Wrote {root/out_name} with {len(new_dl)} samples")

if __name__ == "__main__":
    main()
