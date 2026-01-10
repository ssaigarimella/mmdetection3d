#!/usr/bin/env python3
import argparse, pickle
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-pkl", required=True)
    ap.add_argument("--out-pkl", default="")
    ap.add_argument("--fill", type=int, default=0, help="value to fill num_points_in_gt with")
    args = ap.parse_args()

    in_p = Path(args.in_pkl)
    out_p = Path(args.out_pkl) if args.out_pkl else in_p

    obj = pickle.load(open(in_p, "rb"))

    # If it's already v2 dict format, do nothing.
    if isinstance(obj, dict) and "data_list" in obj:
        print(f"{in_p} already looks like v2 (dict with data_list). No changes made.")
        return

    if not isinstance(obj, list):
        raise RuntimeError(f"Expected old-format list, got {type(obj)}")

    for it in obj:
        ann = it.get("annos", None)
        if not ann:
            continue

        names = ann.get("name", [])
        n = len(names)

        npgt = ann.get("num_points_in_gt", None)

        # Force it to be a NumPy array of length n, so element has .tolist()
        if npgt is None:
            ann["num_points_in_gt"] = np.full((n,), args.fill, dtype=np.int32)
        else:
            arr = np.asarray(npgt, dtype=np.int32)
            if arr.shape == ():  # scalar
                arr = np.full((n,), int(arr), dtype=np.int32)
            elif arr.shape[0] != n:
                # resize safely
                if arr.shape[0] > n:
                    arr = arr[:n]
                else:
                    pad = np.full((n - arr.shape[0],), args.fill, dtype=np.int32)
                    arr = np.concatenate([arr, pad], axis=0)
            ann["num_points_in_gt"] = arr

    with open(out_p, "wb") as f:
        pickle.dump(obj, f)
    print(f"Patched num_points_in_gt to NumPy arrays: {out_p}")

if __name__ == "__main__":
    main()
