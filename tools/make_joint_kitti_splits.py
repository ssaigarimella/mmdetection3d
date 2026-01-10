#!/usr/bin/env python3
import argparse, random
from pathlib import Path

def stems(d: Path, suffix: str):
    if not d.is_dir():
        return set()
    return {p.stem for p in d.glob(f"*{suffix}")}

def write_list(p: Path, ids):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for s in ids:
            f.write(f"{s}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle-root", required=True)
    ap.add_argument("--infra-root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    args = ap.parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1.0")

    veh = Path(args.vehicle_root).resolve()
    inf = Path(args.infra_root).resolve()

    veh_ids = stems(veh / "training" / "label_2", ".txt") & stems(veh / "training" / "velodyne", ".bin")
    inf_ids = stems(inf / "training" / "label_2", ".txt") & stems(inf / "training" / "velodyne", ".bin")
    common = sorted(veh_ids & inf_ids)
    if not common:
        raise RuntimeError("No common IDs with BOTH labels and velodyne in BOTH roots")

    rng = random.Random(args.seed)
    rng.shuffle(common)

    n = len(common)
    n_train = int(round(args.train * n))
    n_val   = int(round(args.val   * n))
    n_test  = n - n_train - n_val

    train_ids = sorted(common[:n_train])
    val_ids   = sorted(common[n_train:n_train+n_val])
    test_ids  = sorted(common[n_train+n_val:])

    for root in [veh, inf]:
        imgsets = root / "ImageSets"
        write_list(imgsets / "train.txt", train_ids)
        write_list(imgsets / "val.txt",   val_ids)
        write_list(imgsets / "test.txt",  test_ids)
        # IMPORTANT: keep trainval as train+val (standard meaning)
        write_list(imgsets / "trainval.txt", sorted(train_ids + val_ids))

    print("Wrote joint splits with common labeled IDs")
    print(f"Total common: {n}")
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")
    print(f"Seed: {args.seed}")

if __name__ == "__main__":
    main()
