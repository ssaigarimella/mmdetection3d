#!/usr/bin/env python3
"""
Run DAIR-V2X official eval.py from within this repo, without modifying DAIR-V2X.
This mirrors scripts/eval_lidar_late_fusion_pointpillars.sh defaults.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dairv2x-root", type=str, required=True,
                    help="Path to DAIR-V2X repo root (contains v2x/).")
    ap.add_argument("--input", type=str, default="",
                    help="DAIR cooperative-vehicle-infrastructure root. If empty, uses <root>/data/DAIR-V2X/cooperative-vehicle-infrastructure.")
    ap.add_argument("--output", type=str, default="",
                    help="Output cache dir. If empty, uses <root>/cache/vic-late-lidar.")
    ap.add_argument("--fusion-method", type=str, default="late_fusion",
                    choices=["veh_only", "inf_only", "late_fusion"],
                    help="Fusion method.")
    ap.add_argument("--dataset", type=str, default="vic-async",
                    choices=["vic-sync", "vic-async"],
                    help="Dataset type.")
    ap.add_argument("--k", type=int, default=0, help="Async delay k (0 for VIC-Sync behavior).")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--pred-classes", nargs="+", default=["car"])
    ap.add_argument("--extended-range", nargs="+", type=float,
                    default=[0, -39.68, -3, 100, 39.68, 1],
                    help="6 numbers: x_min y_min z_min x_max y_max z_max")
    ap.add_argument("--no-comp", action="store_true", help="Disable time compensation.")

    ap.add_argument("--inf-config", type=str, default="",
                    help="Infra config path. Default: <root>/configs/vic3d/late-fusion-pointcloud/pointpillars/trainval_config_i.py")
    ap.add_argument("--veh-config", type=str, default="",
                    help="Veh config path. Default: <root>/configs/vic3d/late-fusion-pointcloud/pointpillars/trainval_config_v.py")
    ap.add_argument("--inf-ckpt", type=str, default="",
                    help="Infra checkpoint path. Default: <root>/configs/vic3d/late-fusion-pointcloud/pointpillars/vic3d_latefusion_inf_pointpillars_*.pth")
    ap.add_argument("--veh-ckpt", type=str, default="",
                    help="Veh checkpoint path. Default: <root>/configs/vic3d/late-fusion-pointcloud/pointpillars/vic3d_latefusion_veh_pointpillars_*.pth")

    ap.add_argument("--split-data-path", type=str, default="",
                    help="Split json path. Default: <root>/data/split_datas/cooperative-split-data.json")

    args = ap.parse_args()

    root = Path(args.dairv2x_root).expanduser().resolve()
    v2x_dir = root / "v2x"
    eval_py = v2x_dir / "eval.py"
    if not eval_py.is_file():
        print(f"ERROR: DAIR-V2X eval.py not found: {eval_py}")
        return 2

    input_root = args.input or str(root / "data" / "DAIR-V2X" / "cooperative-vehicle-infrastructure")
    output_root = args.output or str(root / "cache" / "vic-late-lidar")
    split_json = args.split_data_path or str(root / "data" / "split_datas" / "cooperative-split-data.json")

    default_cfg_dir = root / "configs" / "vic3d" / "late-fusion-pointcloud" / "pointpillars"
    inf_cfg = args.inf_config or str(default_cfg_dir / "trainval_config_i.py")
    veh_cfg = args.veh_config or str(default_cfg_dir / "trainval_config_v.py")

    if args.inf_ckpt:
        inf_ckpt = args.inf_ckpt
    else:
        inf_ckpt = next(default_cfg_dir.glob("vic3d_latefusion_inf_pointpillars_*.pth"), None)
        inf_ckpt = str(inf_ckpt) if inf_ckpt else ""

    if args.veh_ckpt:
        veh_ckpt = args.veh_ckpt
    else:
        veh_ckpt = next(default_cfg_dir.glob("vic3d_latefusion_veh_pointpillars_*.pth"), None)
        veh_ckpt = str(veh_ckpt) if veh_ckpt else ""

    if not inf_ckpt or not Path(inf_ckpt).is_file():
        print(f"ERROR: infra checkpoint not found: {inf_ckpt}")
        return 2
    if not veh_ckpt or not Path(veh_ckpt).is_file():
        print(f"ERROR: vehicle checkpoint not found: {veh_ckpt}")
        return 2

    cmd = [
        sys.executable, str(eval_py),
        "--input", input_root,
        "--output", output_root,
        "--model", args.fusion_method,
        "--dataset", args.dataset,
        "--k", str(args.k),
        "--split", args.split,
        "--split-data-path", split_json,
        "--inf-config-path", inf_cfg,
        "--inf-model-path", inf_ckpt,
        "--veh-config-path", veh_cfg,
        "--veh-model-path", veh_ckpt,
        "--device", str(args.device),
        "--pred-classes",
    ] + list(args.pred_classes) + [
        "--sensortype", "lidar",
        "--extended-range",
    ] + [str(x) for x in args.extended_range] + [
        "--overwrite-cache",
    ]

    if args.no_comp:
        cmd.append("--no-comp")

    print("[CMD]", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(v2x_dir))


if __name__ == "__main__":
    raise SystemExit(main())
