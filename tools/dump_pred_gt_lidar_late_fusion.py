#!/usr/bin/env python3
"""
tools/dump_pred_gt_lidar_late_fusion.py

Full-split dump + optional evaluation for:
  - VEHICLE-only predictions vs UNION GT (vehicle frame)
  - INFRA-only predictions (transformed into vehicle frame) vs UNION GT
  - LATE-FUSION predictions (vehicle + transformed infra) vs UNION GT

Designed to MATCH tools/vis_lidar_late_fusion_single_sample.py:
  - Same dataset pairing (LiDAR filename stem)
  - Same per-side inference wrapper and FOV filtering behavior
  - Same robust auto inversion selection for T_veh_from_inf
  - Same prediction Z fix (if enabled in vis script)
  - Same fusion policy, but returns fused labels too

Important: tools/eval_lidar_ap_from_dump.py expects per-sample keys:
  gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels

So we write BOTH naming styles:
  gt_boxes + gt_boxes_3d
  gt_labels + gt_labels_3d
  pred_boxes + pred_boxes_3d
  pred_scores + pred_scores_3d
  pred_labels + pred_labels_3d
"""

import argparse
import importlib.util
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from mmengine.config import Config
from mmengine.registry import DATASETS

from mmdet3d.apis import init_model
from mmdet3d.utils import register_all_modules


# -------------------------
# Load the vis script as a module (no package assumptions)
# -------------------------

def load_vis_module() -> Any:
    this = Path(__file__).resolve()
    vis_path = this.parent / "vis_lidar_late_fusion_single_sample.py"
    if not vis_path.is_file():
        raise FileNotFoundError(f"Missing: {vis_path}")

    spec = importlib.util.spec_from_file_location(
        "vis_lidar_late_fusion_single_sample", str(vis_path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load spec for vis_lidar_late_fusion_single_sample.py")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = load_vis_module()


# -------------------------
# Label remap helper (MODEL -> CANONICAL)
# -------------------------
#
# Canonical evaluator meaning (what eval_lidar_ap_from_dump.py assumes):
#   0 Car, 1 Pedestrian, 2 Cyclist
#
# Your models are outputting (confirmed by your dx checks):
#   2 -> Car
#   0 -> Pedestrian
#   1 -> Cyclist (if present)
#
# So remap:
#   2 -> 0
#   0 -> 1
#   1 -> 2
#
def remap_model_labels_to_canonical(pred_labels: np.ndarray) -> np.ndarray:
    pl = np.asarray(pred_labels, dtype=np.int64).copy()
    if pl.size == 0:
        return pl
    new = pl.copy()
    new[pl == 2] = 0
    new[pl == 0] = 1
    new[pl == 1] = 2
    return new


# -------------------------
# Box conversion helpers (corners -> [x,y,z,dx,dy,dz,yaw])
# -------------------------

def corners_to_box7(corners_8x3: np.ndarray) -> np.ndarray:
    c = np.asarray(corners_8x3, dtype=np.float64).reshape(8, 3)
    center = c.mean(axis=0)

    v_len = c[0] - c[3]
    v_wid = c[0] - c[1]
    v_hgt = c[4] - c[0]

    dx = float(np.linalg.norm(v_len))
    dy = float(np.linalg.norm(v_wid))
    dz = float(np.linalg.norm(v_hgt))

    yaw = float(np.arctan2(v_len[1], v_len[0]))
    return np.array([center[0], center[1], center[2], dx, dy, dz, yaw], dtype=np.float64)


def cornersN_to_box7N(corners_Nx8x3: np.ndarray) -> np.ndarray:
    if corners_Nx8x3 is None or corners_Nx8x3.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float64)
    out = np.zeros((corners_Nx8x3.shape[0], 7), dtype=np.float64)
    for i in range(corners_Nx8x3.shape[0]):
        out[i] = corners_to_box7(corners_Nx8x3[i])
    return out


# -------------------------
# Fusion with labels (matches vis logic, but keeps cls)
# -------------------------

def fuse_preds_with_labels(
    veh_corners: np.ndarray, veh_centers: np.ndarray, veh_scores: np.ndarray, veh_labels: np.ndarray,
    inf_corners_v: np.ndarray, inf_centers_v: np.ndarray, inf_scores: np.ndarray, inf_labels: np.ndarray,
    match_dist_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if veh_centers.shape[0] == 0 and inf_centers_v.shape[0] == 0:
        return (
            np.zeros((0, 8, 3), np.float64),
            np.zeros((0, 3), np.float64),
            np.zeros((0,), np.float64),
            np.zeros((0,), np.int64),
        )
    if veh_centers.shape[0] == 0:
        return inf_corners_v, inf_centers_v, inf_scores, inf_labels
    if inf_centers_v.shape[0] == 0:
        return veh_corners, veh_centers, veh_scores, veh_labels

    fused_corners: List[np.ndarray] = []
    fused_centers: List[np.ndarray] = []
    fused_scores: List[float] = []
    fused_labels: List[int] = []

    classes = np.unique(np.concatenate([veh_labels, inf_labels], axis=0)) \
        if (veh_labels.size + inf_labels.size) > 0 else np.array([], dtype=int)

    for cls in classes:
        idx_v = np.where(veh_labels == cls)[0]
        idx_i = np.where(inf_labels == cls)[0]

        if idx_v.size == 0:
            for j in idx_i:
                fused_corners.append(inf_corners_v[j])
                fused_centers.append(inf_centers_v[j])
                fused_scores.append(float(inf_scores[j]))
                fused_labels.append(int(cls))
            continue
        if idx_i.size == 0:
            for i in idx_v:
                fused_corners.append(veh_corners[i])
                fused_centers.append(veh_centers[i])
                fused_scores.append(float(veh_scores[i]))
                fused_labels.append(int(cls))
            continue

        Vc = veh_centers[idx_v]
        Ic = inf_centers_v[idx_i]
        cost = np.linalg.norm(Vc[:, None, :] - Ic[None, :, :], axis=2)

        r, c = V.hungarian_min_cost(cost)

        used_v = np.zeros((idx_v.size,), dtype=bool)
        used_i = np.zeros((idx_i.size,), dtype=bool)

        for rr, cc in zip(r, c):
            d = float(cost[rr, cc])
            if d > float(match_dist_m):
                continue
            used_v[rr] = True
            used_i[cc] = True
            i = int(idx_v[rr])
            j = int(idx_i[cc])

            if getattr(V, "FUSE_POLICY", "pick_best") == "pick_best":
                if float(veh_scores[i]) >= float(inf_scores[j]):
                    fused_corners.append(veh_corners[i])
                    fused_centers.append(veh_centers[i])
                    fused_scores.append(float(veh_scores[i]))
                    fused_labels.append(int(cls))
                else:
                    fused_corners.append(inf_corners_v[j])
                    fused_centers.append(inf_centers_v[j])
                    fused_scores.append(float(inf_scores[j]))
                    fused_labels.append(int(cls))
            else:
                fused_corners.append(veh_corners[i])
                fused_centers.append(veh_centers[i])
                fused_scores.append(float(veh_scores[i]))
                fused_labels.append(int(cls))

        for rr, i in enumerate(idx_v):
            if not used_v[rr]:
                fused_corners.append(veh_corners[i])
                fused_centers.append(veh_centers[i])
                fused_scores.append(float(veh_scores[i]))
                fused_labels.append(int(cls))
        for cc, j in enumerate(idx_i):
            if not used_i[cc]:
                fused_corners.append(inf_corners_v[j])
                fused_centers.append(inf_centers_v[j])
                fused_scores.append(float(inf_scores[j]))
                fused_labels.append(int(cls))

    if len(fused_corners) == 0:
        return (
            np.zeros((0, 8, 3), np.float64),
            np.zeros((0, 3), np.float64),
            np.zeros((0,), np.float64),
            np.zeros((0,), np.int64),
        )

    fused_c = np.stack(fused_corners, axis=0).astype(np.float64)
    fused_cent = fused_c.mean(axis=1)
    return (
        fused_c,
        fused_cent,
        np.array(fused_scores, dtype=np.float64),
        np.array(fused_labels, dtype=np.int64),
    )


# -------------------------
# Dump + eval helpers
# -------------------------

def save_dump(path: Path, dump: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_cmd(cmd: List[str], cwd: Path) -> int:
    print("\n[CMD]")
    print("  " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd))
    return int(p.returncode)


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("cfg_vehicle", type=str)
    parser.add_argument("ckpt_vehicle", type=str)
    parser.add_argument("cfg_infra", type=str)
    parser.add_argument("ckpt_infra", type=str)

    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--non-kitti-root", required=True, type=str)

    parser.add_argument("--veh-novatel-key", default="novatel_to_world", type=str)
    parser.add_argument("--veh-lidar2novatel-key", default="lidar_to_novatel", type=str)
    parser.add_argument("--inf-lidar2world-key", default="virtuallidar_to_world", type=str)

    parser.add_argument("--score-thr", default=0.1, type=float)
    parser.add_argument("--outdir", required=True, type=str)
    parser.add_argument("--tag", default="val_unionGT", type=str)

    parser.add_argument("--only-veh", action="store_true")
    parser.add_argument("--only-inf", action="store_true")
    parser.add_argument("--only-late-fusion", action="store_true")

    parser.add_argument("--no-native-gt", action="store_true")
    parser.add_argument("--debug-print", action="store_true")

    parser.add_argument("--max-samples", default=0, type=int)
    parser.add_argument("--run-eval", action="store_true")

    args = parser.parse_args()

    if (args.only_veh + args.only_inf + args.only_late_fusion) > 1:
        raise ValueError("Choose at most one of --only-veh / --only-inf / --only-late-fusion")

    register_all_modules(init_default_scope=True)

    cfg_v = Config.fromfile(args.cfg_vehicle)
    cfg_i = Config.fromfile(args.cfg_infra)

    dataset_v = DATASETS.build(cfg_v.test_dataloader.dataset)
    dataset_i = DATASETS.build(cfg_i.test_dataloader.dataset)
    V.ensure_full_init(dataset_v)
    V.ensure_full_init(dataset_i)

    id2idx_v = V.build_pairid_to_idx(dataset_v)
    id2idx_i = V.build_pairid_to_idx(dataset_i)
    common_ids = V.build_common_ids_in_vehicle_order(id2idx_v, id2idx_i)
    if len(common_ids) == 0:
        raise RuntimeError("No common pair-ids between vehicle and infra splits (pairing is by lidar stem).")

    if args.max_samples and args.max_samples > 0:
        common_ids = common_ids[: int(args.max_samples)]

    model_v = init_model(cfg_v, args.ckpt_vehicle, device=args.device)
    model_i = init_model(cfg_i, args.ckpt_infra, device=args.device)
    model_v.eval()
    model_i.eval()

    non_kitti_root = Path(args.non_kitti_root)
    coop_root = V.get_coop_root(non_kitti_root)
    if not coop_root.exists():
        raise FileNotFoundError(f"coop_root does not exist: {coop_root}")

    use_native_gt = bool(getattr(V, "USE_NATIVE_LIDAR_GT", True)) and (not bool(args.no_native_gt))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dump_paths: Dict[str, Path] = {}
    if not (args.only_inf or args.only_late_fusion):
        dump_paths["veh"] = outdir / f"dump_{args.tag}_vehicle_only.pkl"
    if not (args.only_veh or args.only_late_fusion):
        dump_paths["inf"] = outdir / f"dump_{args.tag}_infra_only.pkl"
    if not (args.only_veh or args.only_inf):
        dump_paths["lf"] = outdir / f"dump_{args.tag}_late_fusion.pkl"

    # Keep class list consistent with your TYPE_TO_LABEL mapping in the vis script.
    # Also matches eval_lidar_ap_from_dump.py (classes.index -> cls_id):
    classes = ["Car", "Pedestrian", "Cyclist"]

    dumps: Dict[str, Dict[str, Any]] = {}
    for k in dump_paths.keys():
        dumps[k] = {
            "classes": classes,
            "meta": {
                "created_unix": time.time(),
                "cfg_vehicle": args.cfg_vehicle,
                "ckpt_vehicle": args.ckpt_vehicle,
                "cfg_infra": args.cfg_infra,
                "ckpt_infra": args.ckpt_infra,
                "non_kitti_root": str(non_kitti_root),
                "coop_root": str(coop_root),
                "score_thr": float(args.score_thr),
                "use_native_gt": bool(use_native_gt),
                "pairing": "LiDAR filename stem intersection, vehicle order",
                "transform": "auto inversion selection via reduced-point NN median (same as vis script)",
                "notes": {
                    "FILTER_GT_BY_FOV": bool(getattr(V, "FILTER_GT_BY_FOV", True)),
                    "FILTER_PRED_BY_FOV": bool(getattr(V, "FILTER_PRED_BY_FOV", True)),
                    "FILTER_FUSED_BY_VEH_FOV": bool(getattr(V, "FILTER_FUSED_BY_VEH_FOV", True)),
                    "FILTER_INF_IN_VEH_FRAME_BY_VEH_FOV": bool(getattr(V, "FILTER_INF_IN_VEH_FRAME_BY_VEH_FOV", False)),
                    "FIX_PRED_Z_BY_OWN_HEIGHT": bool(getattr(V, "FIX_PRED_Z_BY_OWN_HEIGHT", True)),
                    "PRED_Z_HEIGHT_FACTOR": float(getattr(V, "PRED_Z_HEIGHT_FACTOR", -1.0)),
                    "MATCH_DIST_M": float(getattr(V, "MATCH_DIST_M", 2.0)),
                    "FUSE_POLICY": str(getattr(V, "FUSE_POLICY", "pick_best")),
                    "LABEL_REMAP": "model{2:Car,0:Ped,1:Cyc} -> canonical{0:Car,1:Ped,2:Cyc}",
                },
            },
            "samples": [],
        }

    T_cache: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}

    print(f"[INFO] vehicle split len={len(dataset_v)}, infra split len={len(dataset_i)}")
    print(f"[INFO] common pair-ids={len(common_ids)} (processing this many)")
    print(f"[INFO] use_native_gt={use_native_gt} (disable with --no-native-gt)")
    print(f"[INFO] writing dumps under: {outdir}")
    for name, p in dump_paths.items():
        print(f"[INFO] will write {name}: {p}")

    for it, sid in enumerate(common_ids):
        idx_v = id2idx_v[sid]
        idx_i = id2idx_i[sid]

        out_v = V.run_one_side(
            dataset_v, model_v, idx_v,
            score_thr=float(args.score_thr),
            want_full_points=False,
            sid=sid, coop_root=coop_root,
            side_name="vehicle-side",
            use_native_gt=use_native_gt,
        )
        out_i = V.run_one_side(
            dataset_i, model_i, idx_i,
            score_thr=float(args.score_thr),
            want_full_points=False,
            sid=sid, coop_root=coop_root,
            side_name="infrastructure-side",
            use_native_gt=use_native_gt,
        )

        if sid in T_cache:
            T_veh_from_inf, src = T_cache[sid]
        else:
            T_veh_from_inf, src = V.compute_T_veh_from_inf_from_json(
                coop_root=coop_root,
                sid=sid,
                veh_novatel_key=args.veh_novatel_key,
                veh_lidar_to_novatel_key=args.veh_lidar2novatel_key,
                inf_lidar_to_world_key=args.inf_lidar2world_key,
                veh_pts_reduced=out_v["pts_reduced_pipeline"],
                inf_pts_reduced=out_i["pts_reduced_pipeline"],
                debug_print=bool(args.debug_print),
            )
            T_cache[sid] = (T_veh_from_inf, src)

        # UNION GT in vehicle frame
        gt_v_c = out_v["gt_corners"]
        gt_v_l = out_v["gt_labels"].astype(np.int64) if out_v["gt_labels"] is not None else np.zeros((0,), dtype=np.int64)

        gt_i_c_v = V.apply_T_corners(T_veh_from_inf, out_i["gt_corners"])
        gt_i_l = out_i["gt_labels"].astype(np.int64) if out_i["gt_labels"] is not None else np.zeros((0,), dtype=np.int64)

        if bool(getattr(V, "FILTER_INF_IN_VEH_FRAME_BY_VEH_FOV", False)) and gt_i_c_v.shape[0] > 0:
            dummy = np.zeros((gt_i_c_v.shape[0],), dtype=np.int64)
            gt_i_cent_v = gt_i_c_v.mean(axis=1)
            gt_i_c_v, gt_i_cent_v, dummy, _, keep_gti = V.filter_boxes_by_pipeline_fov(
                corners=gt_i_c_v,
                centers=gt_i_cent_v,
                labels=dummy,
                pts_pipeline_xyz=out_v["pts_reduced_pipeline"],
            )
            gt_i_l = gt_i_l[keep_gti]

        union_gt_c = np.concatenate([gt_v_c, gt_i_c_v], axis=0) \
            if (gt_v_c.shape[0] + gt_i_c_v.shape[0]) > 0 else np.zeros((0, 8, 3), dtype=np.float64)
        union_gt_l = np.concatenate([gt_v_l, gt_i_l], axis=0) \
            if (gt_v_l.size + gt_i_l.size) > 0 else np.zeros((0,), dtype=np.int64)

        union_gt_box7 = cornersN_to_box7N(union_gt_c)

        # Vehicle preds
        veh_pred_c = out_v["pred_corners"]
        veh_pred_s = out_v["pred_scores"].astype(np.float64) if out_v["pred_scores"] is not None else np.zeros((0,), dtype=np.float64)

        veh_pred_l_raw = out_v["pred_labels"]
        veh_pred_l_raw = veh_pred_l_raw.astype(np.int64) if veh_pred_l_raw is not None else np.zeros((0,), dtype=np.int64)
        veh_pred_l = remap_model_labels_to_canonical(veh_pred_l_raw)

        veh_pred_box7 = cornersN_to_box7N(veh_pred_c)

        # Infra preds transformed to vehicle
        inf_pred_c_v = V.apply_T_corners(T_veh_from_inf, out_i["pred_corners"])
        inf_pred_s = out_i["pred_scores"].astype(np.float64) if out_i["pred_scores"] is not None else np.zeros((0,), dtype=np.float64)

        inf_pred_l_raw = out_i["pred_labels"]
        inf_pred_l_raw = inf_pred_l_raw.astype(np.int64) if inf_pred_l_raw is not None else np.zeros((0,), dtype=np.int64)
        inf_pred_l = remap_model_labels_to_canonical(inf_pred_l_raw)

        if bool(getattr(V, "FILTER_INF_IN_VEH_FRAME_BY_VEH_FOV", False)) and inf_pred_c_v.shape[0] > 0:
            dummy = np.zeros((inf_pred_c_v.shape[0],), dtype=np.int64)
            inf_pred_cent_v = inf_pred_c_v.mean(axis=1)
            inf_pred_c_v, inf_pred_cent_v, dummy, _, keep_infp = V.filter_boxes_by_pipeline_fov(
                corners=inf_pred_c_v,
                centers=inf_pred_cent_v,
                labels=dummy,
                pts_pipeline_xyz=out_v["pts_reduced_pipeline"],
            )
            inf_pred_s = inf_pred_s[keep_infp]
            inf_pred_l = inf_pred_l[keep_infp]

        inf_pred_box7 = cornersN_to_box7N(inf_pred_c_v)

        # Late fusion
        fused_c, fused_cent, fused_s, fused_l_raw = fuse_preds_with_labels(
            veh_corners=veh_pred_c,
            veh_centers=out_v["pred_centers"],
            veh_scores=veh_pred_s,
            veh_labels=veh_pred_l_raw,  # fusion class grouping uses raw model labels
            inf_corners_v=inf_pred_c_v,
            inf_centers_v=inf_pred_c_v.mean(axis=1) if inf_pred_c_v.shape[0] > 0 else out_i["pred_centers"],
            inf_scores=inf_pred_s,
            inf_labels=inf_pred_l_raw,  # fusion class grouping uses raw model labels
            match_dist_m=float(getattr(V, "MATCH_DIST_M", 2.0)),
        )

        # Now remap fused labels to canonical for evaluation
        fused_l = remap_model_labels_to_canonical(fused_l_raw)

        if bool(getattr(V, "FILTER_FUSED_BY_VEH_FOV", True)) and fused_c.shape[0] > 0:
            dummy = np.zeros((fused_c.shape[0],), dtype=np.int64)
            fused_c2, fused_cent2, dummy, _, keep_f = V.filter_boxes_by_pipeline_fov(
                corners=fused_c,
                centers=fused_cent,
                labels=dummy,
                pts_pipeline_xyz=out_v["pts_reduced_pipeline"],
            )
            fused_s = fused_s[keep_f]
            fused_l = fused_l[keep_f]
            fused_c = fused_c2
            fused_cent = fused_cent2

        fused_box7 = cornersN_to_box7N(fused_c)

        if args.debug_print and (it % 50 == 0):
            print(
                f"[DBG] sid={sid} "
                f"veh_raw={np.unique(veh_pred_l_raw).tolist()} veh={np.unique(veh_pred_l).tolist()} | "
                f"inf_raw={np.unique(inf_pred_l_raw).tolist()} inf={np.unique(inf_pred_l).tolist()} | "
                f"fused_raw={np.unique(fused_l_raw).tolist()} fused={np.unique(fused_l).tolist()}"
            )

        base_record = {
            "sample_id": sid,

            # Aliases for eval_lidar_ap_from_dump.py
            "gt_boxes": union_gt_box7,
            "gt_labels": union_gt_l,

            # Also keep the *_3d versions for your own scripts
            "gt_boxes_3d": union_gt_box7,
            "gt_labels_3d": union_gt_l,

            "debug": {
                "idx_v": int(idx_v),
                "idx_i": int(idx_i),
                "veh_gt_count": int(gt_v_c.shape[0]),
                "inf_gt_count": int(gt_i_c_v.shape[0]),
                "union_gt_count": int(union_gt_box7.shape[0]),
                "veh_pred_count": int(veh_pred_box7.shape[0]),
                "inf_pred_count": int(inf_pred_box7.shape[0]),
                "fused_pred_count": int(fused_box7.shape[0]),
                "transform_src": src,
            },
        }

        if "veh" in dumps:
            rec = dict(base_record)
            rec["pred_boxes"] = veh_pred_box7
            rec["pred_scores"] = veh_pred_s
            rec["pred_labels"] = veh_pred_l
            rec["pred_boxes_3d"] = veh_pred_box7
            rec["pred_scores_3d"] = veh_pred_s
            rec["pred_labels_3d"] = veh_pred_l
            dumps["veh"]["samples"].append(rec)

        if "inf" in dumps:
            rec = dict(base_record)
            rec["pred_boxes"] = inf_pred_box7
            rec["pred_scores"] = inf_pred_s
            rec["pred_labels"] = inf_pred_l
            rec["pred_boxes_3d"] = inf_pred_box7
            rec["pred_scores_3d"] = inf_pred_s
            rec["pred_labels_3d"] = inf_pred_l
            dumps["inf"]["samples"].append(rec)

        if "lf" in dumps:
            rec = dict(base_record)
            rec["pred_boxes"] = fused_box7
            rec["pred_scores"] = fused_s
            rec["pred_labels"] = fused_l
            rec["pred_boxes_3d"] = fused_box7
            rec["pred_scores_3d"] = fused_s
            rec["pred_labels_3d"] = fused_l
            dumps["lf"]["samples"].append(rec)

        if (it + 1) % 25 == 0 or (it + 1) == len(common_ids):
            print(
                f"[INFO] processed {it+1}/{len(common_ids)} sid={sid} | "
                f"union_gt={union_gt_box7.shape[0]} veh_pred={veh_pred_box7.shape[0]} "
                f"inf_pred={inf_pred_box7.shape[0]} fused={fused_box7.shape[0]}"
            )

    for name, path in dump_paths.items():
        save_dump(path, dumps[name])
        print(f"[INFO] wrote: {path}  (samples={len(dumps[name]['samples'])})")

    if args.run_eval:
        repo_root = Path.cwd()
        eval_script = repo_root / "tools" / "eval_lidar_ap_from_dump.py"
        table_script = repo_root / "tools" / "print_metrics_table.py"

        if not eval_script.is_file():
            raise FileNotFoundError(f"Missing eval script: {eval_script}")

        for name, path in dump_paths.items():
            cmd = [sys.executable, str(eval_script), str(path), "--device", str(args.device)]
            rc = run_cmd(cmd, cwd=repo_root)
            if rc != 0:
                raise RuntimeError(f"Eval failed for {name} with code {rc}")

        if table_script.is_file():
            rc = run_cmd([sys.executable, str(table_script)], cwd=repo_root)
            if rc != 0:
                raise RuntimeError(f"print_metrics_table.py failed with code {rc}")
        else:
            print(f"[WARN] missing: {table_script} (skipping consolidated table)")

    print("[DONE]")


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    main()
