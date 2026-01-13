#!/usr/bin/env python3
"""
tools/vis_lidar_late_fusion_single_sample.py

Open3D visual checker for DAIR-V2X-style LiDAR late fusion baseline:
- Runs Vehicle detector and Infra detector on paired sample-id (paired by LiDAR filename stem).
- Visualizes everything in VEHICLE LiDAR frame.

Point clouds (all in vehicle frame):
  - vehicle FULL velodyne (optional)
  - vehicle REDUCED velodyne_reduced (pipeline points; always available)  -> rendered as COLORED CUBES (VoxelGrid)
  - infra FULL velodyne transformed into vehicle frame (optional)
  - infra REDUCED velodyne_reduced transformed into vehicle frame (pipeline points; always available) -> rendered as COLORED CUBES (VoxelGrid)

Boxes (all in vehicle frame):
  - vehicle GT boxes (filtered by vehicle reduced-LiDAR FOV)
  - infra GT boxes transformed into vehicle frame (filtered by infra reduced-LiDAR FOV)
  - vehicle preds (filtered by vehicle reduced-LiDAR FOV)
  - infra preds transformed into vehicle frame (filtered by infra reduced-LiDAR FOV)
  - fused preds after Hungarian matching, optionally filtered by vehicle reduced-LiDAR FOV

IMPORTANT FIX:
- The old version filtered GT by CENTER angles only, with a loose FOV estimate.
- This version matches tools/vis_single_sample_pred_gt.py semantics:
    * infer reduced-LiDAR FOV envelope from PIPELINE points (az, el, range)
    * keep a box if ANY corner is inside that envelope.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import open3d as o3d
except Exception as e:
    raise RuntimeError("Open3D is required. Try: pip install open3d") from e

from mmengine.config import Config
from mmengine.registry import DATASETS
from mmengine.dataset import pseudo_collate

from mmdet3d.apis import init_model
from mmdet3d.utils import register_all_modules


# ============================================================
# VIS SETTINGS (defaults)
# ============================================================

PIPELINE_MAX_POINTS_REDUCED = 250000
PIPELINE_VOXEL_REDUCED = 0.05

PIPELINE_MAX_POINTS_FULL = 250000
PIPELINE_VOXEL_FULL = 0.08

VEH_FULL_POINTS_COLOR      = (0.20, 0.20, 0.20)  # dark gray
INF_FULL_POINTS_COLOR      = (0.05, 0.20, 0.60)  # dark blue

VEH_REDUCED_POINTS_COLOR   = (0.00, 1.00, 1.00)  # bright cyan
INF_REDUCED_POINTS_COLOR   = (1.00, 1.00, 0.00)  # bright yellow

# Make GT colors vastly different from each other and from the warm colors.
GT_VEH_COLOR               = (0.00, 1.00, 0.00)  # pure bright green
GT_INF_COLOR               = (0.10, 0.10, 1.00)  # bright blue

PRED_VEH_COLOR             = (1.00, 0.20, 0.20)  # red
PRED_INF_COLOR             = (1.00, 0.10, 0.80)  # magenta
PRED_FUSED_COLOR           = (1.00, 0.65, 0.05)  # orange

RENDER_VEH_REDUCED_AS_VOXELGRID = True
RENDER_INF_REDUCED_AS_VOXELGRID = True

VEH_REDUCED_VOXELGRID_SIZE = 0.20
INF_REDUCED_VOXELGRID_SIZE = 0.20

RENDER_VEH_FULL_AS_VOXELGRID = False
RENDER_INF_FULL_AS_VOXELGRID = False
VEH_FULL_VOXELGRID_SIZE = 0.12
INF_FULL_VOXELGRID_SIZE = 0.12

# -------------------------
# BOX VISIBILITY (THICK LINES)
# -------------------------
# Open3D LineSet line width is often ignored depending on backend/OS.
# To make boxes "perfectly visible", we render boxes as thick cylinders along edges.
USE_THICK_BOX_EDGES = True
BOX_EDGE_RADIUS_M = 0.035          # thickness in meters (increase if needed)
BOX_EDGE_CYL_RES = 10              # cylinder resolution (higher = smoother)
# Fallback: also try setting render_option.line_width where supported.
TRY_SET_LINE_WIDTH = True
LINE_WIDTH_FALLBACK = 6.0


# ============================================================
# FOV SETTINGS (reduced LiDAR)  [FIXED TO MATCH vis_single_sample_pred_gt.py]
# ============================================================

FOV_SUBSAMPLE_MAX_POINTS = 120000
FOV_AZ_COVERAGE = 0.999

FOV_EL_LOW_PCT  = 0.001
FOV_EL_HIGH_PCT = 0.999
FOV_R_LOW_PCT   = 0.001
FOV_R_HIGH_PCT  = 0.999

FOV_AZ_MARGIN_DEG = 1.0
FOV_EL_MARGIN_DEG = 1.0
FOV_R_MARGIN_M    = 0.5

FOV_REQUIRE_X_POSITIVE = False

MIN_PTS_FOR_FOV = 64

FILTER_GT_BY_FOV = True
FILTER_PRED_BY_FOV = True
FILTER_FUSED_BY_VEH_FOV = True

FILTER_INF_IN_VEH_FRAME_BY_VEH_FOV = False


# ============================================================
# LATE FUSION SETTINGS
# ============================================================

MATCH_DIST_M = 2.0
FUSE_POLICY = "pick_best"  # when matched, pick the higher-score box


# ============================================================
# Legend
# ============================================================

def print_legend() -> None:
    print("[LEGEND]")
    print("  Vehicle FULL points ............ VEH_FULL_POINTS_COLOR (dark gray)")
    print("  Vehicle REDUCED cubes .......... VEH_REDUCED_POINTS_COLOR (bright cyan)  [pipeline points]")
    print("  Infra FULL points -> veh ....... INF_FULL_POINTS_COLOR (dark blue)")
    print("  Infra REDUCED cubes -> veh ..... INF_REDUCED_POINTS_COLOR (bright yellow) [pipeline points]")
    print("  Vehicle GT boxes ............... GT_VEH_COLOR (bright green)")
    print("  Infra GT boxes -> veh .......... GT_INF_COLOR (bright blue)")
    print("  Vehicle predictions ............ PRED_VEH_COLOR (red)")
    print("  Infra predictions -> veh ....... PRED_INF_COLOR (magenta)")
    print("  Fused predictions .............. PRED_FUSED_COLOR (orange)")
    print("[CUBE SIZES]")
    print(f"  VEH_REDUCED_VOXELGRID_SIZE={VEH_REDUCED_VOXELGRID_SIZE}")
    print(f"  INF_REDUCED_VOXELGRID_SIZE={INF_REDUCED_VOXELGRID_SIZE}")
    print("[BOX THICKNESS]")
    print(f"  USE_THICK_BOX_EDGES={USE_THICK_BOX_EDGES} BOX_EDGE_RADIUS_M={BOX_EDGE_RADIUS_M}")


# -------------------------
# Dataset utilities
# -------------------------

def ensure_full_init(dataset) -> None:
    if hasattr(dataset, "full_init"):
        try:
            dataset.full_init()
        except Exception:
            pass

def _get_data_list(dataset) -> Optional[List[Dict]]:
    if hasattr(dataset, "data_list") and isinstance(dataset.data_list, list):
        return dataset.data_list
    if hasattr(dataset, "infos") and isinstance(dataset.infos, list):
        return dataset.infos
    return None

def _get_data_info(dataset, idx: int) -> Optional[Dict]:
    if hasattr(dataset, "get_data_info"):
        try:
            return dataset.get_data_info(idx)
        except Exception:
            return None
    dl = _get_data_list(dataset)
    if dl is not None and 0 <= idx < len(dl):
        return dl[idx]
    return None


# -------------------------
# Point utilities
# -------------------------

def cap_points(pts_xyz: np.ndarray, max_points: int) -> np.ndarray:
    if max_points is None or max_points <= 0 or pts_xyz.shape[0] <= max_points:
        return pts_xyz
    sel = np.random.choice(pts_xyz.shape[0], size=max_points, replace=False)
    return pts_xyz[sel]

def voxel_downsample_points(pts_xyz: np.ndarray, voxel: float) -> np.ndarray:
    if voxel is None or voxel <= 0:
        return pts_xyz
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_xyz)
    pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
    return np.asarray(pcd.points)

def extract_points_from_batch(data_batch: Dict) -> np.ndarray:
    inputs = data_batch.get("inputs", None)
    pts = None

    if isinstance(inputs, dict) and "points" in inputs:
        p = inputs["points"]
        pts = p[0] if isinstance(p, (list, tuple)) else p
    elif isinstance(inputs, (list, tuple)):
        pts = inputs[0]
    else:
        pts = inputs

    if pts is None:
        raise RuntimeError("Could not extract points from data_batch['inputs'].")

    arr = pts if isinstance(pts, np.ndarray) else pts.detach().cpu().numpy()
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise RuntimeError(f"Unexpected points shape: {arr.shape}")
    return arr[:, :3].astype(np.float64)

def resolve_existing_path(p: Path) -> Optional[Path]:
    if p.is_file():
        return p
    q = Path.cwd() / p
    if q.is_file():
        return q
    return None

def load_kitti_bin_xyz(bin_path: Path) -> np.ndarray:
    bp = resolve_existing_path(bin_path)
    if bp is None:
        raise FileNotFoundError(f"Missing bin: {bin_path}")
    pts = np.fromfile(str(bp), dtype=np.float32)
    if pts.size % 4 != 0:
        raise ValueError(f"Unexpected KITTI bin size (not divisible by 4): {bp} (floats={pts.size})")
    pts = pts.reshape(-1, 4)[:, :3].astype(np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts

def guess_full_from_reduced_path(reduced_path: Path) -> Optional[Path]:
    rp = Path(str(reduced_path))
    s = rp.as_posix()

    if "velodyne_reduced" in s:
        fp = Path(s.replace("velodyne_reduced", "velodyne"))
        return resolve_existing_path(fp)

    if "/velodyne/" in s:
        return resolve_existing_path(rp)

    return None

def info_lidar_path(info: Dict) -> Optional[Path]:
    if not isinstance(info, dict):
        return None
    lp = info.get("lidar_points", {})
    if isinstance(lp, dict):
        p = lp.get("lidar_path") or lp.get("pts_path")
        if p:
            return Path(str(p))
    p = info.get("lidar_path")
    if p:
        return Path(str(p))
    return None


def make_pcd(pts: np.ndarray, color_rgb) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.paint_uniform_color(color_rgb)
    return pcd

def make_voxelgrid_from_points(pts: np.ndarray, color_rgb, voxel_size: float) -> o3d.geometry.VoxelGrid:
    pcd = make_pcd(pts, color_rgb)
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=float(voxel_size))
    return vg


# -------------------------
# Pairing ID: LiDAR filename stem
# -------------------------

def norm_sample_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return f"{int(x):06d}"
    s = str(x).strip()
    try:
        stem = Path(s).stem
    except Exception:
        stem = s
    if stem.isdigit():
        return stem.zfill(6) if len(stem) <= 6 else stem
    import re
    m = re.search(r"(\d{6})", stem)
    if m:
        return m.group(1)
    m = re.search(r"(\d{6})", s)
    if m:
        return m.group(1)
    return None

def extract_pair_id_from_info(info: Dict) -> Optional[str]:
    if not isinstance(info, dict):
        return None
    lp = info.get("lidar_points", {})
    cand = []
    if isinstance(lp, dict):
        cand.append(lp.get("lidar_path"))
        cand.append(lp.get("pts_path"))
    cand.append(info.get("lidar_path"))
    cand.append(info.get("sample_idx"))
    cand.append(info.get("sample_id"))
    for x in cand:
        sid = norm_sample_id(x)
        if sid is not None:
            return sid
    return None

def build_pairid_to_idx(dataset) -> Dict[str, int]:
    ensure_full_init(dataset)
    m: Dict[str, int] = {}
    dl = _get_data_list(dataset)
    if isinstance(dl, list) and len(dl) == len(dataset):
        for i, info in enumerate(dl):
            sid = extract_pair_id_from_info(info if isinstance(info, dict) else {})
            if sid is not None and sid not in m:
                m[sid] = i
        return m
    for i in range(len(dataset)):
        info = _get_data_info(dataset, i) or {}
        sid = extract_pair_id_from_info(info)
        if sid is not None and sid not in m:
            m[sid] = i
    return m

def build_common_ids_in_vehicle_order(ids_v: Dict[str, int], ids_i: Dict[str, int]) -> List[str]:
    inv_v = sorted(((idx, sid) for sid, idx in ids_v.items()), key=lambda x: x[0])
    out = []
    for _, sid in inv_v:
        if sid in ids_i:
            out.append(sid)
    return out


# -------------------------
# Boxes extraction from MMDet3D outputs + eval_ann_info
# -------------------------

def pred_from_output(pred_sample, score_thr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_instances = getattr(pred_sample, "pred_instances_3d", None)
    if pred_instances is None or not hasattr(pred_instances, "bboxes_3d"):
        return (np.zeros((0, 8, 3), np.float64),
                np.zeros((0, 3), np.float64),
                np.zeros((0,), np.float64),
                np.zeros((0,), np.int64))

    boxes = pred_instances.bboxes_3d
    if boxes is None or not hasattr(boxes, "corners") or not hasattr(boxes, "tensor"):
        return (np.zeros((0, 8, 3), np.float64),
                np.zeros((0, 3), np.float64),
                np.zeros((0,), np.float64),
                np.zeros((0,), np.int64))

    corners = boxes.corners.detach().cpu().numpy().astype(np.float64)
    centers = boxes.tensor.detach().cpu().numpy().astype(np.float64)[:, :3]

    scores = getattr(pred_instances, "scores_3d", None)
    labels = getattr(pred_instances, "labels_3d", None)

    if scores is None:
        scores_np = np.ones((corners.shape[0],), dtype=np.float64)
    else:
        scores_np = scores.detach().cpu().numpy().astype(np.float64) if not isinstance(scores, np.ndarray) else scores.astype(np.float64)

    if labels is None:
        labels_np = np.zeros((corners.shape[0],), dtype=np.int64)
    else:
        labels_np = labels.detach().cpu().numpy().astype(np.int64) if not isinstance(labels, np.ndarray) else labels.astype(np.int64)

    keep = scores_np >= float(score_thr)
    return corners[keep], centers[keep], scores_np[keep], labels_np[keep]

def gt_from_eval_ann_info(dataset, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ensure_full_init(dataset)
    info = _get_data_info(dataset, idx)
    if not isinstance(info, dict):
        return (np.zeros((0, 8, 3), np.float64),
                np.zeros((0, 3), np.float64),
                np.zeros((0,), np.int64))

    ea = info.get("eval_ann_info", None)
    if not isinstance(ea, dict):
        return (np.zeros((0, 8, 3), np.float64),
                np.zeros((0, 3), np.float64),
                np.zeros((0,), np.int64))

    gt_boxes = ea.get("gt_bboxes_3d", None)
    gt_labels = ea.get("gt_labels_3d", None)
    if gt_boxes is None or not hasattr(gt_boxes, "corners") or not hasattr(gt_boxes, "tensor"):
        return (np.zeros((0, 8, 3), np.float64),
                np.zeros((0, 3), np.float64),
                np.zeros((0,), np.int64))

    corners = gt_boxes.corners.detach().cpu().numpy().astype(np.float64)
    centers = gt_boxes.tensor.detach().cpu().numpy().astype(np.float64)[:, :3]
    if gt_labels is None:
        labels = np.zeros((corners.shape[0],), dtype=np.int64)
    else:
        labels = gt_labels.detach().cpu().numpy().astype(np.int64) if not isinstance(gt_labels, np.ndarray) else gt_labels.astype(np.int64)
    return corners, centers, labels


# ============================================================
# FOV envelope from pipeline points (az, el, range) + corner-based box filter
# ============================================================

def _subsample_rows(x: np.ndarray, max_rows: int) -> np.ndarray:
    if max_rows is None or max_rows <= 0 or x.shape[0] <= max_rows:
        return x
    sel = np.random.choice(x.shape[0], size=max_rows, replace=False)
    return x[sel]

def _minimal_circular_interval_covering(angles_rad: np.ndarray, coverage: float) -> Tuple[float, float]:
    a = np.mod(angles_rad, 2.0 * np.pi)
    a = np.sort(a)
    n = a.size
    if n == 0:
        return 0.0, 0.0

    k = int(np.ceil(float(coverage) * n))
    k = max(1, min(k, n))

    a2 = np.concatenate([a, a + 2.0 * np.pi])

    best_len = 1e18
    best_lo = a[0]
    best_hi = a[0]

    for i in range(n):
        j = i + k - 1
        lo = a2[i]
        hi = a2[j]
        span = hi - lo
        if span < best_len:
            best_len = span
            best_lo = lo
            best_hi = hi

    lo = float(np.mod(best_lo, 2.0 * np.pi))
    hi = float(np.mod(best_hi, 2.0 * np.pi))
    return lo, hi

def _angle_in_interval(ang: np.ndarray, lo: float, hi: float) -> np.ndarray:
    ang = np.mod(ang, 2.0 * np.pi)
    lo = float(np.mod(lo, 2.0 * np.pi))
    hi = float(np.mod(hi, 2.0 * np.pi))
    if lo <= hi:
        return (ang >= lo) & (ang <= hi)
    else:
        return (ang >= lo) | (ang <= hi)

def compute_pipeline_fov(pts_xyz: np.ndarray) -> Dict[str, float]:
    if pts_xyz is None or pts_xyz.shape[0] < MIN_PTS_FOR_FOV:
        return {
            "az_lo": 0.0, "az_hi": 0.0,
            "el_lo": -0.5*np.pi, "el_hi": 0.5*np.pi,
            "r_lo": 0.0, "r_hi": 1e9,
        }

    p = _subsample_rows(pts_xyz, FOV_SUBSAMPLE_MAX_POINTS)

    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]

    r = np.sqrt(x * x + y * y + z * z) + 1e-9
    az = np.arctan2(y, x)
    el = np.arcsin(np.clip(z / r, -1.0, 1.0))

    az_lo, az_hi = _minimal_circular_interval_covering(az, coverage=FOV_AZ_COVERAGE)
    el_lo = float(np.quantile(el, FOV_EL_LOW_PCT))
    el_hi = float(np.quantile(el, FOV_EL_HIGH_PCT))
    r_lo  = float(np.quantile(r,  FOV_R_LOW_PCT))
    r_hi  = float(np.quantile(r,  FOV_R_HIGH_PCT))

    return {"az_lo": az_lo, "az_hi": az_hi, "el_lo": el_lo, "el_hi": el_hi, "r_lo": r_lo, "r_hi": r_hi}

def filter_boxes_by_pipeline_fov(
    corners: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
    pts_pipeline_xyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float], np.ndarray]:
    if corners is None or corners.shape[0] == 0:
        return corners, centers, labels, {}, np.zeros((0,), dtype=bool)
    if pts_pipeline_xyz is None or pts_pipeline_xyz.shape[0] < MIN_PTS_FOR_FOV:
        keep = np.ones((corners.shape[0],), dtype=bool)
        return corners, centers, labels, {"note": "FOV fallback (too few points)"}, keep

    fov = compute_pipeline_fov(pts_pipeline_xyz)

    az_margin = np.deg2rad(float(FOV_AZ_MARGIN_DEG))
    el_margin = np.deg2rad(float(FOV_EL_MARGIN_DEG))

    az_lo = float(np.mod(fov["az_lo"] - az_margin, 2.0 * np.pi))
    az_hi = float(np.mod(fov["az_hi"] + az_margin, 2.0 * np.pi))
    el_lo = float(fov["el_lo"] - el_margin)
    el_hi = float(fov["el_hi"] + el_margin)
    r_lo  = float(max(0.0, fov["r_lo"] - float(FOV_R_MARGIN_M)))
    r_hi  = float(fov["r_hi"] + float(FOV_R_MARGIN_M))

    c = np.asarray(corners, dtype=np.float64)
    x = c[:, :, 0]
    y = c[:, :, 1]
    z = c[:, :, 2]
    r = np.sqrt(x * x + y * y + z * z) + 1e-9
    az = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    el = np.arcsin(np.clip(z / r, -1.0, 1.0))

    in_az = _angle_in_interval(az, az_lo, az_hi)
    in_el = (el >= el_lo) & (el <= el_hi)
    in_r  = (r  >= r_lo)  & (r  <= r_hi)

    in_all = in_az & in_el & in_r
    if FOV_REQUIRE_X_POSITIVE:
        in_all = in_all & (x > 0.0)

    keep = np.any(in_all, axis=1)

    fov_dbg = {
        "az_lo_deg": float(np.rad2deg(az_lo)),
        "az_hi_deg": float(np.rad2deg(az_hi)),
        "el_lo_deg": float(np.rad2deg(el_lo)),
        "el_hi_deg": float(np.rad2deg(el_hi)),
        "r_lo_m": float(r_lo),
        "r_hi_m": float(r_hi),
        "az_coverage": float(FOV_AZ_COVERAGE),
    }

    return corners[keep], centers[keep], labels[keep], fov_dbg, keep


# -------------------------
# NON_KITTI_ROOT JSON transforms
# -------------------------

def _as_np(x):
    return np.array(x, dtype=np.float64)

def quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n <= 0:
        raise ValueError("Invalid quaternion norm")
    qx, qy, qz, qw = q / n

    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)]
    ], dtype=np.float64)

def parse_rigid_json(json_path: Path) -> np.ndarray:
    jp = resolve_existing_path(json_path) or json_path
    if not jp.is_file():
        raise FileNotFoundError(f"Missing transform json: {json_path}")

    obj = json.loads(jp.read_text())

    candidates = [obj]
    for k in ["transform", "Tr", "calib", "data", "extrinsic", "extrinsics"]:
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
            candidates.append(obj[k])

    def find_key(d, keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    R = None
    t = None

    for d in candidates:
        if not isinstance(d, dict):
            continue

        R_raw = find_key(d, ["rotation", "rot", "R"])
        t_raw = find_key(d, ["translation", "trans", "t"])

        if R_raw is not None and t_raw is not None:
            Rn = _as_np(R_raw)
            tn = _as_np(t_raw).reshape(-1)
            if Rn.size == 9:
                Rn = Rn.reshape(3, 3)
            if Rn.shape == (3, 3) and tn.size == 3:
                R, t = Rn, tn
                break

        q_raw = find_key(d, ["quaternion", "quat", "q"])
        if q_raw is not None and t_raw is not None:
            qn = _as_np(q_raw).reshape(-1)
            tn = _as_np(t_raw).reshape(-1)
            if tn.size == 3 and qn.size == 4:
                qx, qy, qz, qw = qn
                if abs(qn[0]) > abs(qn[3]):
                    qw, qx, qy, qz = qn
                R = quat_to_R(qx, qy, qz, qw)
                t = tn
                break

    if R is None or t is None:
        raise ValueError(f"Could not parse rigid transform JSON into (R,t): {jp}")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def apply_T_points(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    if pts_xyz is None or pts_xyz.shape[0] == 0:
        return pts_xyz
    N = pts_xyz.shape[0]
    pts_h = np.ones((N, 4), dtype=np.float64)
    pts_h[:, :3] = pts_xyz.astype(np.float64)
    out = (T @ pts_h.T).T
    return out[:, :3]

def apply_T_corners(T: np.ndarray, corners: np.ndarray) -> np.ndarray:
    if corners is None or corners.shape[0] == 0:
        return corners
    c = corners.reshape(-1, 3)
    c2 = apply_T_points(T, c)
    return c2.reshape(corners.shape[0], 8, 3)

def get_coop_root(non_kitti_root: Path) -> Path:
    cand = non_kitti_root / "cooperative-vehicle-infrastructure"
    if cand.is_dir():
        return cand
    return non_kitti_root

def find_transform_json(root: Path, sample_id: str, must_contain: str) -> Path:
    sample_name = f"{sample_id}.json"
    hits = []
    for p in root.rglob(sample_name):
        s = str(p).lower()
        if must_contain.lower() in s:
            hits.append(p)
    hits = sorted(hits)
    if not hits:
        raise FileNotFoundError(
            f"Could not find {sample_name} under {root} with path containing '{must_contain}'."
        )
    return hits[0]

def compute_T_veh_from_inf_from_json(
    coop_root: Path,
    sid: str,
    veh_novatel_key: str,
    veh_lidar_to_novatel_key: str,
    inf_lidar_to_world_key: str,
    debug_print: bool = False,
) -> Tuple[np.ndarray, Dict[str, str]]:
    veh_novatel_to_world_json = find_transform_json(coop_root, sid, veh_novatel_key)
    veh_lidar_to_novatel_json = find_transform_json(coop_root, sid, veh_lidar_to_novatel_key)
    inf_lidar_to_world_json   = find_transform_json(coop_root, sid, inf_lidar_to_world_key)

    if debug_print:
        print(f"[DEBUG] transform jsons for sid={sid}:")
        print("  veh novatel_to_world:", veh_novatel_to_world_json)
        print("  veh lidar_to_novatel:", veh_lidar_to_novatel_json)
        print("  inf lidar_to_world  :", inf_lidar_to_world_json)

    T_world_from_veh_novatel     = parse_rigid_json(veh_novatel_to_world_json)
    T_veh_novatel_from_veh_lidar = parse_rigid_json(veh_lidar_to_novatel_json)
    T_world_from_veh_lidar       = T_world_from_veh_novatel @ T_veh_novatel_from_veh_lidar

    T_world_from_inf_lidar       = parse_rigid_json(inf_lidar_to_world_json)

    T_veh_from_world = inv_T(T_world_from_veh_lidar)
    T_veh_from_inf   = T_veh_from_world @ T_world_from_inf_lidar

    src = {
        "veh_novatel_to_world": str(veh_novatel_to_world_json),
        "veh_lidar_to_novatel": str(veh_lidar_to_novatel_json),
        "inf_lidar_to_world": str(inf_lidar_to_world_json),
    }
    return T_veh_from_inf, src


# -------------------------
# Late fusion: Hungarian matching
# -------------------------

def hungarian_min_cost(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return r, c
    except Exception:
        r_used = set()
        c_used = set()
        pairs: List[Tuple[int, int]] = []
        flat = [(cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
        flat.sort(key=lambda x: x[0])
        for _, i, j in flat:
            if i in r_used or j in c_used:
                continue
            r_used.add(i)
            c_used.add(j)
            pairs.append((i, j))
        if len(pairs) == 0:
            return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)
        rr = np.array([p[0] for p in pairs], dtype=int)
        cc = np.array([p[1] for p in pairs], dtype=int)
        return rr, cc

def fuse_preds(
    veh_corners: np.ndarray, veh_centers: np.ndarray, veh_scores: np.ndarray, veh_labels: np.ndarray,
    inf_corners_v: np.ndarray, inf_centers_v: np.ndarray, inf_scores: np.ndarray, inf_labels: np.ndarray,
    match_dist_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if veh_centers.shape[0] == 0 and inf_centers_v.shape[0] == 0:
        return np.zeros((0, 8, 3), np.float64), np.zeros((0, 3), np.float64), np.zeros((0,), np.float64)
    if veh_centers.shape[0] == 0:
        return inf_corners_v, inf_centers_v, inf_scores
    if inf_centers_v.shape[0] == 0:
        return veh_corners, veh_centers, veh_scores

    fused_corners: List[np.ndarray] = []
    fused_centers: List[np.ndarray] = []
    fused_scores: List[float] = []

    classes = np.unique(np.concatenate([veh_labels, inf_labels], axis=0)) if (veh_labels.size + inf_labels.size) > 0 else np.array([], dtype=int)

    for cls in classes:
        idx_v = np.where(veh_labels == cls)[0]
        idx_i = np.where(inf_labels == cls)[0]

        if idx_v.size == 0:
            for j in idx_i:
                fused_corners.append(inf_corners_v[j])
                fused_centers.append(inf_centers_v[j])
                fused_scores.append(float(inf_scores[j]))
            continue
        if idx_i.size == 0:
            for i in idx_v:
                fused_corners.append(veh_corners[i])
                fused_centers.append(veh_centers[i])
                fused_scores.append(float(veh_scores[i]))
            continue

        V = veh_centers[idx_v]
        I = inf_centers_v[idx_i]
        cost = np.linalg.norm(V[:, None, :] - I[None, :, :], axis=2)
        r, c = hungarian_min_cost(cost)

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

            if FUSE_POLICY == "pick_best":
                if float(veh_scores[i]) >= float(inf_scores[j]):
                    fused_corners.append(veh_corners[i])
                    fused_centers.append(veh_centers[i])
                    fused_scores.append(float(veh_scores[i]))
                else:
                    fused_corners.append(inf_corners_v[j])
                    fused_centers.append(inf_centers_v[j])
                    fused_scores.append(float(inf_scores[j]))
            else:
                fused_corners.append(veh_corners[i])
                fused_centers.append(veh_centers[i])
                fused_scores.append(float(veh_scores[i]))

        for rr, i in enumerate(idx_v):
            if not used_v[rr]:
                fused_corners.append(veh_corners[i])
                fused_centers.append(veh_centers[i])
                fused_scores.append(float(veh_scores[i]))
        for cc, j in enumerate(idx_i):
            if not used_i[cc]:
                fused_corners.append(inf_corners_v[j])
                fused_centers.append(inf_centers_v[j])
                fused_scores.append(float(inf_scores[j]))

    if len(fused_corners) == 0:
        return np.zeros((0, 8, 3), np.float64), np.zeros((0, 3), np.float64), np.zeros((0,), np.float64)

    return (
        np.stack(fused_corners, axis=0).astype(np.float64),
        np.stack(fused_centers, axis=0).astype(np.float64),
        np.array(fused_scores, dtype=np.float64),
    )


# -------------------------
# Open3D box rendering (THICK EDGES)
# -------------------------

_BOX_EDGES = np.array(
    [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)

def corners_to_lineset(corners_8x3: np.ndarray, color_rgb) -> o3d.geometry.LineSet:
    c = np.asarray(corners_8x3, dtype=np.float64).reshape(8, 3)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(c)
    ls.lines = o3d.utility.Vector2iVector(_BOX_EDGES)
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(color_rgb, dtype=np.float64), (_BOX_EDGES.shape[0], 1))
    )
    return ls

def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z,  y],
                     [z,  0.0, -x],
                     [-y, x,  0.0]], dtype=np.float64)

def _rot_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0:
        c = 1.0
    if c < -1.0:
        c = -1.0

    if np.linalg.norm(v) < 1e-10:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        # 180 deg, pick an orthogonal axis
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v = np.cross(a, axis)
        v = v / (np.linalg.norm(v) + 1e-12)
        K = _skew(v)
        return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)

    s = float(np.linalg.norm(v))
    v = v / (s + 1e-12)
    K = _skew(v)
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)

def _cylinder_between(p0: np.ndarray, p1: np.ndarray, radius: float, res: int) -> Optional[o3d.geometry.TriangleMesh]:
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-8:
        return None

    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=float(radius), height=float(L), resolution=int(res), split=1)
    cyl.compute_vertex_normals()

    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dirv = d / L
    R = _rot_from_a_to_b(z, dirv)

    cyl.rotate(R, center=np.array([0.0, 0.0, 0.0], dtype=np.float64))
    mid = 0.5 * (p0 + p1)
    cyl.translate(mid)
    return cyl

def corners_to_thick_box_mesh(corners_8x3: np.ndarray, color_rgb, radius: float, res: int) -> o3d.geometry.TriangleMesh:
    c = np.asarray(corners_8x3, dtype=np.float64).reshape(8, 3)
    out = o3d.geometry.TriangleMesh()
    for e in _BOX_EDGES:
        p0 = c[int(e[0])]
        p1 = c[int(e[1])]
        cyl = _cylinder_between(p0, p1, radius=radius, res=res)
        if cyl is not None:
            out += cyl
    out.paint_uniform_color(color_rgb)
    out.compute_vertex_normals()
    return out

def add_box_geom(geoms: List[o3d.geometry.Geometry], corners_8x3: np.ndarray, color_rgb) -> None:
    if USE_THICK_BOX_EDGES:
        try:
            geoms.append(corners_to_thick_box_mesh(corners_8x3, color_rgb, BOX_EDGE_RADIUS_M, BOX_EDGE_CYL_RES))
            return
        except Exception:
            pass
    geoms.append(corners_to_lineset(corners_8x3, color_rgb))


# -------------------------
# One-side inference wrapper
# -------------------------

def run_one_side(dataset, model, idx: int, score_thr: float, want_full_points: bool) -> Dict[str, Any]:
    ensure_full_init(dataset)
    info = _get_data_info(dataset, idx) or {}

    data = dataset[idx]
    data_batch = pseudo_collate([data])

    with torch.no_grad():
        outputs = model.test_step(data_batch)
    pred_sample = outputs[0]

    pts_reduced_pipeline = extract_points_from_batch(data_batch)

    pts_reduced_vis = cap_points(pts_reduced_pipeline.copy(), PIPELINE_MAX_POINTS_REDUCED)
    pts_reduced_vis = voxel_downsample_points(pts_reduced_vis, PIPELINE_VOXEL_REDUCED)

    pts_full_vis = None
    full_path_used = None
    reduced_path = info_lidar_path(info)
    if want_full_points and reduced_path is not None:
        full_path = guess_full_from_reduced_path(reduced_path)
        if full_path is not None:
            try:
                pts_full = load_kitti_bin_xyz(full_path)
                pts_full = cap_points(pts_full, PIPELINE_MAX_POINTS_FULL)
                pts_full = voxel_downsample_points(pts_full, PIPELINE_VOXEL_FULL)
                pts_full_vis = pts_full
                full_path_used = str(full_path)
            except Exception:
                pts_full_vis = None
                full_path_used = None

    gt_corners, gt_centers, gt_labels = gt_from_eval_ann_info(dataset, idx)
    pred_corners, pred_centers, pred_scores, pred_labels = pred_from_output(pred_sample, score_thr=score_thr)

    fov_dbg = compute_pipeline_fov(pts_reduced_pipeline)
    fov_dbg_printable = {
        "az_lo_deg": float(np.rad2deg(fov_dbg["az_lo"])),
        "az_hi_deg": float(np.rad2deg(fov_dbg["az_hi"])),
        "el_lo_deg": float(np.rad2deg(fov_dbg["el_lo"])),
        "el_hi_deg": float(np.rad2deg(fov_dbg["el_hi"])),
        "r_lo_m": float(fov_dbg["r_lo"]),
        "r_hi_m": float(fov_dbg["r_hi"]),
        "az_coverage": float(FOV_AZ_COVERAGE),
    }

    if FILTER_GT_BY_FOV and gt_corners.shape[0] > 0:
        gt_corners, gt_centers, gt_labels, fov_dbg2, _ = filter_boxes_by_pipeline_fov(
            corners=gt_corners,
            centers=gt_centers,
            labels=gt_labels,
            pts_pipeline_xyz=pts_reduced_pipeline,
        )
        if isinstance(fov_dbg2, dict) and len(fov_dbg2) > 0:
            fov_dbg_printable = fov_dbg2

    if FILTER_PRED_BY_FOV and pred_corners.shape[0] > 0:
        pred_corners2, pred_centers2, pred_labels2, _, keep_pred = filter_boxes_by_pipeline_fov(
            corners=pred_corners,
            centers=pred_centers,
            labels=pred_labels,
            pts_pipeline_xyz=pts_reduced_pipeline,
        )
        pred_scores = pred_scores[keep_pred]
        pred_corners, pred_centers, pred_labels = pred_corners2, pred_centers2, pred_labels2

    return {
        "info": info,
        "reduced_path": str(reduced_path) if reduced_path is not None else None,
        "full_path": full_path_used,
        "pts_reduced_pipeline": pts_reduced_pipeline,
        "pts_reduced_vis": pts_reduced_vis,
        "pts_full_vis": pts_full_vis,
        "fov_dbg": fov_dbg_printable,
        "gt_corners": gt_corners,
        "gt_centers": gt_centers,
        "gt_labels": gt_labels,
        "pred_corners": pred_corners,
        "pred_centers": pred_centers,
        "pred_scores": pred_scores,
        "pred_labels": pred_labels,
    }


# -------------------------
# Scene builder for a pair-id
# -------------------------

def build_scene_for_sid(
    sid: str,
    dataset_v,
    dataset_i,
    id2idx_v: Dict[str, int],
    id2idx_i: Dict[str, int],
    model_v,
    model_i,
    score_thr: float,
    coop_root: Path,
    veh_novatel_key: str,
    veh_lidar_to_novatel_key: str,
    inf_lidar_to_world_key: str,
    T_cache: Dict[str, Tuple[np.ndarray, Dict[str, str]]],
    show_veh_full: bool,
    show_veh_reduced: bool,
    show_inf_full: bool,
    show_inf_reduced: bool,
    debug_print: bool,
    only_veh: bool,
    only_inf: bool,
) -> Tuple[List[o3d.geometry.Geometry], Dict[str, Any]]:
    idx_v = id2idx_v[sid]
    idx_i = id2idx_i[sid]

    out_v = run_one_side(dataset_v, model_v, idx_v, score_thr=score_thr, want_full_points=show_veh_full)
    out_i = run_one_side(dataset_i, model_i, idx_i, score_thr=score_thr, want_full_points=show_inf_full)

    if sid in T_cache:
        T_veh_from_inf, src = T_cache[sid]
    else:
        T_veh_from_inf, src = compute_T_veh_from_inf_from_json(
            coop_root=coop_root,
            sid=sid,
            veh_novatel_key=veh_novatel_key,
            veh_lidar_to_novatel_key=veh_lidar_to_novatel_key,
            inf_lidar_to_world_key=inf_lidar_to_world_key,
            debug_print=debug_print,
        )
        T_cache[sid] = (T_veh_from_inf, src)

    draw_veh_pred = (not only_inf)
    draw_inf_pred = (not only_veh)
    draw_fused = (not (only_veh or only_inf))

    geoms: List[o3d.geometry.Geometry] = []
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0]))

    if show_veh_full and out_v["pts_full_vis"] is not None and out_v["pts_full_vis"].shape[0] > 0:
        if RENDER_VEH_FULL_AS_VOXELGRID:
            geoms.append(make_voxelgrid_from_points(out_v["pts_full_vis"], VEH_FULL_POINTS_COLOR, VEH_FULL_VOXELGRID_SIZE))
        else:
            geoms.append(make_pcd(out_v["pts_full_vis"], VEH_FULL_POINTS_COLOR))

    if show_veh_reduced and out_v["pts_reduced_vis"] is not None and out_v["pts_reduced_vis"].shape[0] > 0:
        if RENDER_VEH_REDUCED_AS_VOXELGRID:
            geoms.append(make_voxelgrid_from_points(out_v["pts_reduced_vis"], VEH_REDUCED_POINTS_COLOR, VEH_REDUCED_VOXELGRID_SIZE))
        else:
            geoms.append(make_pcd(out_v["pts_reduced_vis"], VEH_REDUCED_POINTS_COLOR))

    if show_inf_full and out_i["pts_full_vis"] is not None and out_i["pts_full_vis"].shape[0] > 0:
        inf_full_v = apply_T_points(T_veh_from_inf, out_i["pts_full_vis"])
        if RENDER_INF_FULL_AS_VOXELGRID:
            geoms.append(make_voxelgrid_from_points(inf_full_v, INF_FULL_POINTS_COLOR, INF_FULL_VOXELGRID_SIZE))
        else:
            geoms.append(make_pcd(inf_full_v, INF_FULL_POINTS_COLOR))

    if show_inf_reduced and out_i["pts_reduced_vis"] is not None and out_i["pts_reduced_vis"].shape[0] > 0:
        inf_red_v = apply_T_points(T_veh_from_inf, out_i["pts_reduced_vis"])
        if RENDER_INF_REDUCED_AS_VOXELGRID:
            geoms.append(make_voxelgrid_from_points(inf_red_v, INF_REDUCED_POINTS_COLOR, INF_REDUCED_VOXELGRID_SIZE))
        else:
            geoms.append(make_pcd(inf_red_v, INF_REDUCED_POINTS_COLOR))

    # GT boxes
    for k in range(out_v["gt_corners"].shape[0]):
        add_box_geom(geoms, out_v["gt_corners"][k], GT_VEH_COLOR)

    inf_gt_corners_v = apply_T_corners(T_veh_from_inf, out_i["gt_corners"])
    inf_gt_centers_v = apply_T_points(T_veh_from_inf, out_i["gt_centers"]) if out_i["gt_centers"].shape[0] > 0 else out_i["gt_centers"]
    if FILTER_INF_IN_VEH_FRAME_BY_VEH_FOV and inf_gt_corners_v.shape[0] > 0:
        dummy_labels = np.zeros((inf_gt_corners_v.shape[0],), dtype=np.int64)
        inf_gt_corners_v, inf_gt_centers_v, _, _, _ = filter_boxes_by_pipeline_fov(
            corners=inf_gt_corners_v,
            centers=inf_gt_centers_v,
            labels=dummy_labels,
            pts_pipeline_xyz=out_v["pts_reduced_pipeline"],
        )
    for k in range(inf_gt_corners_v.shape[0]):
        add_box_geom(geoms, inf_gt_corners_v[k], GT_INF_COLOR)

    # Predictions
    if draw_veh_pred:
        for k in range(out_v["pred_corners"].shape[0]):
            add_box_geom(geoms, out_v["pred_corners"][k], PRED_VEH_COLOR)

    inf_pred_corners_v = apply_T_corners(T_veh_from_inf, out_i["pred_corners"])
    inf_pred_centers_v = apply_T_points(T_veh_from_inf, out_i["pred_centers"]) if out_i["pred_centers"].shape[0] > 0 else out_i["pred_centers"]
    if FILTER_INF_IN_VEH_FRAME_BY_VEH_FOV and inf_pred_corners_v.shape[0] > 0:
        dummy_labels = np.zeros((inf_pred_corners_v.shape[0],), dtype=np.int64)
        inf_pred_corners_v, inf_pred_centers_v, _, _, keep_inf_pr = filter_boxes_by_pipeline_fov(
            corners=inf_pred_corners_v,
            centers=inf_pred_centers_v,
            labels=dummy_labels,
            pts_pipeline_xyz=out_v["pts_reduced_pipeline"],
        )
        out_i["pred_scores"] = out_i["pred_scores"][keep_inf_pr]
        out_i["pred_labels"] = out_i["pred_labels"][keep_inf_pr]

    if draw_inf_pred:
        for k in range(inf_pred_corners_v.shape[0]):
            add_box_geom(geoms, inf_pred_corners_v[k], PRED_INF_COLOR)

    # Fused
    if draw_fused:
        fused_corners, fused_centers, fused_scores = fuse_preds(
            veh_corners=out_v["pred_corners"],
            veh_centers=out_v["pred_centers"],
            veh_scores=out_v["pred_scores"],
            veh_labels=out_v["pred_labels"],
            inf_corners_v=inf_pred_corners_v,
            inf_centers_v=inf_pred_centers_v,
            inf_scores=out_i["pred_scores"],
            inf_labels=out_i["pred_labels"],
            match_dist_m=MATCH_DIST_M,
        )

        if FILTER_FUSED_BY_VEH_FOV and fused_corners.shape[0] > 0:
            dummy_labels = np.zeros((fused_corners.shape[0],), dtype=np.int64)
            fused_corners, fused_centers, _, _, keep_fused = filter_boxes_by_pipeline_fov(
                corners=fused_corners,
                centers=fused_centers,
                labels=dummy_labels,
                pts_pipeline_xyz=out_v["pts_reduced_pipeline"],
            )
            fused_scores = fused_scores[keep_fused]

        for k in range(fused_corners.shape[0]):
            add_box_geom(geoms, fused_corners[k], PRED_FUSED_COLOR)
        fused_pred_count = int(fused_corners.shape[0])
    else:
        fused_pred_count = 0

    dbg = {
        "sid": sid,
        "idx_v": idx_v,
        "idx_i": idx_i,
        "veh_fov": out_v["fov_dbg"],
        "inf_fov": out_i["fov_dbg"],
        "veh_gt": int(out_v["gt_corners"].shape[0]),
        "inf_gt": int(inf_gt_corners_v.shape[0]),
        "veh_pred": int(out_v["pred_corners"].shape[0]) if draw_veh_pred else 0,
        "inf_pred": int(inf_pred_corners_v.shape[0]) if draw_inf_pred else 0,
        "fused_pred": fused_pred_count,
        "transform_src": src,
        "veh_reduced_path": out_v.get("reduced_path"),
        "veh_full_path": out_v.get("full_path"),
        "inf_reduced_path": out_i.get("reduced_path"),
        "inf_full_path": out_i.get("full_path"),
        "only_veh": bool(only_veh),
        "only_inf": bool(only_inf),
    }

    if debug_print:
        print(f"[DEBUG] pair-id={sid} idx_v={idx_v} idx_i={idx_i}")
        print("[DEBUG] veh_fov:", dbg["veh_fov"])
        print("[DEBUG] inf_fov:", dbg["inf_fov"])
        print("[DEBUG] transform src:", dbg["transform_src"])
        print("[DEBUG] veh reduced path:", dbg["veh_reduced_path"])
        print("[DEBUG] veh full path   :", dbg["veh_full_path"])
        print("[DEBUG] inf reduced path:", dbg["inf_reduced_path"])
        print("[DEBUG] inf full path   :", dbg["inf_full_path"])
        print(f"[DEBUG] counts: veh_gt={dbg['veh_gt']} inf_gt={dbg['inf_gt']} veh_pred={dbg['veh_pred']} inf_pred={dbg['inf_pred']} fused={dbg['fused_pred']}")

    return geoms, dbg


def print_help():
    print("Keys:")
    print("  N : next sample (wrap)")
    print("  B : previous sample (wrap)")
    print("  S : save screenshot to --out-dir")
    print("  H : help + status + legend")
    print("  ESC / close window: exit")


# -------------------------
# Main
# -------------------------

def _apply_render_options(vis_like, point_size: float) -> None:
    opt = vis_like.get_render_option()
    if opt is None:
        return
    opt.point_size = float(point_size)
    if TRY_SET_LINE_WIDTH and hasattr(opt, "line_width"):
        try:
            opt.line_width = float(LINE_WIDTH_FALLBACK)
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_vehicle", type=str)
    parser.add_argument("ckpt_vehicle", type=str)
    parser.add_argument("cfg_infra", type=str)
    parser.add_argument("ckpt_infra", type=str)

    parser.add_argument("--sample-id", required=True, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--score-thr", default=0.1, type=float)
    parser.add_argument("--point-size", default=2.0, type=float)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--out-dir", default=None, type=str)
    parser.add_argument("--debug-print", action="store_true")

    parser.add_argument("--non-kitti-root", required=True, type=str)

    parser.add_argument("--veh-novatel-key", default="novatel_to_world", type=str)
    parser.add_argument("--veh-lidar2novatel-key", default="lidar_to_novatel", type=str)
    parser.add_argument("--inf-lidar2world-key", default="virtuallidar_to_world", type=str)

    parser.add_argument("--show-veh-full", action="store_true", help="show vehicle FULL velodyne points (if found)")
    parser.add_argument("--hide-veh-reduced", action="store_true", help="hide vehicle REDUCED (pipeline) points")
    parser.add_argument("--show-inf-full", action="store_true", help="show infra FULL velodyne points (if found)")
    parser.add_argument("--hide-inf-reduced", action="store_true", help="hide infra REDUCED (pipeline) points")

    parser.add_argument("--veh-reduced-cube", default=None, type=float, help="override VEH_REDUCED_VOXELGRID_SIZE")
    parser.add_argument("--inf-reduced-cube", default=None, type=float, help="override INF_REDUCED_VOXELGRID_SIZE")

    parser.add_argument("--only-veh", action="store_true", help="Render ONLY vehicle preds (still render union GT).")
    parser.add_argument("--only-inf", action="store_true", help="Render ONLY infra preds (still render union GT).")

    args = parser.parse_args()

    if args.only_veh and args.only_inf:
        raise ValueError("Choose at most one of --only-veh or --only-inf.")

    global VEH_REDUCED_VOXELGRID_SIZE, INF_REDUCED_VOXELGRID_SIZE
    if args.veh_reduced_cube is not None and args.veh_reduced_cube > 0:
        VEH_REDUCED_VOXELGRID_SIZE = float(args.veh_reduced_cube)
    if args.inf_reduced_cube is not None and args.inf_reduced_cube > 0:
        INF_REDUCED_VOXELGRID_SIZE = float(args.inf_reduced_cube)

    register_all_modules(init_default_scope=True)

    cfg_v = Config.fromfile(args.cfg_vehicle)
    cfg_i = Config.fromfile(args.cfg_infra)

    dataset_v = DATASETS.build(cfg_v.test_dataloader.dataset)
    dataset_i = DATASETS.build(cfg_i.test_dataloader.dataset)
    ensure_full_init(dataset_v)
    ensure_full_init(dataset_i)

    id2idx_v = build_pairid_to_idx(dataset_v)
    id2idx_i = build_pairid_to_idx(dataset_i)
    common_ids = build_common_ids_in_vehicle_order(id2idx_v, id2idx_i)
    if len(common_ids) == 0:
        raise RuntimeError("No common pair-ids between vehicle and infra splits (pairing is by lidar stem).")

    sid0 = norm_sample_id(args.sample_id)
    if sid0 is None or sid0 not in common_ids:
        raise ValueError(f"--sample-id {args.sample_id} not in intersection. Example common ids: {common_ids[:10]}")
    start_pos = common_ids.index(sid0)

    model_v = init_model(cfg_v, args.ckpt_vehicle, device=args.device)
    model_i = init_model(cfg_i, args.ckpt_infra, device=args.device)
    model_v.eval()
    model_i.eval()

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    non_kitti_root = Path(args.non_kitti_root)
    coop_root = get_coop_root(non_kitti_root)
    if not coop_root.exists():
        raise FileNotFoundError(f"coop_root does not exist: {coop_root}")

    show_veh_full = bool(args.show_veh_full)
    show_veh_reduced = not bool(args.hide_veh_reduced)
    show_inf_full = bool(args.show_inf_full)
    show_inf_reduced = not bool(args.hide_inf_reduced)

    T_cache: Dict[str, Tuple[np.ndarray, Dict[str, str]]] = {}

    print(f"[INFO] vehicle split len={len(dataset_v)}, infra split len={len(dataset_i)}")
    print(f"[INFO] common pair-ids = {len(common_ids)}")
    print(f"[INFO] start pair-id={sid0} (pos={start_pos}) score_thr={args.score_thr}")
    print(f"[INFO] non_kitti_root={non_kitti_root}")
    print(f"[INFO] coop_root={coop_root}")
    print(f"[INFO] show clouds: veh_full={show_veh_full}, veh_reduced={show_veh_reduced}, inf_full={show_inf_full}, inf_reduced={show_inf_reduced}")
    if args.only_veh:
        print("[INFO] render mode: ONLY vehicle predictions (union GT still shown)")
    if args.only_inf:
        print("[INFO] render mode: ONLY infra predictions (union GT still shown)")
    print_legend()

    if args.show:
        print_help()

        state = {"pos": start_pos, "last_sid": sid0, "last_dbg": None}

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="LiDAR Late Fusion (Vehicle frame)", width=1600, height=900, visible=True)
        _apply_render_options(vis, args.point_size)

        def redraw(pos: int, save_if_needed: bool = False):
            pos = int(pos) % len(common_ids)
            state["pos"] = pos
            sid = common_ids[pos]
            state["last_sid"] = sid

            geoms, dbg = build_scene_for_sid(
                sid=sid,
                dataset_v=dataset_v,
                dataset_i=dataset_i,
                id2idx_v=id2idx_v,
                id2idx_i=id2idx_i,
                model_v=model_v,
                model_i=model_i,
                score_thr=float(args.score_thr),
                coop_root=coop_root,
                veh_novatel_key=args.veh_novatel_key,
                veh_lidar_to_novatel_key=args.veh_lidar2novatel_key,
                inf_lidar_to_world_key=args.inf_lidar2world_key,
                T_cache=T_cache,
                show_veh_full=show_veh_full,
                show_veh_reduced=show_veh_reduced,
                show_inf_full=show_inf_full,
                show_inf_reduced=show_inf_reduced,
                debug_print=bool(args.debug_print),
                only_veh=bool(args.only_veh),
                only_inf=bool(args.only_inf),
            )
            state["last_dbg"] = dbg

            vis.clear_geometries()
            for g in geoms:
                vis.add_geometry(g)
            vis.poll_events()
            vis.update_renderer()

            print(
                f"[INFO] pair-id={sid} | "
                f"veh_gt={dbg['veh_gt']} inf_gt={dbg['inf_gt']} | "
                f"veh_pred={dbg['veh_pred']} inf_pred={dbg['inf_pred']} fused={dbg['fused_pred']}"
            )

            if out_dir and save_if_needed:
                png = out_dir / f"{sid}_late_fusion.png"
                vis.capture_screen_image(str(png), do_render=True)
                print(f"[INFO] Wrote screenshot: {png}")

        def cb_next(v):
            redraw(state["pos"] + 1, save_if_needed=False)
            return False

        def cb_prev(v):
            redraw(state["pos"] - 1, save_if_needed=False)
            return False

        def cb_save(v):
            if not out_dir:
                print("[WARN] No --out-dir set; cannot save screenshot.")
                return False
            sid = state["last_sid"] or common_ids[state["pos"]]
            png = out_dir / f"{sid}_late_fusion.png"
            v.capture_screen_image(str(png), do_render=True)
            print(f"[INFO] Wrote screenshot: {png}")
            return False

        def cb_help(v):
            print_help()
            print_legend()
            dbg = state.get("last_dbg", None)
            if isinstance(dbg, dict):
                print(f"[INFO] pair-id={dbg['sid']} idx_v={dbg['idx_v']} idx_i={dbg['idx_i']}")
                print("[INFO] veh_fov:", dbg["veh_fov"])
                print("[INFO] inf_fov:", dbg["inf_fov"])
                print("[INFO] transform src:", dbg["transform_src"])
                print("[INFO] veh reduced path:", dbg.get("veh_reduced_path"))
                print("[INFO] veh full path   :", dbg.get("veh_full_path"))
                print("[INFO] inf reduced path:", dbg.get("inf_reduced_path"))
                print("[INFO] inf full path   :", dbg.get("inf_full_path"))
                print("[INFO] only_veh:", dbg.get("only_veh"), "only_inf:", dbg.get("only_inf"))
            return False

        vis.register_key_callback(ord("N"), cb_next)
        vis.register_key_callback(ord("B"), cb_prev)
        vis.register_key_callback(ord("S"), cb_save)
        vis.register_key_callback(ord("H"), cb_help)

        redraw(start_pos, save_if_needed=bool(out_dir))
        vis.run()
        vis.destroy_window()
        return

    if not out_dir:
        raise ValueError("If --show is off, you must provide --out-dir to save an image.")

    geoms, dbg = build_scene_for_sid(
        sid=sid0,
        dataset_v=dataset_v,
        dataset_i=dataset_i,
        id2idx_v=id2idx_v,
        id2idx_i=id2idx_i,
        model_v=model_v,
        model_i=model_i,
        score_thr=float(args.score_thr),
        coop_root=coop_root,
        veh_novatel_key=args.veh_novatel_key,
        veh_lidar_to_novatel_key=args.veh_lidar2novatel_key,
        inf_lidar_to_world_key=args.inf_lidar2world_key,
        T_cache=T_cache,
        show_veh_full=show_veh_full,
        show_veh_reduced=show_veh_reduced,
        show_inf_full=show_inf_full,
        show_inf_reduced=show_inf_reduced,
        debug_print=bool(args.debug_print),
        only_veh=bool(args.only_veh),
        only_inf=bool(args.only_inf),
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="offscreen", width=1600, height=900, visible=False)
    _apply_render_options(vis, args.point_size)

    for g in geoms:
        vis.add_geometry(g)
    vis.poll_events()
    vis.update_renderer()
    png = out_dir / f"{sid0}_late_fusion.png"
    vis.capture_screen_image(str(png), do_render=True)
    vis.destroy_window()
    print(f"[INFO] Wrote screenshot: {png}")


if __name__ == "__main__":
    main()
