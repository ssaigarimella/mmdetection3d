#!/usr/bin/env python3
"""
tools/vis_single_sample_pred_gt.py

Visualize ONE sample: LiDAR points + GT boxes + predicted boxes.

Browse the entire split in the SAME Open3D window:
  - Press N for next sample, B for previous
  - Press S to save a screenshot for the current sample to --out-dir

Points shown can be:
    * "pipeline": exactly what the model sees (from dataset pipeline / pseudo_collate)
    * "full": load the corresponding full velodyne file (velodyne/*.bin) if present
    * "both": overlay both clouds

GT boxes are taken from dataset.get_data_info(idx)["eval_ann_info"]["gt_bboxes_3d"]
(what test.py uses).

NEW:
- GT_VIS_MODE controls whether to show:
    * "all"          : all GT boxes
    * "pipeline_fov" : keep only GT boxes that lie within the angular FOV (azimuth, elevation, range)
                       implied by the reduced (pipeline) point cloud.

Important: The FOV filter uses ONLY pipeline points + GT box corners.
No calibs, no transforms, no "points inside box" logic.

Usage:
  CFG=configs/_custom/pp_vehicle_synth_3class.py
  CKPT=work_dirs/pp_vehicle_synth_3class/epoch_1.pth

  python3 tools/vis_single_sample_pred_gt.py "$CFG" "$CKPT" \
    --sample-id 000002 \
    --score-thr 0.1 \
    --show \
    --out-dir work_dirs/pp_vehicle_synth_3class/vis_browse
"""

import argparse
import os
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
# IN-CODE PARAMS (edit these instead of CLI if you want)
# ============================================================

# "pipeline" = what the model sees (likely velodyne_reduced in your setup)
# "full"     = load velodyne/*.bin if present (no reduced/frustum clipping)
# "both"     = overlay both clouds
POINTS_VIS_MODE = "both"   # "pipeline" | "full" | "both"

# Downsample settings for each cloud (separately)
PIPELINE_MAX_POINTS = 200000
PIPELINE_VOXEL = 0.05

FULL_MAX_POINTS = 400000
FULL_VOXEL = 0.05

# Colors
PIPELINE_POINTS_COLOR = (0.70, 0.70, 0.70)  # gray
FULL_POINTS_COLOR = (0.25, 0.55, 1.00)      # blue-ish

# If True, filter GT boxes by cfg.model.voxel_layer.point_cloud_range (center-based)
GT_RANGE_FILTER = False

# Which GT boxes to show:
#   "all"          : show all GT boxes
#   "pipeline_fov" : keep only GT boxes inside FOV implied by reduced (pipeline) points
GT_VIS_MODE = "pipeline_fov"   # "all" | "pipeline_fov"

# ---------- FOV filter tuning (pipeline_fov) ----------
# We define the reduced lidar FOV by the angular + range envelope of the pipeline points.
# Use robust percentiles to avoid rare outliers expanding the FOV.
FOV_SUBSAMPLE_MAX_POINTS = 120000   # subsample pipeline points when estimating FOV (speed)
FOV_AZ_COVERAGE = 0.999             # fraction of points whose azimuth should be covered by the inferred interval
FOV_EL_LOW_PCT = 0.001              # elevation lower percentile
FOV_EL_HIGH_PCT = 0.999             # elevation upper percentile
FOV_R_LOW_PCT = 0.001               # range lower percentile
FOV_R_HIGH_PCT = 0.999              # range upper percentile

# Margins added to the inferred FOV (to avoid over-pruning)
FOV_AZ_MARGIN_DEG = 1.0
FOV_EL_MARGIN_DEG = 1.0
FOV_R_MARGIN_M = 0.5

# If True, also require some part of the box to be in front of the sensor (x>0 in lidar frame).
# Set False if your lidar frame convention is different or you want full 360 behavior.
FOV_REQUIRE_X_POSITIVE = False

# If True, automatically save screenshot on every N/B navigation (can be slow)
AUTO_SAVE_ON_NAV = False

# Render the PIPELINE cloud as voxels so it stands out over FULL points
PIPELINE_RENDER_AS_VOXELGRID = True
PIPELINE_RENDER_VOXEL_SIZE = 0.20   # meters; increase to 0.30 if still too thin

# ============================================================


# -------------------------
# ID utilities
# -------------------------

def norm_sample_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return f"{int(x):06d}"
    s = str(x).strip()
    if ("/" in s) or ("\\" in s) or s.endswith((".bin", ".png", ".pcd", ".npy")):
        try:
            s = Path(s).stem
        except Exception:
            pass
    if s.isdigit():
        return s.zfill(6) if len(s) <= 6 else s
    return s


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


def find_sample_index(dataset, sample_id: str) -> int:
    ensure_full_init(dataset)
    target = norm_sample_id(sample_id)
    n = len(dataset)
    first_ids: List[str] = []

    def extract_sid(info: Dict) -> Optional[str]:
        if not isinstance(info, dict):
            return None
        sid = info.get("sample_idx") or info.get("sample_id")
        if sid is not None:
            return norm_sample_id(sid)

        lp = info.get("lidar_points", None)
        if isinstance(lp, dict):
            p = lp.get("lidar_path") or lp.get("pts_path")
            if p is not None:
                return norm_sample_id(p)

        if "lidar_path" in info:
            return norm_sample_id(info.get("lidar_path"))

        return None

    dl = _get_data_list(dataset)
    if dl is not None and len(dl) == n:
        for i, info in enumerate(dl):
            sid = extract_sid(info)
            if sid is not None and len(first_ids) < 10:
                first_ids.append(sid)
            if sid == target:
                return i
    else:
        for i in range(n):
            info = _get_data_info(dataset, i) or {}
            sid = extract_sid(info)
            if sid is not None and len(first_ids) < 10:
                first_ids.append(sid)
            if sid == target:
                return i

    raise ValueError(
        f"Could not locate sample-id={target} in current test dataset split (len={n}).\n"
        f"Examples of sample IDs in this split: {first_ids}"
    )


# -------------------------
# Points extraction / loading
# -------------------------

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


def voxel_downsample_points(pts_xyz: np.ndarray, voxel: float) -> np.ndarray:
    if voxel is None or voxel <= 0:
        return pts_xyz
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_xyz)
    pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
    return np.asarray(pcd.points)


def cap_points(pts_xyz: np.ndarray, max_points: int) -> np.ndarray:
    if max_points is None or max_points <= 0 or pts_xyz.shape[0] <= max_points:
        return pts_xyz
    sel = np.random.choice(pts_xyz.shape[0], size=max_points, replace=False)
    return pts_xyz[sel]


def _safe_load_kitti_bin(path: Path) -> Optional[np.ndarray]:
    try:
        raw = np.fromfile(str(path), dtype=np.float32)
        if raw.size < 4:
            return None
        raw = raw.reshape(-1, 4)[:, :3]
        return raw.astype(np.float64)
    except Exception:
        return None


def resolve_lidar_path_from_info(info: Dict) -> Optional[Path]:
    if not isinstance(info, dict):
        return None
    lp = info.get("lidar_points", {})
    lidar_path = None
    if isinstance(lp, dict):
        lidar_path = lp.get("lidar_path") or lp.get("pts_path")
    if lidar_path is None:
        lidar_path = info.get("lidar_path")
    if lidar_path is None:
        return None
    return Path(str(lidar_path))


def try_load_full_velodyne(dataset, idx: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Try to load a full velodyne point cloud for this sample by mapping:
      .../velodyne_reduced/XXXXXX.bin -> .../velodyne/XXXXXX.bin
    """
    info = _get_data_info(dataset, idx) or {}
    p = resolve_lidar_path_from_info(info)
    if p is None:
        return None, None

    p_str = p.as_posix()
    if "velodyne_reduced" in p_str:
        full_p = Path(p_str.replace("velodyne_reduced", "velodyne"))
    else:
        full_p = p

    if not full_p.exists():
        return None, None

    pts = _safe_load_kitti_bin(full_p)
    if pts is None:
        return None, None

    return pts, os.path.realpath(str(full_p))


# -------------------------
# Pred boxes -> corners
# -------------------------

def pred_corners_from_output(pred_sample) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    pred_instances = getattr(pred_sample, "pred_instances_3d", None)
    if pred_instances is None or not hasattr(pred_instances, "bboxes_3d"):
        return None, None

    boxes = pred_instances.bboxes_3d
    if boxes is None or not hasattr(boxes, "corners"):
        return None, None

    corners = boxes.corners.detach().cpu().numpy().astype(np.float64)

    scores = getattr(pred_instances, "scores_3d", None)
    if scores is not None and not isinstance(scores, np.ndarray):
        scores = scores.detach().cpu().numpy()

    return corners, scores


# -------------------------
# GT from eval_ann_info (test.py aligned)
# -------------------------

def gt_corners_from_eval_ann_info(dataset, idx: int) -> Optional[np.ndarray]:
    ensure_full_init(dataset)
    info = _get_data_info(dataset, idx)
    if not isinstance(info, dict):
        return None
    ea = info.get("eval_ann_info", None)
    if not isinstance(ea, dict):
        return None
    gt_boxes = ea.get("gt_bboxes_3d", None)
    if gt_boxes is None or not hasattr(gt_boxes, "corners"):
        return None
    return gt_boxes.corners.detach().cpu().numpy().astype(np.float64)


def gt_centers_from_eval_ann_info(dataset, idx: int) -> Optional[np.ndarray]:
    info = _get_data_info(dataset, idx) or {}
    ea = info.get("eval_ann_info", {}) if isinstance(info, dict) else {}
    gt_boxes = ea.get("gt_bboxes_3d", None)
    if gt_boxes is None or not hasattr(gt_boxes, "tensor"):
        return None
    t = gt_boxes.tensor.detach().cpu().numpy()
    if t.ndim != 2 or t.shape[1] < 3:
        return None
    return t[:, :3].astype(np.float64)


def get_voxel_point_cloud_range(cfg: Config) -> Optional[np.ndarray]:
    try:
        pcr = cfg.model.voxel_layer.point_cloud_range
        return np.array(pcr, dtype=np.float64)
    except Exception:
        return None


def filter_boxes_by_center_range(corners: np.ndarray, centers_xyz: np.ndarray, pcr: np.ndarray) -> np.ndarray:
    x0, y0, z0, x1, y1, z1 = pcr.tolist()
    c = centers_xyz
    keep = (
        (c[:, 0] >= x0) & (c[:, 0] <= x1) &
        (c[:, 1] >= y0) & (c[:, 1] <= y1) &
        (c[:, 2] >= z0) & (c[:, 2] <= z1)
    )
    return corners[keep]


# -------------------------
# FOV filter from pipeline points (purely angular envelope)
# -------------------------

def _subsample_rows(x: np.ndarray, max_rows: int) -> np.ndarray:
    if max_rows is None or max_rows <= 0 or x.shape[0] <= max_rows:
        return x
    sel = np.random.choice(x.shape[0], size=max_rows, replace=False)
    return x[sel]


def _minimal_circular_interval_covering(angles_rad: np.ndarray, coverage: float) -> Tuple[float, float]:
    """
    Find the smallest circular interval [lo, hi] on [0, 2pi) that contains
    at least coverage fraction of angles.

    Returns (lo, hi) in radians, in [0, 2pi). Interval is the shorter arc from lo to hi
    going forward (may have hi < lo meaning wrap, but we keep lo<=hi by construction here).
    """
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

    lo = np.mod(best_lo, 2.0 * np.pi)
    hi = np.mod(best_hi, 2.0 * np.pi)

    # If the best interval wrapped, represent it as a forward interval with lo>hi in wrap form.
    # For membership tests we will use a helper that supports wrap.
    return float(lo), float(hi)


def _angle_in_interval(ang: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    ang, lo, hi are in [0,2pi).
    If interval does not wrap: lo <= hi, then lo <= ang <= hi.
    If interval wraps: lo > hi, then ang >= lo OR ang <= hi.
    """
    ang = np.mod(ang, 2.0 * np.pi)
    lo = float(np.mod(lo, 2.0 * np.pi))
    hi = float(np.mod(hi, 2.0 * np.pi))
    if lo <= hi:
        return (ang >= lo) & (ang <= hi)
    else:
        return (ang >= lo) | (ang <= hi)


def compute_pipeline_fov(
    pts_xyz: np.ndarray,
    subsample_max: int,
    az_coverage: float,
    el_low_pct: float,
    el_high_pct: float,
    r_low_pct: float,
    r_high_pct: float,
) -> Dict[str, float]:
    """
    Infer an (azimuth interval, elevation interval, range interval) from pipeline points.
    All angles are in radians.
    """
    if pts_xyz is None or pts_xyz.shape[0] == 0:
        return {
            "az_lo": 0.0, "az_hi": 0.0,
            "el_lo": 0.0, "el_hi": 0.0,
            "r_lo": 0.0, "r_hi": 0.0,
        }

    p = _subsample_rows(pts_xyz, subsample_max)

    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]

    r = np.sqrt(x * x + y * y + z * z) + 1e-9
    az = np.arctan2(y, x)                 # [-pi, pi]
    el = np.arcsin(np.clip(z / r, -1.0, 1.0))  # [-pi/2, pi/2]

    az_lo, az_hi = _minimal_circular_interval_covering(az, coverage=az_coverage)
    el_lo = float(np.quantile(el, el_low_pct))
    el_hi = float(np.quantile(el, el_high_pct))
    r_lo = float(np.quantile(r, r_low_pct))
    r_hi = float(np.quantile(r, r_high_pct))

    return {"az_lo": az_lo, "az_hi": az_hi, "el_lo": el_lo, "el_hi": el_hi, "r_lo": r_lo, "r_hi": r_hi}


def filter_gt_by_pipeline_fov(
    gt_corners: np.ndarray,
    pts_pipeline_xyz: np.ndarray,
    az_margin_deg: float,
    el_margin_deg: float,
    r_margin_m: float,
    require_x_positive: bool,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    """
    Keep GT boxes that intersect the pipeline FOV envelope in (azimuth, elevation, range).

    Rule:
      Keep a GT box if ANY of its corners falls within:
        az in [az_lo, az_hi] (circular interval, with margin)
        el in [el_lo, el_hi] (with margin)
        r  in [r_lo,  r_hi ] (with margin)
      and optionally x>0 for that corner if require_x_positive is True.

    Returns:
      (filtered_gt_corners, fov_dict_with_margins, keep_mask)
    """
    if gt_corners is None or gt_corners.shape[0] == 0:
        return gt_corners, {}, np.zeros((0,), dtype=bool)
    if pts_pipeline_xyz is None or pts_pipeline_xyz.shape[0] == 0:
        return gt_corners[:0], {}, np.zeros((gt_corners.shape[0],), dtype=bool)

    fov = compute_pipeline_fov(
        pts_xyz=pts_pipeline_xyz,
        subsample_max=FOV_SUBSAMPLE_MAX_POINTS,
        az_coverage=FOV_AZ_COVERAGE,
        el_low_pct=FOV_EL_LOW_PCT,
        el_high_pct=FOV_EL_HIGH_PCT,
        r_low_pct=FOV_R_LOW_PCT,
        r_high_pct=FOV_R_HIGH_PCT,
    )

    az_margin = np.deg2rad(float(az_margin_deg))
    el_margin = np.deg2rad(float(el_margin_deg))

    az_lo = float(np.mod(fov["az_lo"] - az_margin, 2.0 * np.pi))
    az_hi = float(np.mod(fov["az_hi"] + az_margin, 2.0 * np.pi))
    el_lo = float(fov["el_lo"] - el_margin)
    el_hi = float(fov["el_hi"] + el_margin)
    r_lo = float(max(0.0, fov["r_lo"] - float(r_margin_m)))
    r_hi = float(fov["r_hi"] + float(r_margin_m))

    c = np.asarray(gt_corners, dtype=np.float64)  # (N,8,3)
    x = c[:, :, 0]
    y = c[:, :, 1]
    z = c[:, :, 2]
    r = np.sqrt(x * x + y * y + z * z) + 1e-9
    az = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    el = np.arcsin(np.clip(z / r, -1.0, 1.0))

    in_az = _angle_in_interval(az, az_lo, az_hi)         # (N,8)
    in_el = (el >= el_lo) & (el <= el_hi)                # (N,8)
    in_r = (r >= r_lo) & (r <= r_hi)                     # (N,8)

    in_all = in_az & in_el & in_r

    if require_x_positive:
        in_all = in_all & (x > 0.0)

    keep = np.any(in_all, axis=1)

    fov_dbg = {
        "az_lo": az_lo, "az_hi": az_hi,
        "el_lo": el_lo, "el_hi": el_hi,
        "r_lo": r_lo, "r_hi": r_hi,
        "az_coverage": float(FOV_AZ_COVERAGE),
    }

    return gt_corners[keep], fov_dbg, keep


# -------------------------
# Robust LineSet creation (no corners ordering assumptions)
# -------------------------

def order_face_cycle_xy(pts4: np.ndarray) -> np.ndarray:
    c = pts4[:, :2].mean(axis=0)
    ang = np.arctan2(pts4[:, 1] - c[1], pts4[:, 0] - c[0])
    return pts4[np.argsort(ang)]


def match_top_to_bottom(bottom4: np.ndarray, top4: np.ndarray) -> np.ndarray:
    top_rem = top4.copy()
    ordered_top = []
    used = np.zeros((top_rem.shape[0],), dtype=bool)
    for b in bottom4:
        d = np.linalg.norm(top_rem[:, :2] - b[:2], axis=1)
        d[used] = 1e9
        j = int(np.argmin(d))
        used[j] = True
        ordered_top.append(top_rem[j])
    return np.stack(ordered_top, axis=0)


def corners_to_lineset(corners_8x3: np.ndarray, color_rgb) -> o3d.geometry.LineSet:
    c = np.asarray(corners_8x3, dtype=np.float64).reshape(8, 3)

    z = c[:, 2]
    bot_idx = np.argsort(z)[:4]
    top_idx = np.argsort(z)[4:]

    bottom = order_face_cycle_xy(c[bot_idx])
    top = order_face_cycle_xy(c[top_idx])
    top = match_top_to_bottom(bottom, top)

    pts = np.vstack([bottom, top])  # 0-3 bottom, 4-7 top

    lines = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        dtype=np.int32,
    )

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(color_rgb, dtype=np.float64), (lines.shape[0], 1))
    )
    return ls


# -------------------------
# Scene builder for one dataset index
# -------------------------

def build_scene_for_idx(
    dataset,
    cfg: Config,
    model,
    idx: int,
    score_thr: float,
    points_mode: str,
    debug_print: bool,
) -> Tuple[str, List[o3d.geometry.Geometry], Dict[str, Any]]:
    """
    Returns: (sample_id_str, geometries, debug_dict)
    """
    info = _get_data_info(dataset, idx) or {}
    sid = norm_sample_id(info.get("sample_idx", idx)) or f"{idx:06d}"

    # pipeline sample + inference
    data = dataset[idx]
    data_batch = pseudo_collate([data])

    with torch.no_grad():
        outputs = model.test_step(data_batch)
    pred_sample = outputs[0]

    # pipeline points (what model saw) - keep raw for FOV filtering
    pts_pipeline_raw = extract_points_from_batch(data_batch)

    # full points (if exist)
    pts_full, full_path_used = try_load_full_velodyne(dataset, idx)

    # choose clouds for visualization (downsampled/capped)
    pts_pipeline_vis = None
    pts_full_vis = None

    if points_mode in ("pipeline", "both"):
        pts_pipeline_vis = cap_points(pts_pipeline_raw.copy(), PIPELINE_MAX_POINTS)
        pts_pipeline_vis = voxel_downsample_points(pts_pipeline_vis, PIPELINE_VOXEL)

    if points_mode in ("full", "both"):
        if pts_full is None:
            pts_full_vis = None
        else:
            pts_full_vis = cap_points(pts_full.copy(), FULL_MAX_POINTS)
            pts_full_vis = voxel_downsample_points(pts_full_vis, FULL_VOXEL)

    # preds
    pred_corners_all, scores = pred_corners_from_output(pred_sample)
    pred_corners = None
    kept_pred = 0
    total_pred = 0
    if pred_corners_all is not None:
        total_pred = int(pred_corners_all.shape[0])
        if scores is not None:
            keep = scores >= float(score_thr)
            pred_corners = pred_corners_all[keep]
            kept_pred = int(keep.sum())
        else:
            pred_corners = pred_corners_all
            kept_pred = int(pred_corners.shape[0])

    # GT from eval_ann_info (test.py aligned)
    gt_corners = gt_corners_from_eval_ann_info(dataset, idx)
    gt_centers = gt_centers_from_eval_ann_info(dataset, idx)

    gt_count_before = int(gt_corners.shape[0]) if gt_corners is not None else 0

    # optional GT range filter (existing)
    if GT_RANGE_FILTER and gt_corners is not None and gt_centers is not None:
        pcr = get_voxel_point_cloud_range(cfg)
        if pcr is not None:
            gt_corners = filter_boxes_by_center_range(gt_corners, gt_centers, pcr)

    # NEW: pure FOV-based GT visualization filter
    fov_dbg = None
    keep_mask = None
    if GT_VIS_MODE == "pipeline_fov" and gt_corners is not None:
        gt_corners, fov_dbg, keep_mask = filter_gt_by_pipeline_fov(
            gt_corners=gt_corners,
            pts_pipeline_xyz=pts_pipeline_raw,
            az_margin_deg=FOV_AZ_MARGIN_DEG,
            el_margin_deg=FOV_EL_MARGIN_DEG,
            r_margin_m=FOV_R_MARGIN_M,
            require_x_positive=FOV_REQUIRE_X_POSITIVE,
        )

    gt_count_after = int(gt_corners.shape[0]) if gt_corners is not None else 0

    # debug: resolved lidar path
    lp = info.get("lidar_points", {})
    lidar_path = None
    if isinstance(lp, dict):
        lidar_path = lp.get("lidar_path") or lp.get("pts_path")
    if lidar_path is None:
        lidar_path = info.get("lidar_path")
    lidar_resolved = os.path.realpath(str(lidar_path)) if lidar_path is not None else None

    dbg = {
        "idx": idx,
        "sid": sid,
        "lidar_path": lidar_path,
        "lidar_resolved": lidar_resolved,
        "full_velodyne_resolved": full_path_used,
        "pred_kept": kept_pred,
        "pred_total": total_pred,
        "gt_count_before": gt_count_before,
        "gt_count": gt_count_after,
        "gt_vis_mode": GT_VIS_MODE,
        "fov_dbg": fov_dbg,
        "keep_mask": keep_mask,
        "pts_pipeline_n": int(pts_pipeline_vis.shape[0]) if pts_pipeline_vis is not None else 0,
        "pts_full_n": int(pts_full_vis.shape[0]) if pts_full_vis is not None else 0,
    }

    if debug_print:
        print(f"[DEBUG] idx={idx} sid={sid}")
        if lidar_path is not None:
            print("[DEBUG] lidar_path from info:", lidar_path)
            print("[DEBUG] resolved:", lidar_resolved)
        if full_path_used is not None:
            print("[DEBUG] full velodyne resolved:", full_path_used)

        if pts_pipeline_raw is not None and pts_pipeline_raw.shape[0] > 0:
            mn = pts_pipeline_raw.min(axis=0)
            mx = pts_pipeline_raw.max(axis=0)
            print("[DEBUG] pipeline RAW xyz min:", mn.tolist(), "max:", mx.tolist())

        if GT_VIS_MODE == "pipeline_fov" and isinstance(fov_dbg, dict):
            az_lo = fov_dbg["az_lo"]
            az_hi = fov_dbg["az_hi"]
            el_lo = fov_dbg["el_lo"]
            el_hi = fov_dbg["el_hi"]
            r_lo = fov_dbg["r_lo"]
            r_hi = fov_dbg["r_hi"]
            print("[DEBUG] inferred FOV (with margins):")
            print("        az_lo/az_hi deg:", float(np.rad2deg(az_lo)), float(np.rad2deg(az_hi)))
            print("        el_lo/el_hi deg:", float(np.rad2deg(el_lo)), float(np.rad2deg(el_hi)))
            print("        r_lo/r_hi m:", float(r_lo), float(r_hi))
            print(f"[DEBUG] kept {gt_count_after}/{gt_count_before} GT by pipeline_fov")

    geometries: List[o3d.geometry.Geometry] = []
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0]))

    if pts_pipeline_vis is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_pipeline_vis.astype(np.float64))
        pcd.paint_uniform_color(PIPELINE_POINTS_COLOR)

        if PIPELINE_RENDER_AS_VOXELGRID:
            vg = o3d.geometry.VoxelGrid.create_from_point_cloud(
                pcd, voxel_size=float(PIPELINE_RENDER_VOXEL_SIZE)
            )
            geometries.append(vg)
        else:
            geometries.append(pcd)

    if pts_full_vis is not None:
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts_full_vis.astype(np.float64))
        pcd2.paint_uniform_color(FULL_POINTS_COLOR)
        geometries.append(pcd2)

    if gt_corners is not None:
        for k in range(gt_corners.shape[0]):
            geometries.append(corners_to_lineset(gt_corners[k], (0.2, 1.0, 0.2)))

    if pred_corners is not None:
        for k in range(pred_corners.shape[0]):
            geometries.append(corners_to_lineset(pred_corners[k], (1.0, 0.2, 0.2)))

    return sid, geometries, dbg


def print_help():
    print("Keys:")
    print("  N : next sample (wrap)")
    print("  B : previous sample (wrap)")
    print("  S : save screenshot to --out-dir")
    print("  H : print this help + current status")
    print("  ESC / close window: exit")


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str)
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--sample-id", required=True, type=str, help="starting sample id (6-digit)")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--score-thr", default=0.1, type=float)
    parser.add_argument("--point-size", default=2.0, type=float)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--out-dir", default=None, type=str)
    parser.add_argument("--points-mode", default=None, type=str, choices=["pipeline", "full", "both"])
    parser.add_argument("--debug-print", action="store_true")
    args = parser.parse_args()

    points_mode = args.points_mode if args.points_mode is not None else POINTS_VIS_MODE

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.cfg)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    ensure_full_init(dataset)

    start_idx = find_sample_index(dataset, args.sample_id)
    n = len(dataset)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    model = init_model(cfg, args.ckpt, device=args.device)
    model.eval()

    print(f"[INFO] Dataset split len={n}. Starting at dataset_idx={start_idx}. points_mode={points_mode} GT_VIS_MODE={GT_VIS_MODE}")
    if args.show:
        print_help()

        state = {
            "idx": start_idx,
            "n": n,
            "last_sid": None,
            "last_dbg": None,
        }

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Pred vs GT browser", width=1600, height=900, visible=True)
        opt = vis.get_render_option()
        if opt is not None:
            opt.point_size = float(args.point_size)

        def redraw(current_idx: int, save_if_needed: bool = False):
            current_idx = int(current_idx) % n
            state["idx"] = current_idx

            sid, geoms, dbg = build_scene_for_idx(
                dataset=dataset,
                cfg=cfg,
                model=model,
                idx=current_idx,
                score_thr=float(args.score_thr),
                points_mode=points_mode,
                debug_print=bool(args.debug_print),
            )

            state["last_sid"] = sid
            state["last_dbg"] = dbg

            vis.clear_geometries()
            for g in geoms:
                vis.add_geometry(g)

            vis.poll_events()
            vis.update_renderer()

            msg = (
                f"[INFO] idx={current_idx}/{n-1} sid={sid} | "
                f"GT={dbg['gt_count']}"
            )
            if dbg.get("gt_count_before", dbg["gt_count"]) != dbg["gt_count"]:
                msg += f" (kept {dbg['gt_count']}/{dbg['gt_count_before']} by {dbg['gt_vis_mode']})"
            msg += (
                f" | Pred kept={dbg['pred_kept']}/{dbg['pred_total']} | "
                f"pts(pipeline)={dbg['pts_pipeline_n']} pts(full)={dbg['pts_full_n']}"
            )
            print(msg)

            if out_dir and save_if_needed:
                png = out_dir / f"{sid}_pred_gt.png"
                vis.capture_screen_image(str(png), do_render=True)
                print(f"[INFO] Wrote screenshot: {png}")

        def cb_next(v):
            redraw(state["idx"] + 1, save_if_needed=AUTO_SAVE_ON_NAV)
            return False

        def cb_prev(v):
            redraw(state["idx"] - 1, save_if_needed=AUTO_SAVE_ON_NAV)
            return False

        def cb_save(v):
            if not out_dir:
                print("[WARN] No --out-dir set; cannot save screenshot.")
                return False
            sid = state["last_sid"] or f"{state['idx']:06d}"
            png = out_dir / f"{sid}_pred_gt.png"
            v.capture_screen_image(str(png), do_render=True)
            print(f"[INFO] Wrote screenshot: {png}")
            return False

        def cb_help(v):
            print_help()
            dbg = state.get("last_dbg", None)
            if isinstance(dbg, dict):
                print(f"[INFO] current idx={dbg['idx']} sid={dbg['sid']}")
                print(f"[INFO] GT_VIS_MODE={dbg.get('gt_vis_mode', GT_VIS_MODE)}")
                if dbg.get("lidar_resolved") is not None:
                    print(f"[INFO] lidar_resolved: {dbg['lidar_resolved']}")
                if dbg.get("full_velodyne_resolved") is not None:
                    print(f"[INFO] full_velodyne_resolved: {dbg['full_velodyne_resolved']}")
                if isinstance(dbg.get("fov_dbg", None), dict):
                    fd = dbg["fov_dbg"]
                    print("[INFO] FOV az(deg):", float(np.rad2deg(fd["az_lo"])), float(np.rad2deg(fd["az_hi"])))
                    print("[INFO] FOV el(deg):", float(np.rad2deg(fd["el_lo"])), float(np.rad2deg(fd["el_hi"])))
                    print("[INFO] FOV r(m):", float(fd["r_lo"]), float(fd["r_hi"]))
            return False

        vis.register_key_callback(ord("N"), cb_next)
        vis.register_key_callback(ord("B"), cb_prev)
        vis.register_key_callback(ord("S"), cb_save)
        vis.register_key_callback(ord("H"), cb_help)

        redraw(start_idx, save_if_needed=bool(out_dir))  # keep old behavior: save first frame if out-dir set
        vis.run()
        vis.destroy_window()
        return

    # Non-interactive mode: keep old behavior (render once offscreen-ish to file)
    if not out_dir:
        raise ValueError("If --show is off, you must provide --out-dir to save an image.")

    sid, geoms, dbg = build_scene_for_idx(
        dataset=dataset,
        cfg=cfg,
        model=model,
        idx=start_idx,
        score_thr=float(args.score_thr),
        points_mode=points_mode,
        debug_print=bool(args.debug_print),
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="offscreen", width=1600, height=900, visible=False)
    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = float(args.point_size)

    for g in geoms:
        vis.add_geometry(g)

    vis.poll_events()
    vis.update_renderer()
    png = out_dir / f"{sid}_pred_gt.png"
    vis.capture_screen_image(str(png), do_render=True)
    vis.destroy_window()
    print(f"[INFO] Wrote screenshot: {png}")


if __name__ == "__main__":
    main()
