#!/usr/bin/env python3
"""
tools/dump_pred_gt_lidar_late_fusion.py

Dump GT + LATE-FUSION predictions in VEHICLE LiDAR frame for an entire paired split.

Aligned with tools/dump_pred_gt_lidar.py:
- Same FOV inference method: circular-mean unwrap + percentiles on (daz, el)
- Same margins and thresholds
- Same box test policy (CENTER)
- All filtering uses VEHICLE pipeline FOV (since eval is in VEHICLE frame)

Fusion behavior:
- Run vehicle detector on vehicle sample
- Run infra detector on infra sample
- Transform infra preds into vehicle LiDAR frame using JSON transforms under --non-kitti-root
- Optional FOV filtering (center test) for GT and preds using VEHICLE pipeline FOV
- Per-class matching by rotated BEV IoU gate (greedy, one-to-one, high IoU first)
- If matched: keep VEHICLE box geometry, fused_score = max(v_score, i_score)
- Keep unmatched vehicle + unmatched infra
- Post-fusion: class-wise rotated BEV NMS using VEHICLE cfg test_cfg (nms_thr, max_num)

Output pickle:
  {
    "classes": [...],
    "fov_filter": {...},
    "fusion": {...},
    "samples": [
      {
        "sample_id": "000031",
        "fov": {...} or None,
        "pred_boxes":  (N,7) float32 [x,y,z,dx,dy,dz,yaw]
        "pred_scores": (N,)  float32
        "pred_labels": (N,)  int64
        "gt_boxes":    (M,7) float32
        "gt_labels":   (M,)  int64
      },
      ...
    ]
  }
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from mmengine.config import Config
from mmengine.registry import DATASETS
from mmengine.dataset import pseudo_collate

from mmdet3d.apis import init_model
from mmdet3d.utils import register_all_modules


# ============================================================
# Debug steps
# ============================================================
# step 0: works correctly as expected when true
DEBUG_VEHICLE_ONLY = False

# step 1: has no effect; so just leave as false
APPLY_POST_FUSION_NMS = False


# ============================================================
# MATCH tools/dump_pred_gt_lidar.py
# ============================================================

FILTER_GT_BY_PIPELINE_FOV = True
FILTER_PRED_BY_PIPELINE_FOV = True
FILTER_FUSED_BY_VEH_FOV = True

FOV_BOX_TEST = "center"

USE_ANGLE_PERCENTILES = True
AZIMUTH_PCTS = (0.5, 99.5)
ELEVATION_PCTS = (0.5, 99.5)

AZ_MARGIN_DEG = 0.25
EL_MARGIN_DEG = 0.25

MIN_PTS_FOR_FOV = 64

# ============================================================
# Fusion params (tune these if needed)
# ============================================================

MATCH_BEV_IOU_THR_DEFAULT = 0.10
MATCH_BEV_IOU_THR_BY_CLASSNAME = {
    "Car": 0.10,
    "Pedestrian": 0.05,
    "Cyclist": 0.05,
}

# Prune IoU computations by center distance in BEV
PREMATCH_CENTER_DIST_M = 12.0



# ============================================================
# small utils
# ============================================================

def ensure_full_init(dataset) -> None:
    if hasattr(dataset, "full_init"):
        try:
            dataset.full_init()
        except Exception:
            pass

def to_np(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

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

def get_data_info(dataset, idx: int) -> Dict:
    if hasattr(dataset, "get_data_info"):
        out = dataset.get_data_info(idx)
        return out if isinstance(out, dict) else {}
    if hasattr(dataset, "data_list") and isinstance(dataset.data_list, list):
        if 0 <= idx < len(dataset.data_list) and isinstance(dataset.data_list[idx], dict):
            return dataset.data_list[idx]
    if hasattr(dataset, "infos") and isinstance(dataset.infos, list):
        if 0 <= idx < len(dataset.infos) and isinstance(dataset.infos[idx], dict):
            return dataset.infos[idx]
    return {}

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
        return np.zeros((0, 3), dtype=np.float64)
    arr = pts if isinstance(pts, np.ndarray) else pts.detach().cpu().numpy()
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float64)
    return arr[:, :3].astype(np.float64)


# ============================================================
# GT/pred extraction
# ============================================================

def extract_pred(pred_sample, min_score: float):
    pred_instances = getattr(pred_sample, "pred_instances_3d", None)
    if pred_instances is None:
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    boxes3d = getattr(pred_instances, "bboxes_3d", None)
    scores = getattr(pred_instances, "scores_3d", None)
    labels = getattr(pred_instances, "labels_3d", None)

    if boxes3d is None or not hasattr(boxes3d, "tensor"):
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    boxes = to_np(boxes3d.tensor).astype(np.float32)
    if boxes.shape[1] > 7:
        boxes = boxes[:, :7]

    scores = to_np(scores).astype(np.float32) if scores is not None else np.ones((boxes.shape[0],), dtype=np.float32)
    labels = to_np(labels).astype(np.int64) if labels is not None else np.zeros((boxes.shape[0],), dtype=np.int64)

    if min_score > 0:
        keep = scores >= float(min_score)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

    return boxes, scores, labels

def extract_gt_from_eval_ann_info(info: Dict):
    ea = info.get("eval_ann_info", None)
    if not isinstance(ea, dict):
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    gt_boxes3d = ea.get("gt_bboxes_3d", None)
    if gt_boxes3d is None or not hasattr(gt_boxes3d, "tensor"):
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    gt_boxes = to_np(gt_boxes3d.tensor).astype(np.float32)
    if gt_boxes.shape[1] > 7:
        gt_boxes = gt_boxes[:, :7]

    gt_labels = ea.get("gt_labels_3d", None)
    if gt_labels is None:
        gt_labels = ea.get("gt_labels", None)
    if gt_labels is None:
        return gt_boxes, np.zeros((gt_boxes.shape[0],), dtype=np.int64)

    gt_labels = to_np(gt_labels).astype(np.int64).reshape(-1)
    return gt_boxes, gt_labels


# ============================================================
# FOV utilities
# ============================================================

def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def compute_fov_from_points(pts_xyz: np.ndarray) -> Optional[Dict[str, float]]:
    if pts_xyz is None or pts_xyz.shape[0] < int(MIN_PTS_FOR_FOV):
        return None

    x = pts_xyz[:, 0]
    y = pts_xyz[:, 1]
    z = pts_xyz[:, 2]

    r_xy = np.sqrt(x * x + y * y)
    valid = np.isfinite(r_xy) & np.isfinite(z) & np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return None

    x = x[valid]
    y = y[valid]
    z = z[valid]
    r_xy = r_xy[valid]

    az = np.arctan2(y, x)
    el = np.arctan2(z, np.maximum(r_xy, 1e-12))

    az0 = float(np.arctan2(np.mean(np.sin(az)), np.mean(np.cos(az))))
    daz = wrap_to_pi(az - az0)

    if USE_ANGLE_PERCENTILES:
        daz_min, daz_max = np.percentile(daz, AZIMUTH_PCTS).astype(np.float64).tolist()
        el_min, el_max = np.percentile(el, ELEVATION_PCTS).astype(np.float64).tolist()
    else:
        daz_min = float(np.min(daz))
        daz_max = float(np.max(daz))
        el_min = float(np.min(el))
        el_max = float(np.max(el))

    daz_margin = np.deg2rad(float(AZ_MARGIN_DEG))
    el_margin = np.deg2rad(float(EL_MARGIN_DEG))

    return dict(
        az0=float(az0),
        daz_min=float(daz_min - daz_margin),
        daz_max=float(daz_max + daz_margin),
        el_min=float(el_min - el_margin),
        el_max=float(el_max + el_margin),
    )

def boxes7_to_corners_xyz(boxes7: np.ndarray) -> np.ndarray:
    if boxes7 is None or boxes7.shape[0] == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)

    c = boxes7[:, 0:3].astype(np.float32)
    dx = boxes7[:, 3].astype(np.float32)
    dy = boxes7[:, 4].astype(np.float32)
    dz = boxes7[:, 5].astype(np.float32)
    yaw = boxes7[:, 6].astype(np.float32)

    hx = 0.5 * dx
    hy = 0.5 * dy
    hz = 0.5 * dz

    corners_local = np.array(
        [
            [+1, +1, +1],
            [+1, -1, +1],
            [-1, -1, +1],
            [-1, +1, +1],
            [+1, +1, -1],
            [+1, -1, -1],
            [-1, -1, -1],
            [-1, +1, -1],
        ],
        dtype=np.float32,
    )

    corners = np.repeat(corners_local[None, :, :], boxes7.shape[0], axis=0)
    corners[:, :, 0] *= hx[:, None]
    corners[:, :, 1] *= hy[:, None]
    corners[:, :, 2] *= hz[:, None]

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    x0 = corners[:, :, 0].copy()
    y0 = corners[:, :, 1].copy()
    corners[:, :, 0] = cy[:, None] * x0 - sy[:, None] * y0
    corners[:, :, 1] = sy[:, None] * x0 + cy[:, None] * y0

    corners[:, :, 0] += c[:, None, 0]
    corners[:, :, 1] += c[:, None, 1]
    corners[:, :, 2] += c[:, None, 2]
    return corners.astype(np.float32)

def in_fov_points_xyz(pts_xyz: np.ndarray, fov: Dict[str, float]) -> np.ndarray:
    x = pts_xyz[..., 0]
    y = pts_xyz[..., 1]
    z = pts_xyz[..., 2]
    r_xy = np.sqrt(x * x + y * y)
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.maximum(r_xy, 1e-12))

    daz = wrap_to_pi(az - float(fov["az0"]))
    ok_az = (daz >= float(fov["daz_min"])) & (daz <= float(fov["daz_max"]))
    ok_el = (el >= float(fov["el_min"])) & (el <= float(fov["el_max"]))
    return ok_az & ok_el

def filter_boxes_by_fov(boxes7: np.ndarray, labels: np.ndarray, fov: Optional[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    if fov is None or boxes7 is None or boxes7.shape[0] == 0:
        return boxes7, labels

    if FOV_BOX_TEST == "center":
        centers = boxes7[:, 0:3].astype(np.float64)
        keep = in_fov_points_xyz(centers, fov)
    elif FOV_BOX_TEST == "any_corner":
        corners = boxes7_to_corners_xyz(boxes7).astype(np.float64)
        keep = np.any(in_fov_points_xyz(corners, fov), axis=1)
    else:
        raise ValueError(f"Unknown FOV_BOX_TEST={FOV_BOX_TEST}")

    return boxes7[keep], labels[keep]

def filter_preds_by_fov(boxes7: np.ndarray, scores: np.ndarray, labels: np.ndarray, fov: Optional[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fov is None or boxes7 is None or boxes7.shape[0] == 0:
        return boxes7, scores, labels

    if FOV_BOX_TEST == "center":
        centers = boxes7[:, 0:3].astype(np.float64)
        keep = in_fov_points_xyz(centers, fov)
    elif FOV_BOX_TEST == "any_corner":
        corners = boxes7_to_corners_xyz(boxes7).astype(np.float64)
        keep = np.any(in_fov_points_xyz(corners, fov), axis=1)
    else:
        raise ValueError(f"Unknown FOV_BOX_TEST={FOV_BOX_TEST}")

    return boxes7[keep], scores[keep], labels[keep]


# ============================================================
# Rotated BEV IoU + NMS (pure numpy)
# ============================================================

def _rect_corners_bev_xy(box7: np.ndarray) -> np.ndarray:
    x, y, dx, dy, yaw = float(box7[0]), float(box7[1]), float(box7[3]), float(box7[4]), float(box7[6])
    hx = 0.5 * dx
    hy = 0.5 * dy
    pts = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]], dtype=np.float64)
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    pts = (R @ pts.T).T
    pts[:, 0] += x
    pts[:, 1] += y
    return pts

def _poly_area(poly: np.ndarray) -> float:
    if poly is None or poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def _is_inside(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= 0.0

def _line_intersection(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = a
    x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return p2.copy()
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py], dtype=np.float64)

def _convex_clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    output = subject
    for i in range(clipper.shape[0]):
        a = clipper[i]
        b = clipper[(i + 1) % clipper.shape[0]]
        if output.shape[0] == 0:
            break
        input_list = output
        out_pts = []
        S = input_list[-1]
        for E in input_list:
            Ein = _is_inside(E, a, b)
            Sin = _is_inside(S, a, b)
            if Ein:
                if not Sin:
                    out_pts.append(_line_intersection(S, E, a, b))
                out_pts.append(E)
            elif Sin:
                out_pts.append(_line_intersection(S, E, a, b))
            S = E
        output = np.array(out_pts, dtype=np.float64) if len(out_pts) > 0 else np.zeros((0, 2), dtype=np.float64)
    return output

def bev_iou_rotated(box_a: np.ndarray, box_b: np.ndarray) -> float:
    pa = _rect_corners_bev_xy(box_a)
    pb = _rect_corners_bev_xy(box_b)
    inter = _convex_clip(pa, pb)
    ia = _poly_area(inter)
    if ia <= 0.0:
        return 0.0
    aa = _poly_area(pa)
    ab = _poly_area(pb)
    den = aa + ab - ia
    if den <= 0.0:
        return 0.0
    return float(ia / den)

def classwise_bev_nms_rotated(
    boxes7: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_thr: float,
    max_num: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if boxes7 is None or boxes7.shape[0] == 0:
        return boxes7, scores, labels

    keep_all: List[int] = []

    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if idx.size == 0:
            continue

        order = idx[np.argsort(scores[idx])[::-1]]
        kept_cls: List[int] = []

        while order.size > 0:
            i = int(order[0])
            kept_cls.append(i)
            if order.size == 1:
                break

            rest = order[1:]
            new_rest = []
            for j in rest:
                j = int(j)
                if bev_iou_rotated(boxes7[i], boxes7[j]) <= float(iou_thr):
                    new_rest.append(j)
            order = np.array(new_rest, dtype=int)

        keep_all.extend(kept_cls)

    if len(keep_all) == 0:
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    keep_all = np.array(keep_all, dtype=int)

    if max_num is not None and int(max_num) > 0 and keep_all.size > int(max_num):
        keep_all = keep_all[np.argsort(scores[keep_all])[::-1][: int(max_num)]]

    return boxes7[keep_all], scores[keep_all], labels[keep_all]


# ============================================================
# JSON transforms (infra lidar -> vehicle lidar)
# ============================================================

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
    obj = json.loads(json_path.read_text())
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
            Rn = np.array(R_raw, dtype=np.float64)
            tn = np.array(t_raw, dtype=np.float64).reshape(-1)
            if Rn.size == 9:
                Rn = Rn.reshape(3, 3)
            if Rn.shape == (3, 3) and tn.size == 3:
                R, t = Rn, tn
                break

        q_raw = find_key(d, ["quaternion", "quat", "q"])
        if q_raw is not None and t_raw is not None:
            qn = np.array(q_raw, dtype=np.float64).reshape(-1)
            tn = np.array(t_raw, dtype=np.float64).reshape(-1)
            if tn.size == 3 and qn.size == 4:
                qx, qy, qz, qw = qn
                if abs(qn[0]) > abs(qn[3]):
                    qw, qx, qy, qz = qn
                R = quat_to_R(qx, qy, qz, qw)
                t = tn
                break

    if R is None or t is None:
        raise ValueError(f"Could not parse rigid transform JSON into (R,t): {json_path}")

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

def get_coop_root(non_kitti_root: Path) -> Path:
    cand = non_kitti_root / "cooperative-vehicle-infrastructure"
    return cand if cand.is_dir() else non_kitti_root

def index_transform_jsons(coop_root: Path, veh_novatel_key: str, veh_lidar2novatel_key: str, inf_lidar2world_key: str):
    m_veh_n2w: Dict[str, Path] = {}
    m_veh_l2n: Dict[str, Path] = {}
    m_inf_l2w: Dict[str, Path] = {}

    kn = veh_novatel_key.lower()
    kl = veh_lidar2novatel_key.lower()
    ki = inf_lidar2world_key.lower()

    for p in coop_root.rglob("*.json"):
        sid = norm_sample_id(p.stem)
        if sid is None:
            continue
        s = str(p).lower()
        if kn in s and sid not in m_veh_n2w:
            m_veh_n2w[sid] = p
        if kl in s and sid not in m_veh_l2n:
            m_veh_l2n[sid] = p
        if ki in s and sid not in m_inf_l2w:
            m_inf_l2w[sid] = p

    return m_veh_n2w, m_veh_l2n, m_inf_l2w

def compute_T_veh_from_inf(
    sid: str,
    m_veh_n2w: Dict[str, Path],
    m_veh_l2n: Dict[str, Path],
    m_inf_l2w: Dict[str, Path],
) -> np.ndarray:
    p1 = m_veh_n2w.get(sid, None)
    p2 = m_veh_l2n.get(sid, None)
    p3 = m_inf_l2w.get(sid, None)
    if p1 is None or p2 is None or p3 is None:
        raise FileNotFoundError(
            f"Missing transform json for sid={sid}. Found: "
            f"veh_n2w={p1 is not None} veh_l2n={p2 is not None} inf_l2w={p3 is not None}"
        )

    T_world_from_veh_novatel = parse_rigid_json(p1)
    T_veh_novatel_from_veh_lidar = parse_rigid_json(p2)
    T_world_from_veh_lidar = T_world_from_veh_novatel @ T_veh_novatel_from_veh_lidar

    T_world_from_inf_lidar = parse_rigid_json(p3)

    T_veh_from_world = inv_T(T_world_from_veh_lidar)
    T_veh_from_inf = T_veh_from_world @ T_world_from_inf_lidar
    return T_veh_from_inf

def transform_boxes7_upright(T: np.ndarray, boxes7: np.ndarray) -> np.ndarray:
    if boxes7 is None or boxes7.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float32)

    R = T[:3, :3].astype(np.float64)
    t = T[:3, 3].astype(np.float64)

    out = boxes7.astype(np.float64).copy()

    c = out[:, 0:3]
    c2 = (R @ c.T).T + t[None, :]
    out[:, 0:3] = c2

    yaw = out[:, 6]
    h = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1)
    h2 = (R @ h.T).T
    out[:, 6] = np.arctan2(h2[:, 1], h2[:, 0])

    return out.astype(np.float32)


# ============================================================
# Pairing
# ============================================================

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
    if hasattr(dataset, "data_list") and isinstance(dataset.data_list, list) and len(dataset.data_list) == len(dataset):
        for i, info in enumerate(dataset.data_list):
            sid = extract_pair_id_from_info(info if isinstance(info, dict) else {})
            if sid is not None and sid not in m:
                m[sid] = i
        return m
    for i in range(len(dataset)):
        info = get_data_info(dataset, i)
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


# ============================================================
# Run one side
# ============================================================

def run_one_side(dataset, model, idx: int, min_score: float):
    info = get_data_info(dataset, idx)
    sid = extract_pair_id_from_info(info)
    if sid is None:
        sid = norm_sample_id(info.get("sample_idx", idx)) or f"{idx:06d}"

    data = dataset[idx]
    data_batch = pseudo_collate([data])

    pts_pipeline = extract_points_from_batch(data_batch)

    with torch.no_grad():
        outputs = model.test_step(data_batch)
    pred_sample = outputs[0]

    pb, ps, pl = extract_pred(pred_sample, min_score=float(min_score))
    gb, gl = extract_gt_from_eval_ann_info(info)

    return dict(
        sid=sid,
        pts_pipeline=pts_pipeline,
        pred_boxes=pb.astype(np.float32),
        pred_scores=ps.astype(np.float32),
        pred_labels=pl.astype(np.int64),
        gt_boxes=gb.astype(np.float32),
        gt_labels=gl.astype(np.int64),
    )


# ============================================================
# Fusion
# ============================================================

def _match_iou_thr_for_class(class_names: List[str], cls_id: int) -> float:
    name = None
    if class_names is not None and 0 <= int(cls_id) < len(class_names):
        name = str(class_names[int(cls_id)])
    if name in MATCH_BEV_IOU_THR_BY_CLASSNAME:
        return float(MATCH_BEV_IOU_THR_BY_CLASSNAME[name])
    return float(MATCH_BEV_IOU_THR_DEFAULT)

def fuse_by_bev_iou_keep_vehicle_geom_maxscore(
    class_names: List[str],
    veh_boxes: np.ndarray, veh_scores: np.ndarray, veh_labels: np.ndarray,
    inf_boxes_v: np.ndarray, inf_scores: np.ndarray, inf_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if veh_boxes.shape[0] == 0 and inf_boxes_v.shape[0] == 0:
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    if veh_boxes.shape[0] == 0:
        return inf_boxes_v.astype(np.float32), inf_scores.astype(np.float32), inf_labels.astype(np.int64)
    if inf_boxes_v.shape[0] == 0:
        return veh_boxes.astype(np.float32), veh_scores.astype(np.float32), veh_labels.astype(np.int64)

    fused_boxes: List[np.ndarray] = []
    fused_scores: List[float] = []
    fused_labels: List[int] = []

    all_classes = np.unique(np.concatenate([veh_labels, inf_labels], axis=0)) if (veh_labels.size + inf_labels.size) > 0 else np.array([], dtype=int)

    for cls in all_classes:
        idx_v = np.where(veh_labels == cls)[0]
        idx_i = np.where(inf_labels == cls)[0]

        if idx_v.size == 0:
            for j in idx_i:
                fused_boxes.append(inf_boxes_v[j])
                fused_scores.append(float(inf_scores[j]))
                fused_labels.append(int(cls))
            continue

        if idx_i.size == 0:
            for i in idx_v:
                fused_boxes.append(veh_boxes[i])
                fused_scores.append(float(veh_scores[i]))
                fused_labels.append(int(cls))
            continue

        thr = _match_iou_thr_for_class(class_names, int(cls))

        Vc = veh_boxes[idx_v, 0:2].astype(np.float64)
        Ic = inf_boxes_v[idx_i, 0:2].astype(np.float64)
        dmat = np.linalg.norm(Vc[:, None, :] - Ic[None, :, :], axis=2)

        pairs: List[Tuple[float, int, int]] = []
        for a in range(idx_v.size):
            close_js = np.where(dmat[a] <= float(PREMATCH_CENTER_DIST_M))[0]
            for b in close_js:
                i = int(idx_v[a])
                j = int(idx_i[b])
                iou = bev_iou_rotated(veh_boxes[i], inf_boxes_v[j])
                if iou >= float(thr):
                    pairs.append((float(iou), i, j))

        pairs.sort(key=lambda x: x[0], reverse=True)

        matched_v: set = set()
        matched_i: set = set()
        for iou, i, j in pairs:
            if i in matched_v or j in matched_i:
                continue
            matched_v.add(i)
            matched_i.add(j)

            fused_boxes.append(veh_boxes[i])
            fused_scores.append(float(max(float(veh_scores[i]), float(inf_scores[j]))))
            fused_labels.append(int(cls))

        for i in idx_v:
            i = int(i)
            if i not in matched_v:
                fused_boxes.append(veh_boxes[i])
                fused_scores.append(float(veh_scores[i]))
                fused_labels.append(int(cls))

        for j in idx_i:
            j = int(j)
            if j not in matched_i:
                fused_boxes.append(inf_boxes_v[j])
                fused_scores.append(float(inf_scores[j]))
                fused_labels.append(int(cls))

    if len(fused_boxes) == 0:
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.stack(fused_boxes, axis=0).astype(np.float32),
        np.array(fused_scores, dtype=np.float32),
        np.array(fused_labels, dtype=np.int64),
    )


# ============================================================
# Vehicle test_cfg extraction (from cfg file)
# ============================================================

def get_vehicle_test_cfg(cfg_v: Config) -> Dict[str, Any]:
    tc = None
    try:
        tc = cfg_v.model.get("test_cfg", None)
    except Exception:
        tc = None
    if not isinstance(tc, dict):
        tc = {}

    nms_thr = float(tc.get("nms_thr", 0.01))
    max_num = int(tc.get("max_num", 50))
    nms_pre = int(tc.get("nms_pre", 100))
    score_thr = float(tc.get("score_thr", 0.0))
    return dict(nms_thr=nms_thr, max_num=max_num, nms_pre=nms_pre, score_thr=score_thr)


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg_vehicle", type=str)
    ap.add_argument("ckpt_vehicle", type=str)
    ap.add_argument("cfg_infra", type=str)
    ap.add_argument("ckpt_infra", type=str)

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--debug-every", type=int, default=0)

    ap.add_argument("--non-kitti-root", type=str, required=True)
    ap.add_argument("--veh-novatel-key", default="novatel_to_world", type=str)
    ap.add_argument("--veh-lidar2novatel-key", default="lidar_to_novatel", type=str)
    ap.add_argument("--inf-lidar2world-key", default="virtuallidar_to_world", type=str)

    # STEP 0 switch: vehicle-only passthrough (no infra, no fusion, no extra NMS)
    ap.add_argument("--vehicle-only", action="store_true",
                    help="Step-0 sanity: output vehicle predictions only (after vehicle FOV filtering).")

    args = ap.parse_args()

    # Allow global debug flag to force Step-0 without CLI
    if DEBUG_VEHICLE_ONLY:
        args.vehicle_only = True


    register_all_modules(init_default_scope=True)

    cfg_v = Config.fromfile(args.cfg_vehicle)
    cfg_i = Config.fromfile(args.cfg_infra)

    # class names
    class_names = None
    for cfg in [cfg_v, cfg_i]:
        if class_names is not None:
            break
        try:
            class_names = list(cfg.metainfo["classes"])
        except Exception:
            pass
        if class_names is None:
            try:
                class_names = list(cfg.class_names)
            except Exception:
                pass
    if class_names is None:
        class_names = ["Pedestrian", "Cyclist", "Car"]

    # vehicle test cfg + effective score threshold (match model inference behavior)
    test_cfg_v = get_vehicle_test_cfg(cfg_v)
    effective_min_score = float(max(float(args.min_score), float(test_cfg_v["score_thr"])))

    print(
        "[INFO] vehicle test_cfg: "
        f"score_thr={test_cfg_v['score_thr']} nms_thr={test_cfg_v['nms_thr']} "
        f"nms_pre={test_cfg_v['nms_pre']} max_num={test_cfg_v['max_num']}"
    )
    print(f"[INFO] effective_min_score = max(--min-score, vehicle score_thr) = {effective_min_score}")
    print(f"[INFO] mode: {'VEHICLE_ONLY' if args.vehicle_only else 'LATE_FUSION'}")

    # build datasets
    dataset_v = DATASETS.build(cfg_v.test_dataloader.dataset)
    dataset_i = DATASETS.build(cfg_i.test_dataloader.dataset)
    ensure_full_init(dataset_v)
    ensure_full_init(dataset_i)

    # pair ids
    id2idx_v = build_pairid_to_idx(dataset_v)
    id2idx_i = build_pairid_to_idx(dataset_i)
    common_ids = build_common_ids_in_vehicle_order(id2idx_v, id2idx_i)
    if len(common_ids) == 0:
        raise RuntimeError("No common pair-ids between vehicle and infra splits (pairing is by lidar stem).")

    n = len(common_ids)
    if args.max_samples is not None and args.max_samples > 0:
        n = min(n, int(args.max_samples))

    # transforms index (only needed for fusion, but keep indexing so output is consistent)
    non_kitti_root = Path(args.non_kitti_root)
    coop_root = get_coop_root(non_kitti_root)
    if not coop_root.exists():
        raise FileNotFoundError(f"coop_root does not exist: {coop_root}")

    print(f"[INFO] vehicle split len={len(dataset_v)}, infra split len={len(dataset_i)}")
    print(f"[INFO] common paired samples={len(common_ids)} (dumping {n})")
    print(f"[INFO] coop_root={coop_root}")

    m_veh_n2w, m_veh_l2n, m_inf_l2w = index_transform_jsons(
        coop_root=coop_root,
        veh_novatel_key=args.veh_novatel_key,
        veh_lidar2novatel_key=args.veh_lidar2novatel_key,
        inf_lidar2world_key=args.inf_lidar2world_key,
    )
    print(f"[INFO] indexed transforms: veh_n2w={len(m_veh_n2w)} veh_l2n={len(m_veh_l2n)} inf_l2w={len(m_inf_l2w)}")

    # init models
    model_v = init_model(cfg_v, args.ckpt_vehicle, device=args.device)
    model_i = init_model(cfg_i, args.ckpt_infra, device=args.device)
    model_v.eval()
    model_i.eval()

    samples_out: List[Dict[str, Any]] = []

    for k in range(n):
        sid = common_ids[k]
        idx_v = id2idx_v[sid]
        idx_i = id2idx_i[sid]

        # always run vehicle side (this defines frame + FOV)
        out_v = run_one_side(dataset_v, model_v, idx_v, min_score=effective_min_score)

        veh_fov = compute_fov_from_points(out_v["pts_pipeline"])

        vb, vs, vl = out_v["pred_boxes"], out_v["pred_scores"], out_v["pred_labels"]
        gb, gl = out_v["gt_boxes"], out_v["gt_labels"]

        # match dump_pred_gt_lidar.py filtering behavior
        if FILTER_GT_BY_PIPELINE_FOV:
            gb, gl = filter_boxes_by_fov(gb, gl, veh_fov)
        if FILTER_PRED_BY_PIPELINE_FOV:
            vb, vs, vl = filter_preds_by_fov(vb, vs, vl, veh_fov)

        if args.vehicle_only:
            # STEP 0 passthrough: vehicle predictions only (already NMS'ed by model)
            fused_boxes, fused_scores, fused_labels = vb, vs, vl
        else:
            # run infra side only when fusing
            out_i = run_one_side(dataset_i, model_i, idx_i, min_score=effective_min_score)

            # transform infra preds into vehicle frame
            T_veh_from_inf = compute_T_veh_from_inf(sid, m_veh_n2w, m_veh_l2n, m_inf_l2w)
            ib_v = transform_boxes7_upright(T_veh_from_inf, out_i["pred_boxes"])
            iscore = out_i["pred_scores"]
            ilabel = out_i["pred_labels"]

            # filter transformed infra preds by VEHICLE FOV (fairness in vehicle frame)
            if FILTER_PRED_BY_PIPELINE_FOV:
                ib_v, iscore, ilabel = filter_preds_by_fov(ib_v, iscore, ilabel, veh_fov)

            # fuse
            fused_boxes, fused_scores, fused_labels = fuse_by_bev_iou_keep_vehicle_geom_maxscore(
                class_names=class_names,
                veh_boxes=vb,
                veh_scores=vs,
                veh_labels=vl,
                inf_boxes_v=ib_v,
                inf_scores=iscore,
                inf_labels=ilabel,
            )

            # optional fused FOV filter
            if FILTER_FUSED_BY_VEH_FOV and FILTER_PRED_BY_PIPELINE_FOV:
                fused_boxes, fused_scores, fused_labels = filter_preds_by_fov(
                    fused_boxes, fused_scores, fused_labels, veh_fov
                )

            # post-fusion NMS (your NMS, not model's)
            if APPLY_POST_FUSION_NMS and fused_boxes.shape[0] > 0:
                fused_boxes, fused_scores, fused_labels = classwise_bev_nms_rotated(
                    fused_boxes, fused_scores, fused_labels,
                    iou_thr=float(test_cfg_v["nms_thr"]),
                    max_num=int(test_cfg_v["max_num"]),
                )

        samples_out.append(
            dict(
                sample_id=sid,
                fov=veh_fov,
                pred_boxes=fused_boxes.astype(np.float32),
                pred_scores=fused_scores.astype(np.float32),
                pred_labels=fused_labels.astype(np.int64),
                gt_boxes=gb.astype(np.float32),
                gt_labels=gl.astype(np.int64),
            )
        )

        if args.debug_every and (k % int(args.debug_every) == 0):
            if args.vehicle_only:
                print(
                    f"[DEBUG] k={k} sid={sid} "
                    f"vehGT={gb.shape[0]} vehPred={vb.shape[0]} passthrough={fused_boxes.shape[0]}"
                )
            else:
                # these are only defined in fusion path
                print(
                    f"[DEBUG] k={k} sid={sid} "
                    f"vehGT={gb.shape[0]} vehPred={vb.shape[0]} infPredVehFrame={ib_v.shape[0]} fusedPostNMS={fused_boxes.shape[0]}"
                )

        if (k + 1) % 50 == 0 or k == n - 1:
            print(f"[INFO] dumped {k+1}/{n}")

    out = dict(
        classes=class_names,
        fov_filter=dict(
            filter_gt=bool(FILTER_GT_BY_PIPELINE_FOV),
            filter_pred=bool(FILTER_PRED_BY_PIPELINE_FOV),
            filter_fused_by_veh_fov=bool(FILTER_FUSED_BY_VEH_FOV),
            box_test=str(FOV_BOX_TEST),
            use_percentiles=bool(USE_ANGLE_PERCENTILES),
            azimuth_pcts=tuple(AZIMUTH_PCTS),
            elevation_pcts=tuple(ELEVATION_PCTS),
            az_margin_deg=float(AZ_MARGIN_DEG),
            el_margin_deg=float(EL_MARGIN_DEG),
            min_pts_for_fov=int(MIN_PTS_FOR_FOV),
            note="All filtering uses VEHICLE pipeline FOV to match dump_pred_gt_lidar.py.",
        ),
        fusion=dict(
            mode=("vehicle_only_passthrough" if args.vehicle_only else "late_fusion"),
            fuse_policy=("vehicle_only_passthrough"
                         if args.vehicle_only else
                         "bev_iou_keep_vehicle_geom_maxscore_then_classwise_bev_nms"),
            match_bev_iou_thr_default=float(MATCH_BEV_IOU_THR_DEFAULT),
            match_bev_iou_thr_by_classname=dict(MATCH_BEV_IOU_THR_BY_CLASSNAME),
            prematch_center_dist_m=float(PREMATCH_CENTER_DIST_M),
            post_fusion_nms=bool(APPLY_POST_FUSION_NMS) if not args.vehicle_only else False,
            post_fusion_nms_thr=float(test_cfg_v["nms_thr"]),
            post_fusion_max_num=int(test_cfg_v["max_num"]),
            effective_min_score=float(effective_min_score),
            frame="vehicle_lidar",
            veh_novatel_key=str(args.veh_novatel_key),
            veh_lidar2novatel_key=str(args.veh_lidar2novatel_key),
            inf_lidar2world_key=str(args.inf_lidar2world_key),
        ),
        samples=samples_out,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(out_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[INFO] Wrote dump: {out_path}")

if __name__ == "__main__":
    main()
