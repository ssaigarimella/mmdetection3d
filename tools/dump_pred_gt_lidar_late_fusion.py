#!/usr/bin/env python3
"""
tools/dump_pred_gt_lidar_late_fusion.py

Dump GT + LATE-FUSION predictions in VEHICLE LiDAR frame for an entire paired split.

Output pickle matches tools/dump_pred_gt_lidar.py format:
  {
    "classes": [...],
    "fov_filter": {...},
    "fusion": {...},
    "samples": [
      {
        "sample_id": "000031",

        # vehicle pipeline FOV (optional sanity)
        "fov": { "az0":..., "daz_min":..., "daz_max":..., "el_min":..., "el_max":... },

        # fused preds in VEHICLE LiDAR frame
        "pred_boxes":  (N,7) float32 [x,y,z,dx,dy,dz,yaw]
        "pred_scores": (N,)  float32
        "pred_labels": (N,)  int64

        # vehicle GT in VEHICLE LiDAR frame
        "gt_boxes":    (M,7) float32
        "gt_labels":   (M,)  int64
      },
      ...
    ]
  }

Fusion behavior (matches your Open3D late-fusion baseline):
- Run vehicle detector on vehicle sample
- Run infra detector on infra sample
- Transform infra preds into vehicle LiDAR frame using JSON transforms under --non-kitti-root
- (Optional) FOV-filter per-side preds using that side's pipeline FOV
- Fuse by Hungarian matching (per class) on center distance (meters) with a distance gate
- pick_best: keep higher-score box when matched; keep all unmatched boxes
- (Optional) FOV-filter fused preds using vehicle pipeline FOV

Usage:
  python3 tools/dump_pred_gt_lidar_late_fusion.py CFG_V CKPT_V CFG_I CKPT_I \
    --non-kitti-root /path/to/dair_v2x_synth_FULL \
    --out work_dirs/late_fusion/dump_val_lidar.pkl \
    --device cuda:0
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
# IN-CODE PARAMS (match your Open3D baseline intent)
# ============================================================

# Toggle FOV filtering using pipeline points
FILTER_GT_BY_PIPELINE_FOV = True
FILTER_PRED_BY_PIPELINE_FOV = True
FILTER_FUSED_BY_VEH_FOV = True

# How to test if a box is "in FOV"
FOV_BOX_TEST = "center"   # "center" | "any_corner"

# Robust bounds from points
USE_ANGLE_PERCENTILES = True
AZIMUTH_PCTS = (0.5, 99.5)
ELEVATION_PCTS = (0.5, 99.5)

AZ_MARGIN_DEG = 0.25
EL_MARGIN_DEG = 0.25
MIN_PTS_FOR_FOV = 64

# Late fusion settings
MATCH_DIST_M = 2.0
FUSE_POLICY = "pick_best"   # "pick_best" only (baseline)

# ============================================================


# -------------------------
# small utils
# -------------------------

def ensure_full_init(dataset) -> None:
    if hasattr(dataset, "full_init"):
        try:
            dataset.full_init()
        except Exception:
            pass

def to_np(x) -> np.ndarray:
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


# -------------------------
# GT/pred extraction (7D upright boxes)
# -------------------------

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


# -------------------------
# FOV utilities (pure angular, same structure as your existing dump)
# -------------------------

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

def boxes7_to_corners_xyz(boxes7: np.ndarray) -> np.ndarray:
    """
    boxes7: (N,7) [x,y,z,dx,dy,dz,yaw]
    returns: (N,8,3)
    """
    if boxes7.shape[0] == 0:
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


# -------------------------
# JSON transforms (infra lidar -> vehicle lidar)
# Optimized for full-split dumping: index JSON files once.
# -------------------------

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
                # tolerate qw-first layouts
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
    """
    Build sid->path maps for each transform type by scanning coop_root once.
    """
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

    T_world_from_veh_novatel = parse_rigid_json(p1)      # novatel -> world
    T_veh_novatel_from_veh_lidar = parse_rigid_json(p2)  # lidar -> novatel
    T_world_from_veh_lidar = T_world_from_veh_novatel @ T_veh_novatel_from_veh_lidar

    T_world_from_inf_lidar = parse_rigid_json(p3)        # inf lidar -> world

    T_veh_from_world = inv_T(T_world_from_veh_lidar)
    T_veh_from_inf = T_veh_from_world @ T_world_from_inf_lidar
    return T_veh_from_inf

def transform_boxes7_upright(T: np.ndarray, boxes7: np.ndarray) -> np.ndarray:
    """
    Rigidly transform upright boxes into a new LiDAR frame.
    - centers: apply full SE(3)
    - yaw: rotate heading vector by R, then re-project to XY plane
    - dims unchanged
    """
    if boxes7 is None or boxes7.shape[0] == 0:
        return boxes7.astype(np.float32) if isinstance(boxes7, np.ndarray) else np.zeros((0, 7), dtype=np.float32)

    R = T[:3, :3].astype(np.float64)
    t = T[:3, 3].astype(np.float64)

    out = boxes7.astype(np.float64).copy()

    c = out[:, 0:3]
    c2 = (R @ c.T).T + t[None, :]
    out[:, 0:3] = c2

    yaw = out[:, 6]
    h = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1)  # (N,3)
    h2 = (R @ h.T).T
    out[:, 6] = np.arctan2(h2[:, 1], h2[:, 0])

    return out.astype(np.float32)


# -------------------------
# Pairing: lidar filename stem
# -------------------------

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


# -------------------------
# Hungarian + fuse
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

def fuse_preds_pick_best(
    veh_boxes: np.ndarray, veh_scores: np.ndarray, veh_labels: np.ndarray,
    inf_boxes_v: np.ndarray, inf_scores: np.ndarray, inf_labels: np.ndarray,
    match_dist_m: float,
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

    classes = np.unique(np.concatenate([veh_labels, inf_labels], axis=0)) if (veh_labels.size + inf_labels.size) > 0 else np.array([], dtype=int)

    for cls in classes:
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

        Vc = veh_boxes[idx_v, 0:3]
        Ic = inf_boxes_v[idx_i, 0:3]
        cost = np.linalg.norm(Vc[:, None, :] - Ic[None, :, :], axis=2)
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

            if float(veh_scores[i]) >= float(inf_scores[j]):
                fused_boxes.append(veh_boxes[i])
                fused_scores.append(float(veh_scores[i]))
                fused_labels.append(int(cls))
            else:
                fused_boxes.append(inf_boxes_v[j])
                fused_scores.append(float(inf_scores[j]))
                fused_labels.append(int(cls))

        for rr, i in enumerate(idx_v):
            if not used_v[rr]:
                fused_boxes.append(veh_boxes[i])
                fused_scores.append(float(veh_scores[i]))
                fused_labels.append(int(cls))

        for cc, j in enumerate(idx_i):
            if not used_i[cc]:
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


# -------------------------
# run one side
# -------------------------

def run_one_side(dataset, model, idx: int, min_score: float) -> Dict[str, Any]:
    info = get_data_info(dataset, idx)
    sid = extract_pair_id_from_info(info)
    if sid is None:
        sid = norm_sample_id(info.get("sample_idx", idx)) or f"{idx:06d}"

    data = dataset[idx]
    data_batch = pseudo_collate([data])

    pts_pipeline = extract_points_from_batch(data_batch)
    fov = compute_fov_from_points(pts_pipeline)

    with torch.no_grad():
        outputs = model.test_step(data_batch)
    pred_sample = outputs[0]

    pb, ps, pl = extract_pred(pred_sample, min_score=float(min_score))
    gb, gl = extract_gt_from_eval_ann_info(info)

    if FILTER_GT_BY_PIPELINE_FOV:
        gb, gl = filter_boxes_by_fov(gb, gl, fov)

    if FILTER_PRED_BY_PIPELINE_FOV:
        pb, ps, pl = filter_preds_by_fov(pb, ps, pl, fov)

    return dict(
        sid=sid,
        fov=fov,
        pred_boxes=pb.astype(np.float32),
        pred_scores=ps.astype(np.float32),
        pred_labels=pl.astype(np.int64),
        gt_boxes=gb.astype(np.float32),
        gt_labels=gl.astype(np.int64),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg_vehicle", type=str)
    ap.add_argument("ckpt_vehicle", type=str)
    ap.add_argument("cfg_infra", type=str)
    ap.add_argument("ckpt_infra", type=str)

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--min-score", type=float, default=0.0, help="Optional speed filter. Default keeps all preds.")
    ap.add_argument("--max-samples", type=int, default=-1, help="Debug: dump only first K paired samples.")
    ap.add_argument("--debug-every", type=int, default=0)

    ap.add_argument("--non-kitti-root", type=str, required=True)
    ap.add_argument("--veh-novatel-key", default="novatel_to_world", type=str)
    ap.add_argument("--veh-lidar2novatel-key", default="lidar_to_novatel", type=str)
    ap.add_argument("--inf-lidar2world-key", default="virtuallidar_to_world", type=str)

    ap.add_argument("--match-dist-m", type=float, default=MATCH_DIST_M)

    args = ap.parse_args()

    register_all_modules(init_default_scope=True)

    cfg_v = Config.fromfile(args.cfg_vehicle)
    cfg_i = Config.fromfile(args.cfg_infra)

    # classes: prefer cfg_v metainfo, then cfg_i, then fallback
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

    dataset_v = DATASETS.build(cfg_v.test_dataloader.dataset)
    dataset_i = DATASETS.build(cfg_i.test_dataloader.dataset)
    ensure_full_init(dataset_v)
    ensure_full_init(dataset_i)

    id2idx_v = build_pairid_to_idx(dataset_v)
    id2idx_i = build_pairid_to_idx(dataset_i)
    common_ids = build_common_ids_in_vehicle_order(id2idx_v, id2idx_i)
    if len(common_ids) == 0:
        raise RuntimeError("No common pair-ids between vehicle and infra splits (pairing is by lidar stem).")

    n = len(common_ids)
    if args.max_samples is not None and args.max_samples > 0:
        n = min(n, int(args.max_samples))

    non_kitti_root = Path(args.non_kitti_root)
    coop_root = get_coop_root(non_kitti_root)
    if not coop_root.exists():
        raise FileNotFoundError(f"coop_root does not exist: {coop_root}")

    print(f"[INFO] vehicle split len={len(dataset_v)}, infra split len={len(dataset_i)}")
    print(f"[INFO] common paired samples={len(common_ids)} (dumping {n})")
    print(f"[INFO] coop_root={coop_root}")

    # index all transforms once
    m_veh_n2w, m_veh_l2n, m_inf_l2w = index_transform_jsons(
        coop_root=coop_root,
        veh_novatel_key=args.veh_novatel_key,
        veh_lidar2novatel_key=args.veh_lidar2novatel_key,
        inf_lidar2world_key=args.inf_lidar2world_key,
    )
    print(f"[INFO] indexed transforms: veh_n2w={len(m_veh_n2w)} veh_l2n={len(m_veh_l2n)} inf_l2w={len(m_inf_l2w)}")

    model_v = init_model(cfg_v, args.ckpt_vehicle, device=args.device)
    model_i = init_model(cfg_i, args.ckpt_infra, device=args.device)
    model_v.eval()
    model_i.eval()

    samples_out: List[Dict[str, Any]] = []

    for k in range(n):
        sid = common_ids[k]
        idx_v = id2idx_v[sid]
        idx_i = id2idx_i[sid]

        out_v = run_one_side(dataset_v, model_v, idx_v, min_score=float(args.min_score))
        out_i = run_one_side(dataset_i, model_i, idx_i, min_score=float(args.min_score))

        # sanity: ensure both sides correspond to same sid when possible
        if out_v["sid"] is not None and out_v["sid"] != sid:
            pass
        if out_i["sid"] is not None and out_i["sid"] != sid:
            pass

        # compute transform and move infra preds into vehicle frame
        T_veh_from_inf = compute_T_veh_from_inf(sid, m_veh_n2w, m_veh_l2n, m_inf_l2w)
        inf_pred_boxes_v = transform_boxes7_upright(T_veh_from_inf, out_i["pred_boxes"])

        # fuse in vehicle frame
        fused_boxes, fused_scores, fused_labels = fuse_preds_pick_best(
            veh_boxes=out_v["pred_boxes"],
            veh_scores=out_v["pred_scores"],
            veh_labels=out_v["pred_labels"],
            inf_boxes_v=inf_pred_boxes_v,
            inf_scores=out_i["pred_scores"],
            inf_labels=out_i["pred_labels"],
            match_dist_m=float(args.match_dist_m),
        )

        # optional: apply vehicle FOV to fused preds (matches your Open3D “veh frame evaluation” intent)
        if FILTER_FUSED_BY_VEH_FOV:
            fused_boxes, fused_scores, fused_labels = filter_preds_by_fov(fused_boxes, fused_scores, fused_labels, out_v["fov"])

        samples_out.append(
            dict(
                sample_id=sid,
                fov=out_v["fov"],
                pred_boxes=fused_boxes.astype(np.float32),
                pred_scores=fused_scores.astype(np.float32),
                pred_labels=fused_labels.astype(np.int64),
                gt_boxes=out_v["gt_boxes"].astype(np.float32),
                gt_labels=out_v["gt_labels"].astype(np.int64),
            )
        )

        if args.debug_every and (k % int(args.debug_every) == 0):
            print(
                f"[DEBUG] k={k} sid={sid} "
                f"vehGT={out_v['gt_boxes'].shape[0]} "
                f"vehPred={out_v['pred_boxes'].shape[0]} "
                f"infPred={out_i['pred_boxes'].shape[0]} "
                f"fused={fused_boxes.shape[0]}"
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
        ),
        fusion=dict(
            match_dist_m=float(args.match_dist_m),
            fuse_policy=str(FUSE_POLICY),
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
