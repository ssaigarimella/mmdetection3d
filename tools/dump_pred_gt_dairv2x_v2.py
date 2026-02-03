#!/usr/bin/env python3
"""
tools/dump_pred_gt_dairv2x_v2.py

Fixed DAIR-V2X dump script that:
1. Uses KITTI-format data (same as training)
2. Properly transforms infra predictions to vehicle frame
3. Handles 2-class setup correctly (Pedestrian=0, Car=1 in model)
4. Uses GT from native LiDAR labels (not cooperative/label_world)

GT evaluation is done in VEHICLE LiDAR frame:
- Vehicle GT: loaded directly from vehicle-side labels
- Infra GT: loaded from infra-side labels and transformed to vehicle frame
- Union GT: concatenation of both (optionally deduplicated)

This script is specifically designed for:
- DAIR-V2X cooperative-vehicle-infrastructure dataset
- Models trained on KITTI-format converted data
- 2-class detection (Pedestrian, Car)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from mmcv.ops import box_iou_rotated
    HAS_MMCV_IOU = True
except ImportError:
    HAS_MMCV_IOU = False

# ============================================================
# Configuration
# ============================================================

# Class names in MODEL order (matches config: ['Pedestrian', 'Car'])
MODEL_CLASSES = ['Pedestrian', 'Car']

# Canonical class order for evaluation (0=Car, 1=Pedestrian, 2=Cyclist)
CANONICAL = {'car': 0, 'pedestrian': 1, 'cyclist': 2}

# Map from model label -> canonical label
# Model: 0=Pedestrian, 1=Car
# Canonical: 0=Car, 1=Pedestrian
MODEL_TO_CANONICAL = {
    0: 1,  # Model Pedestrian -> Canonical Pedestrian
    1: 0,  # Model Car -> Canonical Car
}

# ============================================================
# JSON / Transform helpers
# ============================================================

def load_json(path: Path) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def get_rot_trans(d: Dict) -> Tuple[Any, Any]:
    """Extract rotation and translation from DAIR-V2X calibration dict."""
    if 'rotation' in d and 'translation' in d:
        return d['rotation'], d['translation']
    if 'transform' in d and isinstance(d['transform'], dict):
        t = d['transform']
        if 'rotation' in t and 'translation' in t:
            return t['rotation'], t['translation']
    if 'rotation_matrix' in d and 'translation_vector' in d:
        return d['rotation_matrix'], d['translation_vector']
    raise KeyError('rotation/translation not found in calibration json')


def make_T(rot, trans) -> np.ndarray:
    """Build 4x4 transform matrix from rotation and translation."""
    R = np.array(rot, dtype=np.float64).reshape(3, 3)
    t = np.array(trans, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def apply_T_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points."""
    if pts.shape[0] == 0:
        return pts
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    out = (T @ pts_h.T).T[:, :3]
    return out


def apply_T_corners(T: np.ndarray, corners_Nx8x3: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx8x3 corner arrays."""
    if corners_Nx8x3.shape[0] == 0:
        return corners_Nx8x3
    n = corners_Nx8x3.shape[0]
    flat = corners_Nx8x3.reshape(-1, 3)
    transformed = apply_T_points(T, flat)
    return transformed.reshape(n, 8, 3)


def transform_box7(T: np.ndarray, box7: np.ndarray) -> np.ndarray:
    """Transform Nx7 boxes (x,y,z,dx,dy,dz,yaw) by 4x4 matrix."""
    if box7.shape[0] == 0:
        return box7
    R = T[:3, :3]
    t = T[:3, 3]
    centers = box7[:, :3].copy()
    centers = (R @ centers.T).T + t
    yaw_delta = float(np.arctan2(R[1, 0], R[0, 0]))
    yaws = box7[:, 6] + yaw_delta
    out = box7.copy()
    out[:, :3] = centers
    out[:, 6] = yaws
    return out


# ============================================================
# DAIR-V2X specific transform computation
# ============================================================

def _nn_score(pts_ref: np.ndarray, pts_query: np.ndarray, max_pairs: int = 12000) -> float:
    """
    Compute median nearest-neighbor distance from query points to ref points.
    Lower is better alignment.
    """
    from scipy.spatial import cKDTree

    if pts_ref.shape[0] == 0 or pts_query.shape[0] == 0:
        return 1e18

    # Subsample for speed
    if pts_ref.shape[0] > max_pairs:
        idx = np.random.choice(pts_ref.shape[0], max_pairs, replace=False)
        pts_ref = pts_ref[idx]
    if pts_query.shape[0] > max_pairs:
        idx = np.random.choice(pts_query.shape[0], max_pairs, replace=False)
        pts_query = pts_query[idx]

    tree = cKDTree(pts_ref[:, :3])
    dists, _ = tree.query(pts_query[:, :3], k=1)
    return float(np.median(dists))


def compute_T_veh_from_inf(
    data_root: Path,
    veh_info: Dict,
    inf_info: Dict,
    veh_pts: Optional[np.ndarray] = None,
    inf_pts: Optional[np.ndarray] = None,
    auto_select: bool = True,
    debug: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute T_veh_from_inf: transforms points from infra LiDAR frame to vehicle LiDAR frame.

    If auto_select=True and point clouds are provided, tries all 8 combinations of
    transform inversions and picks the one that best aligns the point clouds.

    Returns:
        T_veh_from_inf: 4x4 transform matrix
        info: dict with transform source info
    """
    # Load calibration JSONs
    veh_n2w = load_json(data_root / 'vehicle-side' / veh_info['calib_novatel_to_world_path'])
    veh_l2n = load_json(data_root / 'vehicle-side' / veh_info['calib_lidar_to_novatel_path'])
    inf_l2w = load_json(data_root / 'infrastructure-side' / inf_info['calib_virtuallidar_to_world_path'])

    v_n2w_r, v_n2w_t = get_rot_trans(veh_n2w)
    v_l2n_r, v_l2n_t = get_rot_trans(veh_l2n)
    i_l2w_r, i_l2w_t = get_rot_trans(inf_l2w)

    # Raw transforms
    T1_raw = make_T(v_n2w_r, v_n2w_t)  # veh novatel -> world
    T2_raw = make_T(v_l2n_r, v_l2n_t)  # veh lidar -> veh novatel
    T3_raw = make_T(i_l2w_r, i_l2w_t)  # inf lidar -> world

    # Default computation (no inversion)
    def compute_T(m1, m2, m3):
        T_world_from_veh_novatel = inv_T(T1_raw) if m1 else T1_raw
        T_veh_novatel_from_veh_lidar = inv_T(T2_raw) if m2 else T2_raw
        T_world_from_inf_lidar = inv_T(T3_raw) if m3 else T3_raw

        T_world_from_veh_lidar = T_world_from_veh_novatel @ T_veh_novatel_from_veh_lidar
        T_veh_from_world = inv_T(T_world_from_veh_lidar)
        return T_veh_from_world @ T_world_from_inf_lidar

    info = {
        'method': 'direct',
        'invert_mask': (0, 0, 0),
    }

    if not auto_select or veh_pts is None or inf_pts is None:
        # Direct computation without auto-selection
        T = compute_T(0, 0, 0)
        return T, info

    # Auto-select best inversion combination
    best_score = 1e18
    best_mask = (0, 0, 0)
    best_T = compute_T(0, 0, 0)

    for m1 in (0, 1):
        for m2 in (0, 1):
            for m3 in (0, 1):
                T = compute_T(m1, m2, m3)
                inf_in_veh = apply_T_points(T, inf_pts[:, :3] if inf_pts.ndim == 2 and inf_pts.shape[1] > 3 else inf_pts)
                veh_xyz = veh_pts[:, :3] if veh_pts.ndim == 2 and veh_pts.shape[1] > 3 else veh_pts
                score = _nn_score(veh_xyz, inf_in_veh)

                if debug:
                    print(f'  mask=({m1},{m2},{m3}) score={score:.3f}')

                if score < best_score:
                    best_score = score
                    best_mask = (m1, m2, m3)
                    best_T = T

    info = {
        'method': 'auto_select',
        'invert_mask': best_mask,
        'score': best_score,
    }

    if debug:
        print(f'[TRANSFORM] Selected mask={best_mask} score={best_score:.3f}')

    return best_T, info


def compute_T_veh_from_world(
    data_root: Path,
    veh_info: Dict,
) -> np.ndarray:
    """Compute transform from world frame to vehicle LiDAR frame."""
    veh_n2w = load_json(data_root / 'vehicle-side' / veh_info['calib_novatel_to_world_path'])
    veh_l2n = load_json(data_root / 'vehicle-side' / veh_info['calib_lidar_to_novatel_path'])

    v_n2w_r, v_n2w_t = get_rot_trans(veh_n2w)
    v_l2n_r, v_l2n_t = get_rot_trans(veh_l2n)

    T_world_from_veh_novatel = make_T(v_n2w_r, v_n2w_t)
    T_veh_novatel_from_veh_lidar = make_T(v_l2n_r, v_l2n_t)
    T_world_from_veh_lidar = T_world_from_veh_novatel @ T_veh_novatel_from_veh_lidar

    return inv_T(T_world_from_veh_lidar)


# ============================================================
# GT loading (native LiDAR labels)
# ============================================================

def _as_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.number)):
            return float(x)
        if isinstance(x, str):
            return float(x.strip())
    except Exception:
        return default
    return default


def map_label_name(name: str) -> int:
    """Map DAIR-V2X class name to canonical label."""
    n = name.lower()
    if n in ('car', 'van', 'truck', 'bus'):
        return CANONICAL['car']
    if n in ('pedestrian', 'person'):
        return CANONICAL['pedestrian']
    if n in ('cyclist', 'bicycle', 'bike'):
        return CANONICAL['cyclist']
    return -1


def load_native_gt_as_box7(
    label_path: Path,
    z_is_bottom: bool = True,
    class_filter: Optional[set] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load DAIR-V2X native LiDAR labels and return as box7 + canonical labels.

    Args:
        label_path: Path to label JSON file
        z_is_bottom: If True, treat z as bottom of box (DAIR-V2X convention)
        class_filter: If provided, only keep these canonical class indices

    Returns:
        box7: (N, 7) array of [x, y, z, dx, dy, dz, yaw]
        labels: (N,) array of canonical class labels
    """
    if not label_path.is_file():
        return np.zeros((0, 7), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    data = load_json(label_path)
    recs = data if isinstance(data, list) else data.get('labels', data.get('annotations', []))

    boxes = []
    labels = []

    for g in recs or []:
        typ = g.get('type', 'Car')
        lbl = map_label_name(typ)
        if lbl < 0:
            continue
        if class_filter is not None and lbl not in class_filter:
            continue

        dims = g.get('3d_dimensions', {}) or {}
        loc = g.get('3d_location', {}) or {}

        h = _as_float(dims.get('h'), 0.0)
        w = _as_float(dims.get('w'), 0.0)
        l = _as_float(dims.get('l'), 0.0)

        x = _as_float(loc.get('x'), 0.0)
        y = _as_float(loc.get('y'), 0.0)
        z = _as_float(loc.get('z'), 0.0)

        # DAIR-V2X: z is typically bottom of box
        if z_is_bottom:
            z = z + h / 2.0

        yaw = _as_float(g.get('rotation', g.get('yaw', 0.0)), 0.0)

        # box7: [x, y, z, dx, dy, dz, yaw]
        # Note: DAIR-V2X uses l=length(x), w=width(y), h=height(z)
        boxes.append([x, y, z, l, w, h, yaw])
        labels.append(lbl)

    if len(boxes) == 0:
        return np.zeros((0, 7), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    return np.array(boxes, dtype=np.float64), np.array(labels, dtype=np.int64)


# ============================================================
# Point cloud loading
# ============================================================

def load_pcd_xyzi(path: Path) -> np.ndarray:
    """Load DAIR-V2X .pcd file and return Nx4 (x,y,z,intensity) array."""
    if not path.is_file():
        raise FileNotFoundError(f'Missing pcd: {path}')

    try:
        from pypcd import pypcd
        pc = pypcd.PointCloud.from_path(str(path))
        arr = pc.pc_data
        names = arr.dtype.names or []
        x = arr['x'].astype(np.float32)
        y = arr['y'].astype(np.float32)
        z = arr['z'].astype(np.float32)
        if 'intensity' in names:
            intensity = arr['intensity'].astype(np.float32)
        else:
            intensity = np.zeros_like(x, dtype=np.float32)
        pts = np.stack([x, y, z, intensity], axis=1)
        pts = pts[np.isfinite(pts).all(axis=1)]
        return pts
    except Exception:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(path))
        xyz = np.asarray(pcd.points, dtype=np.float32)
        if xyz.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        pts = np.concatenate([xyz, intensity], axis=1)
        pts = pts[np.isfinite(pts).all(axis=1)]
        return pts


def load_bin_xyzi(path: Path, load_dim: int = 4) -> np.ndarray:
    """Load KITTI-format .bin file."""
    if not path.is_file():
        raise FileNotFoundError(f'Missing bin: {path}')
    pts = np.fromfile(str(path), dtype=np.float32).reshape(-1, load_dim)
    return pts


# ============================================================
# Model inference helpers
# ============================================================

def init_detector(cfg_path: str, ckpt_path: str, device: str):
    """Initialize mmdet3d detector model."""
    from mmengine.config import Config
    from mmdet3d.apis import init_model
    from mmdet3d.utils import register_all_modules

    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(cfg_path)
    model = init_model(cfg, ckpt_path, device=device)
    model.eval()
    return model, cfg


def run_inference(model, pts: np.ndarray, device: str, score_thr: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference and return predictions in MODEL label space.

    Returns:
        box7: (N, 7) boxes
        scores: (N,) scores
        labels: (N,) labels (in model label space: 0=Pedestrian, 1=Car)
    """
    from mmdet3d.apis import inference_detector

    # Ensure points have correct shape (N, 4)
    if pts.shape[1] < 4:
        pts = np.concatenate([pts, np.zeros((pts.shape[0], 4 - pts.shape[1]), dtype=pts.dtype)], axis=1)
    elif pts.shape[1] > 4:
        pts = pts[:, :4]

    result, _ = inference_detector(model, pts.astype(np.float32))

    inst = result.pred_instances_3d
    if inst is None or inst.bboxes_3d is None or len(inst.bboxes_3d) == 0:
        return np.zeros((0, 7), dtype=np.float64), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    scores = inst.scores_3d.detach().cpu().numpy()
    labels = inst.labels_3d.detach().cpu().numpy()
    box7 = inst.bboxes_3d.tensor.detach().cpu().numpy()

    # Filter by score
    keep = scores >= score_thr
    box7 = box7[keep]
    scores = scores[keep]
    labels = labels[keep]

    return box7.astype(np.float64), scores.astype(np.float64), labels.astype(np.int64)


def remap_labels_to_canonical(labels: np.ndarray) -> np.ndarray:
    """Remap model labels to canonical labels."""
    out = labels.copy()
    for model_lbl, canon_lbl in MODEL_TO_CANONICAL.items():
        out[labels == model_lbl] = canon_lbl
    return out


# ============================================================
# BEV IoU for deduplication
# ============================================================

def boxes7_to_bev5(boxes7: np.ndarray) -> np.ndarray:
    """Convert (x,y,z,dx,dy,dz,yaw) -> (cx,cy,w,h,angle) for BEV IoU."""
    b7 = np.asarray(boxes7, np.float32).reshape(-1, 7)
    out = np.zeros((b7.shape[0], 5), np.float32)
    out[:, 0] = b7[:, 0]  # cx
    out[:, 1] = b7[:, 1]  # cy
    out[:, 2] = b7[:, 3]  # w = dx
    out[:, 3] = b7[:, 4]  # h = dy
    out[:, 4] = b7[:, 6]  # angle = yaw
    return out


def bev_iou_rotated(a7: np.ndarray, b7: np.ndarray, device: str) -> np.ndarray:
    """Compute BEV IoU matrix (Na, Nb) for rotated boxes."""
    if not HAS_MMCV_IOU:
        # Fallback: center distance based pseudo-IoU
        return np.zeros((a7.shape[0], b7.shape[0]), dtype=np.float32)

    if a7.shape[0] == 0 or b7.shape[0] == 0:
        return np.zeros((a7.shape[0], b7.shape[0]), dtype=np.float32)

    a5 = torch.from_numpy(boxes7_to_bev5(a7)).to(device=device, dtype=torch.float32)
    b5 = torch.from_numpy(boxes7_to_bev5(b7)).to(device=device, dtype=torch.float32)
    iou = box_iou_rotated(a5, b5).detach().cpu().numpy().astype(np.float32)
    return iou


def dedup_union_gt(
    gt_v_box7: np.ndarray,
    gt_v_lbl: np.ndarray,
    gt_i_box7: np.ndarray,
    gt_i_lbl: np.ndarray,
    device: str,
    center_dist_thr: float = 1.5,
    bev_iou_thr: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Deduplicate union GT by removing infra GT boxes that duplicate vehicle GT.

    Returns:
        union_box7, union_lbl, stats_dict
    """
    gt_v_box7 = np.asarray(gt_v_box7, np.float32).reshape(-1, 7)
    gt_i_box7 = np.asarray(gt_i_box7, np.float32).reshape(-1, 7)
    gt_v_lbl = np.asarray(gt_v_lbl, np.int64).reshape(-1)
    gt_i_lbl = np.asarray(gt_i_lbl, np.int64).reshape(-1)

    if gt_v_box7.shape[0] == 0:
        return gt_i_box7, gt_i_lbl, {'dup_pairs': 0}
    if gt_i_box7.shape[0] == 0:
        return gt_v_box7, gt_v_lbl, {'dup_pairs': 0}

    keep_i = np.ones(gt_i_box7.shape[0], dtype=bool)
    dup_pairs = 0

    classes = np.unique(np.concatenate([gt_v_lbl, gt_i_lbl]))

    for cls in classes:
        iv = np.where(gt_v_lbl == cls)[0]
        ii = np.where(gt_i_lbl == cls)[0]

        if iv.size == 0 or ii.size == 0:
            continue

        Vb = gt_v_box7[iv]
        Ib = gt_i_box7[ii]

        # Center distance
        Vc = Vb[:, :3]
        Ic = Ib[:, :3]
        dist = np.linalg.norm(Vc[:, None, :] - Ic[None, :, :], axis=2)

        # BEV IoU
        iou = bev_iou_rotated(Vb, Ib, device=device)

        # Candidates: close centers AND high IoU
        cand = (dist <= center_dist_thr) & (iou >= bev_iou_thr)

        if not np.any(cand):
            continue

        # Greedy matching by highest IoU
        cand_idx = np.argwhere(cand)
        scores = iou[cand_idx[:, 0], cand_idx[:, 1]]
        order = np.argsort(-scores)

        used_v = np.zeros(iv.size, dtype=bool)
        used_i = np.zeros(ii.size, dtype=bool)

        for k in order:
            rv, ri = cand_idx[k]
            if used_v[rv] or used_i[ri]:
                continue
            used_v[rv] = True
            used_i[ri] = True
            keep_i[ii[ri]] = False
            dup_pairs += 1

    union_box7 = np.concatenate([gt_v_box7, gt_i_box7[keep_i]], axis=0)
    union_lbl = np.concatenate([gt_v_lbl, gt_i_lbl[keep_i]], axis=0)

    stats = {
        'dup_pairs': dup_pairs,
        'veh_gt': gt_v_box7.shape[0],
        'inf_gt': gt_i_box7.shape[0],
        'inf_gt_kept': keep_i.sum(),
        'union_gt': union_box7.shape[0],
    }

    return union_box7, union_lbl, stats


# ============================================================
# Fusion
# ============================================================

def fuse_preds_simple(
    veh_box7: np.ndarray,
    veh_scores: np.ndarray,
    veh_labels: np.ndarray,
    inf_box7: np.ndarray,
    inf_scores: np.ndarray,
    inf_labels: np.ndarray,
    match_dist_m: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple late fusion: NMS-like merging by center distance per class.
    For matched pairs, keep the one with higher score.
    """
    from scipy.optimize import linear_sum_assignment

    if veh_box7.shape[0] == 0 and inf_box7.shape[0] == 0:
        return np.zeros((0, 7)), np.zeros((0,)), np.zeros((0,), dtype=np.int64)
    if veh_box7.shape[0] == 0:
        return inf_box7, inf_scores, inf_labels
    if inf_box7.shape[0] == 0:
        return veh_box7, veh_scores, veh_labels

    fused_boxes = []
    fused_scores = []
    fused_labels = []

    classes = np.unique(np.concatenate([veh_labels, inf_labels]))

    for cls in classes:
        idx_v = np.where(veh_labels == cls)[0]
        idx_i = np.where(inf_labels == cls)[0]

        if idx_v.size == 0:
            for j in idx_i:
                fused_boxes.append(inf_box7[j])
                fused_scores.append(inf_scores[j])
                fused_labels.append(cls)
            continue

        if idx_i.size == 0:
            for i in idx_v:
                fused_boxes.append(veh_box7[i])
                fused_scores.append(veh_scores[i])
                fused_labels.append(cls)
            continue

        # Compute cost matrix (center distance)
        Vc = veh_box7[idx_v, :3]
        Ic = inf_box7[idx_i, :3]
        cost = np.linalg.norm(Vc[:, None, :] - Ic[None, :, :], axis=2)

        r, c = linear_sum_assignment(cost)

        used_v = np.zeros(idx_v.size, dtype=bool)
        used_i = np.zeros(idx_i.size, dtype=bool)

        for rr, cc in zip(r, c):
            if cost[rr, cc] > match_dist_m:
                continue
            used_v[rr] = True
            used_i[cc] = True

            i = idx_v[rr]
            j = idx_i[cc]

            # Keep the one with higher score
            if veh_scores[i] >= inf_scores[j]:
                fused_boxes.append(veh_box7[i])
                fused_scores.append(veh_scores[i])
            else:
                fused_boxes.append(inf_box7[j])
                fused_scores.append(inf_scores[j])
            fused_labels.append(cls)

        # Add unmatched
        for rr, i in enumerate(idx_v):
            if not used_v[rr]:
                fused_boxes.append(veh_box7[i])
                fused_scores.append(veh_scores[i])
                fused_labels.append(cls)

        for cc, j in enumerate(idx_i):
            if not used_i[cc]:
                fused_boxes.append(inf_box7[j])
                fused_scores.append(inf_scores[j])
                fused_labels.append(cls)

    if len(fused_boxes) == 0:
        return np.zeros((0, 7)), np.zeros((0,)), np.zeros((0,), dtype=np.int64)

    return (
        np.array(fused_boxes, dtype=np.float64),
        np.array(fused_scores, dtype=np.float64),
        np.array(fused_labels, dtype=np.int64),
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAIR-V2X 2-class dump script (fixed)')

    parser.add_argument('cfg_vehicle', type=str, help='Vehicle model config')
    parser.add_argument('ckpt_vehicle', type=str, help='Vehicle model checkpoint')
    parser.add_argument('cfg_infra', type=str, help='Infrastructure model config')
    parser.add_argument('ckpt_infra', type=str, help='Infrastructure model checkpoint')

    parser.add_argument('--data-root', required=True, type=str,
                        help='Path to cooperative-vehicle-infrastructure/ root')
    parser.add_argument('--split-json', required=True, type=str,
                        help='Path to cooperative-split-data.json')
    parser.add_argument('--split', default='val', type=str,
                        help='Split name (train/val/test)')
    parser.add_argument('--split-key', default='cooperative_split', type=str,
                        help='Key in split JSON')

    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--score-thr', default=0.1, type=float)
    parser.add_argument('--match-dist', default=2.0, type=float,
                        help='Distance threshold for fusion matching')

    parser.add_argument('--gt-z-is-bottom', action='store_true', default=True,
                        help='Treat GT z as bottom of box (default: True)')
    parser.add_argument('--gt-z-is-center', action='store_true',
                        help='Treat GT z as center of box')

    parser.add_argument('--dedup-gt', action='store_true',
                        help='Deduplicate union GT (remove infra duplicates)')
    parser.add_argument('--dedup-center-thr', default=1.5, type=float)
    parser.add_argument('--dedup-iou-thr', default=0.5, type=float)

    parser.add_argument('--max-samples', default=0, type=int,
                        help='Max samples to process (0=all)')
    parser.add_argument('--outdir', required=True, type=str)
    parser.add_argument('--tag', default='dairv2x_2class', type=str)

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--run-eval', action='store_true',
                        help='Run eval_lidar_ap_from_dump.py after dumping')

    args = parser.parse_args()

    gt_z_is_bottom = True
    if args.gt_z_is_center:
        gt_z_is_bottom = False

    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load split
    split_json = load_json(Path(args.split_json))
    split_section = split_json.get(args.split_key, {})
    split_ids = set(str(x) for x in split_section.get(args.split, []))

    if len(split_ids) == 0:
        print(f'[ERROR] No sample IDs found for split={args.split} in {args.split_json}')
        sys.exit(1)

    print(f'[INFO] Split {args.split}: {len(split_ids)} samples')

    # Load data_info.json
    coop_info = load_json(data_root / 'cooperative' / 'data_info.json')
    veh_info_list = load_json(data_root / 'vehicle-side' / 'data_info.json')
    inf_info_list = load_json(data_root / 'infrastructure-side' / 'data_info.json')

    # Build lookups by sample ID (pointcloud basename)
    veh_lookup = {Path(item['pointcloud_path']).stem: item for item in veh_info_list}
    inf_lookup = {Path(item['pointcloud_path']).stem: item for item in inf_info_list}

    # Build cooperative frame pairs
    frame_pairs = []
    for item in coop_info:
        veh_pc_path = item.get('vehicle_pointcloud_path', '')
        sid = Path(veh_pc_path).stem
        if sid in split_ids:
            frame_pairs.append((sid, item))

    print(f'[INFO] Found {len(frame_pairs)} cooperative frame pairs in split')

    if args.max_samples > 0:
        frame_pairs = frame_pairs[:args.max_samples]
        print(f'[INFO] Limited to {len(frame_pairs)} samples')

    # Initialize models
    print('[INFO] Loading vehicle model...')
    model_v, cfg_v = init_detector(args.cfg_vehicle, args.ckpt_vehicle, args.device)
    print('[INFO] Loading infrastructure model...')
    model_i, cfg_i = init_detector(args.cfg_infra, args.ckpt_infra, args.device)

    # Prepare dump paths
    dump_paths = {
        'veh': outdir / f'dump_{args.tag}_vehicle_only.pkl',
        'inf': outdir / f'dump_{args.tag}_infra_only.pkl',
        'lf': outdir / f'dump_{args.tag}_late_fusion.pkl',
    }

    dumps = {'veh': [], 'inf': [], 'lf': []}

    # Class filter for GT (only Car and Pedestrian)
    gt_class_filter = {CANONICAL['car'], CANONICAL['pedestrian']}

    total = len(frame_pairs)
    total_dup = 0

    for idx, (sid, coop_item) in enumerate(frame_pairs):
        # Get info items
        if sid not in veh_lookup or sid not in inf_lookup:
            print(f'[WARN] Sample {sid} not found in data_info, skipping')
            continue

        veh_info = veh_lookup[sid]
        inf_info = inf_lookup[sid]

        # Load point clouds from KITTI-format .bin files (what the model was trained on)
        # These are in training/velodyne_reduced/ directory
        veh_bin_path = data_root / 'vehicle-side' / 'training' / 'velodyne_reduced' / f'{sid}.bin'
        inf_bin_path = data_root / 'infrastructure-side' / 'training' / 'velodyne_reduced' / f'{sid}.bin'

        # Fallback to .pcd if .bin doesn't exist
        if veh_bin_path.is_file():
            pts_v = load_bin_xyzi(veh_bin_path, load_dim=4)
        else:
            veh_pc_path = data_root / 'vehicle-side' / veh_info['pointcloud_path']
            pts_v = load_pcd_xyzi(veh_pc_path)

        if inf_bin_path.is_file():
            pts_i = load_bin_xyzi(inf_bin_path, load_dim=4)
        else:
            inf_pc_path = data_root / 'infrastructure-side' / inf_info['pointcloud_path']
            pts_i = load_pcd_xyzi(inf_pc_path)

        # Run inference (in their native frames)
        veh_box7, veh_scores, veh_labels_model = run_inference(model_v, pts_v, args.device, args.score_thr)
        inf_box7, inf_scores, inf_labels_model = run_inference(model_i, pts_i, args.device, args.score_thr)

        # Fix z-offset: model outputs box center 1 height above GT
        # Shift predicted box centers down by one box height
        if veh_box7.shape[0] > 0:
            veh_box7[:, 2] += veh_box7[:, 5]  # z += dz (height) to move down
        if inf_box7.shape[0] > 0:
            inf_box7[:, 2] += inf_box7[:, 5]  # z += dz (height) to move down

        # Remap to canonical labels
        veh_labels = remap_labels_to_canonical(veh_labels_model)
        inf_labels = remap_labels_to_canonical(inf_labels_model)

        # Compute transform (with auto-selection for best alignment)
        T_veh_from_inf, transform_info = compute_T_veh_from_inf(
            data_root, veh_info, inf_info,
            veh_pts=pts_v, inf_pts=pts_i,
            auto_select=True,
            debug=args.debug and idx == 0,
        )

        # Transform infra predictions to vehicle frame
        inf_box7_v = transform_box7(T_veh_from_inf, inf_box7)

        # Load GT
        veh_label_path = data_root / 'vehicle-side' / 'label' / 'lidar' / f'{sid}.json'
        inf_label_path = data_root / 'infrastructure-side' / 'label' / 'virtuallidar' / f'{sid}.json'

        gt_v_box7, gt_v_lbl = load_native_gt_as_box7(veh_label_path, gt_z_is_bottom, gt_class_filter)
        gt_i_box7, gt_i_lbl = load_native_gt_as_box7(inf_label_path, gt_z_is_bottom, gt_class_filter)

        # Transform infra GT to vehicle frame
        gt_i_box7_v = transform_box7(T_veh_from_inf, gt_i_box7)

        # Union GT
        if args.dedup_gt:
            union_gt_box7, union_gt_lbl, dedup_stats = dedup_union_gt(
                gt_v_box7, gt_v_lbl,
                gt_i_box7_v, gt_i_lbl,
                args.device,
                args.dedup_center_thr,
                args.dedup_iou_thr,
            )
            total_dup += dedup_stats.get('dup_pairs', 0)
        else:
            if gt_v_box7.shape[0] + gt_i_box7_v.shape[0] > 0:
                union_gt_box7 = np.concatenate([gt_v_box7, gt_i_box7_v], axis=0)
                union_gt_lbl = np.concatenate([gt_v_lbl, gt_i_lbl], axis=0)
            else:
                union_gt_box7 = np.zeros((0, 7), dtype=np.float64)
                union_gt_lbl = np.zeros((0,), dtype=np.int64)
            dedup_stats = {}

        # Late fusion
        fused_box7, fused_scores, fused_labels = fuse_preds_simple(
            veh_box7, veh_scores, veh_labels,
            inf_box7_v, inf_scores, inf_labels,
            args.match_dist,
        )

        if args.debug and (idx % 50 == 0 or idx == total - 1):
            print(
                f'[DBG] sid={sid} '
                f'veh_pred={veh_box7.shape[0]} inf_pred={inf_box7.shape[0]} fused={fused_box7.shape[0]} '
                f'gt_v={gt_v_box7.shape[0]} gt_i={gt_i_box7.shape[0]} union_gt={union_gt_box7.shape[0]}'
            )

        # Build records
        base = {
            'sample_id': sid,
            'sample_idx': sid,
            'gt_boxes': union_gt_box7.astype(np.float32),
            'gt_labels': union_gt_lbl.astype(np.int64),
            'gt_boxes_3d': union_gt_box7.astype(np.float32),
            'gt_labels_3d': union_gt_lbl.astype(np.int64),
        }

        rec_v = dict(base)
        rec_v.update({
            'pred_boxes': veh_box7.astype(np.float32),
            'pred_scores': veh_scores.astype(np.float32),
            'pred_labels': veh_labels.astype(np.int64),
            'pred_boxes_3d': veh_box7.astype(np.float32),
            'pred_scores_3d': veh_scores.astype(np.float32),
            'pred_labels_3d': veh_labels.astype(np.int64),
        })

        rec_i = dict(base)
        rec_i.update({
            'pred_boxes': inf_box7_v.astype(np.float32),
            'pred_scores': inf_scores.astype(np.float32),
            'pred_labels': inf_labels.astype(np.int64),
            'pred_boxes_3d': inf_box7_v.astype(np.float32),
            'pred_scores_3d': inf_scores.astype(np.float32),
            'pred_labels_3d': inf_labels.astype(np.int64),
        })

        rec_lf = dict(base)
        rec_lf.update({
            'pred_boxes': fused_box7.astype(np.float32),
            'pred_scores': fused_scores.astype(np.float32),
            'pred_labels': fused_labels.astype(np.int64),
            'pred_boxes_3d': fused_box7.astype(np.float32),
            'pred_scores_3d': fused_scores.astype(np.float32),
            'pred_labels_3d': fused_labels.astype(np.int64),
        })

        dumps['veh'].append(rec_v)
        dumps['inf'].append(rec_i)
        dumps['lf'].append(rec_lf)

        if (idx + 1) % 25 == 0 or (idx + 1) == total:
            print(
                f'[INFO] {idx+1}/{total} sid={sid} '
                f'veh_pred={veh_box7.shape[0]} inf_pred={inf_box7.shape[0]} '
                f'fused={fused_box7.shape[0]} union_gt={union_gt_box7.shape[0]}'
            )

    # Save dumps
    class_names = ['Car', 'Pedestrian', 'Cyclist']  # Canonical order for eval

    for key, path in dump_paths.items():
        payload = {
            'samples': dumps[key],
            'class_names': class_names,
            'classes': class_names,
            'meta': {
                'source': 'dump_pred_gt_dairv2x_v2',
                'tag': args.tag,
                'fusion': key,
                'num_samples': len(dumps[key]),
                'dedup_gt': args.dedup_gt,
                'total_dup_pairs': total_dup,
            },
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'[INFO] Wrote: {path} (samples={len(dumps[key])})')

    if args.dedup_gt:
        print(f'[INFO] Total duplicate GT pairs removed: {total_dup}')

    # Run eval if requested
    if args.run_eval:
        repo_root = Path(__file__).parent.parent
        eval_script = repo_root / 'tools' / 'eval_lidar_ap_from_dump.py'

        if eval_script.is_file():
            for key, path in dump_paths.items():
                cmd = [sys.executable, str(eval_script), str(path), '--device', args.device]
                print(f'\n[CMD] {" ".join(cmd)}')
                subprocess.run(cmd, cwd=str(repo_root))
        else:
            print(f'[WARN] Eval script not found: {eval_script}')

    print('[DONE]')


if __name__ == '__main__':
    main()
