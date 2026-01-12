#!/usr/bin/env python3
"""
tools/dump_pred_gt_lidar.py

Dump predictions + GT boxes in LiDAR frame for the ENTIRE split in cfg.test_dataloader.dataset.

No transforms. No KITTI camera conversion. No calib.

IMPORTANT (FOV-correct):
- Optionally filters BOTH GT and preds by the *reduced/pipeline LiDAR FOV* derived
  from the pipeline point cloud angles (azimuth + elevation).
- This is purely an angular FOV test, NOT "points inside bbox".

Output is a pickle with:
  {
    "classes": [...],
    "fov_filter": {...},
    "samples": [
      {
        "sample_id": "000002",
        "fov": {
          "az0": float,          # radians, circular mean direction used for unwrap
          "daz_min": float,      # radians (relative to az0)
          "daz_max": float,      # radians (relative to az0)
          "el_min": float,       # radians
          "el_max": float,       # radians
        },

        "pred_boxes": (N,7) float32 [x,y,z,dx,dy,dz,yaw]   # after optional FOV filter
        "pred_scores": (N,) float32
        "pred_labels": (N,) int64 in [0..C-1]

        "gt_boxes": (M,7) float32                          # after optional FOV filter
        "gt_labels": (M,) int64
      },
      ...
    ]
  }

Usage:
  python3 tools/dump_pred_gt_lidar.py CFG CKPT --out dump.pkl
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from mmengine.config import Config
from mmengine.registry import DATASETS
from mmengine.dataset import pseudo_collate

from mmdet3d.apis import init_model
from mmdet3d.utils import register_all_modules


# ============================================================
# IN-CODE PARAMS
# ============================================================

# Toggle FOV filtering
FILTER_GT_BY_PIPELINE_FOV = True
FILTER_PRED_BY_PIPELINE_FOV = True

# How to test if a box is "in FOV"
#   "center"     : box center angles must be inside [az,el] bounds
#   "any_corner" : any 3D corner angles inside [az,el] bounds
FOV_BOX_TEST = "center"   # "center" | "any_corner"

# Robust bounds from points
# Use percentiles to avoid rare outlier points widening the FOV
USE_ANGLE_PERCENTILES = True
AZIMUTH_PCTS = (0.5, 99.5)     # percentiles over unwrapped delta-azimuth
ELEVATION_PCTS = (0.5, 99.5)   # percentiles over elevation

# Extra angular margin (degrees) around derived FOV
AZ_MARGIN_DEG = 0.25
EL_MARGIN_DEG = 0.25

# Minimum number of pipeline points to trust FOV; otherwise no filtering (keeps all boxes)
MIN_PTS_FOR_FOV = 64

# ============================================================


def norm_sample_id(x: Any) -> str:
    if x is None:
        return "000000"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):06d}"
    s = str(x).strip()
    if s.isdigit():
        return s.zfill(6) if len(s) <= 6 else s
    return s


def ensure_full_init(dataset) -> None:
    if hasattr(dataset, "full_init"):
        try:
            dataset.full_init()
        except Exception:
            pass


def get_data_info(dataset, idx: int) -> Dict:
    if hasattr(dataset, "get_data_info"):
        out = dataset.get_data_info(idx)
        return out if isinstance(out, dict) else {}
    if hasattr(dataset, "data_list") and isinstance(dataset.data_list, list):
        if 0 <= idx < len(dataset.data_list):
            return dataset.data_list[idx] if isinstance(dataset.data_list[idx], dict) else {}
    if hasattr(dataset, "infos") and isinstance(dataset.infos, list):
        if 0 <= idx < len(dataset.infos):
            return dataset.infos[idx] if isinstance(dataset.infos[idx], dict) else {}
    return {}


def to_np(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_points_from_batch(data_batch: Dict) -> np.ndarray:
    """
    Returns (N,3) float64 pipeline points from the dataset pipeline batch.
    This is the reduced/frustum-clipped cloud if that's what your pipeline provides.
    """
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
# FOV utilities (pure angular)
# -------------------------

def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def compute_fov_from_points(pts_xyz: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Derive an angular FOV from the pipeline points:
      az = atan2(y, x)
      el = atan2(z, sqrt(x^2 + y^2))

    Handles az wrap-around by unwrapping relative to circular mean az0, then taking bounds
    on delta-azimuth daz in [-pi, pi].

    Returns dict with az0, daz_min, daz_max, el_min, el_max (radians).
    """
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

    # circular mean for az
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

    # margins
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
    """
    boxes7: (N,7) [x,y,z,dx,dy,dz,yaw]
    returns: (N,8,3)
    """
    if boxes7.shape[0] == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)

    c = boxes7[:, 0:3].astype(np.float32)  # (N,3)
    dx = boxes7[:, 3].astype(np.float32)
    dy = boxes7[:, 4].astype(np.float32)
    dz = boxes7[:, 5].astype(np.float32)
    yaw = boxes7[:, 6].astype(np.float32)

    hx = 0.5 * dx
    hy = 0.5 * dy
    hz = 0.5 * dz

    # 8 corners in local box frame
    # order doesn't matter for our "any corner inside" test
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
    )  # (8,3)

    corners = corners_local[None, :, :].copy()  # (1,8,3)
    corners = np.repeat(corners, boxes7.shape[0], axis=0)  # (N,8,3)

    corners[:, :, 0] *= hx[:, None]
    corners[:, :, 1] *= hy[:, None]
    corners[:, :, 2] *= hz[:, None]

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # rotate XY
    x0 = corners[:, :, 0].copy()
    y0 = corners[:, :, 1].copy()
    corners[:, :, 0] = cy[:, None] * x0 - sy[:, None] * y0
    corners[:, :, 1] = sy[:, None] * x0 + cy[:, None] * y0

    # translate
    corners[:, :, 0] += c[:, None, 0]
    corners[:, :, 1] += c[:, None, 1]
    corners[:, :, 2] += c[:, None, 2]

    return corners.astype(np.float32)


def in_fov_points_xyz(pts_xyz: np.ndarray, fov: Dict[str, float]) -> np.ndarray:
    """
    pts_xyz: (...,3)
    returns mask (...)
    """
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
        corners = boxes7_to_corners_xyz(boxes7).astype(np.float64)  # (N,8,3)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", type=str)
    ap.add_argument("ckpt", type=str)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--min-score", type=float, default=0.0, help="Optional speed filter. Default keeps all preds.")
    ap.add_argument("--max-samples", type=int, default=-1, help="Debug: dump only first K samples.")
    ap.add_argument("--debug-every", type=int, default=0, help="If >0, prints FOV stats every K samples.")
    args = ap.parse_args()

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.cfg)

    # classes: try common locations
    class_names = None
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

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    ensure_full_init(dataset)

    model = init_model(cfg, args.ckpt, device=args.device)
    model.eval()

    n = len(dataset)
    if args.max_samples is not None and args.max_samples > 0:
        n = min(n, int(args.max_samples))

    samples_out = []
    for idx in range(n):
        info = get_data_info(dataset, idx)
        sid = norm_sample_id(info.get("sample_idx", idx))

        data = dataset[idx]
        data_batch = pseudo_collate([data])

        # pipeline points => FOV
        pts_pipeline = extract_points_from_batch(data_batch)
        fov = compute_fov_from_points(pts_pipeline)

        with torch.no_grad():
            outputs = model.test_step(data_batch)
        pred_sample = outputs[0]

        pb, ps, pl = extract_pred(pred_sample, min_score=float(args.min_score))
        gb, gl = extract_gt_from_eval_ann_info(info)

        # Apply FOV filtering (pure angles) if enabled and fov is available
        if FILTER_GT_BY_PIPELINE_FOV:
            gb, gl = filter_boxes_by_fov(gb, gl, fov)

        if FILTER_PRED_BY_PIPELINE_FOV:
            pb, ps, pl = filter_preds_by_fov(pb, ps, pl, fov)

        samples_out.append(
            dict(
                sample_id=sid,
                fov=fov,
                pred_boxes=pb.astype(np.float32),
                pred_scores=ps.astype(np.float32),
                pred_labels=pl.astype(np.int64),
                gt_boxes=gb.astype(np.float32),
                gt_labels=gl.astype(np.int64),
            )
        )

        if args.debug_every and (idx % int(args.debug_every) == 0) and fov is not None:
            print(
                f"[DEBUG] idx={idx} sid={sid} "
                f"az0={fov['az0']:.3f} daz=[{fov['daz_min']:.3f},{fov['daz_max']:.3f}] "
                f"el=[{fov['el_min']:.3f},{fov['el_max']:.3f}] "
                f"GT={gb.shape[0]} Pred={pb.shape[0]}"
            )

        if (idx + 1) % 50 == 0 or idx == n - 1:
            print(f"[INFO] dumped {idx+1}/{n}")

    out = dict(
        classes=class_names,
        fov_filter=dict(
            filter_gt=bool(FILTER_GT_BY_PIPELINE_FOV),
            filter_pred=bool(FILTER_PRED_BY_PIPELINE_FOV),
            box_test=str(FOV_BOX_TEST),
            use_percentiles=bool(USE_ANGLE_PERCENTILES),
            azimuth_pcts=tuple(AZIMUTH_PCTS),
            elevation_pcts=tuple(ELEVATION_PCTS),
            az_margin_deg=float(AZ_MARGIN_DEG),
            el_margin_deg=float(EL_MARGIN_DEG),
            min_pts_for_fov=int(MIN_PTS_FOR_FOV),
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
