#!/usr/bin/env python3
"""
test_late_fusion_lidar_hungarian.py

Late fusion for LiDAR detections, following the DAIR-V2X description:
- Train vehicle-view and infrastructure-view detectors separately
- Convert infrastructure predictions into vehicle LiDAR frame (T_i2v)
- Match by Euclidean distance (XY) using Hungarian assignment
- Merge matched predictions

This script is compatible with MMEngine DumpResults outputs where each sample is a dict
and pred_instances_3d is also a dict:
  pred_instances_3d = { 'bboxes_3d': ..., 'scores_3d': ..., 'labels_3d': ... }

It outputs a fused PKL in the same "list of dict" structure as the vehicle PKL.
"""

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:
    raise RuntimeError("This script requires scipy for Hungarian matching: pip install scipy") from e

try:
    import torch
except Exception as e:
    raise RuntimeError("This script requires torch.") from e

try:
    from mmengine.fileio import load as mmload
    from mmengine.fileio import dump as mmdump
except Exception as e:
    raise RuntimeError("This script requires mmengine (same env as MMDetection3D).") from e

try:
    from mmengine.structures import InstanceData
except Exception:
    InstanceData = None  # type: ignore

try:
    from mmdet3d.structures import LiDARInstance3DBoxes
except Exception as e:
    raise RuntimeError("This script requires mmdet3d (MMDetection3D).") from e


# -----------------------
# Small utilities
# -----------------------

def _to_numpy(x: Any) -> np.ndarray:
    if x is None:
        return np.asarray([])
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_torch_1d(x: Any, dtype: torch.dtype) -> torch.Tensor:
    if x is None:
        return torch.zeros((0,), dtype=dtype)
    if torch.is_tensor(x):
        return x.to(dtype=dtype)
    arr = np.asarray(x)
    if arr.size == 0:
        return torch.zeros((0,), dtype=dtype)
    return torch.from_numpy(arr).to(dtype=dtype)


def _parse_matrix4x4(obj: Any) -> np.ndarray:
    arr = np.array(obj, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.size == 16:
        return arr.reshape(4, 4)
    raise ValueError(f"Transform must be 4x4 or flat 16, got shape {arr.shape}.")


def load_i2v_from_json(json_path: Path) -> Dict[str, np.ndarray]:
    with open(json_path, "r") as f:
        data = json.load(f)
    out: Dict[str, np.ndarray] = {}
    for k, v in data.items():
        out[str(k)] = _parse_matrix4x4(v)
    return out


def load_i2v_for_sample(sample_idx: str,
                        i2v_map: Optional[Dict[str, np.ndarray]],
                        i2v_dir: Optional[Path]) -> np.ndarray:
    if i2v_map is not None:
        if sample_idx not in i2v_map:
            raise KeyError(f"sample_idx {sample_idx} not found in --i2v-json mapping.")
        return i2v_map[sample_idx]

    if i2v_dir is not None:
        txt = i2v_dir / f"{sample_idx}.txt"
        jsn = i2v_dir / f"{sample_idx}.json"
        if txt.is_file():
            mat = np.loadtxt(txt, dtype=np.float64)
            return _parse_matrix4x4(mat)
        if jsn.is_file():
            with open(jsn, "r") as f:
                mat = json.load(f)
            return _parse_matrix4x4(mat)
        raise FileNotFoundError(f"Could not find {txt} or {jsn} for sample_idx={sample_idx}")

    # If you do not provide transforms, we assume identity (mostly useful for debugging).
    return np.eye(4, dtype=np.float64)


def get_sample_idx(item: Any, fallback_i: int) -> str:
    # Common case: list entries are dicts dumped by MMEngine
    if isinstance(item, dict):
        for key in ["sample_idx", "lidar_idx", "frame_id", "sample_id"]:
            if key in item:
                return str(item[key])
        if "metainfo" in item and isinstance(item["metainfo"], dict):
            for key in ["sample_idx", "lidar_idx", "frame_id", "sample_id"]:
                if key in item["metainfo"]:
                    return str(item["metainfo"][key])

    # DataSample style
    if hasattr(item, "metainfo") and callable(getattr(item, "metainfo")):
        meta = item.metainfo()
        if isinstance(meta, dict):
            for key in ["sample_idx", "lidar_idx", "frame_id", "sample_id"]:
                if key in meta:
                    return str(meta[key])

    return f"{fallback_i:06d}"


# -----------------------
# Pred structure adapters
# -----------------------

Pred3D = Union[Dict[str, Any], Any]  # dict or InstanceData-like


def _pred_get(pred: Pred3D, key: str) -> Any:
    if isinstance(pred, dict):
        return pred.get(key, None)
    # InstanceData or similar
    if hasattr(pred, key):
        return getattr(pred, key)
    return None


def _pred_set(pred: Pred3D, key: str, value: Any) -> Pred3D:
    if isinstance(pred, dict):
        pred[key] = value
        return pred
    if hasattr(pred, key):
        setattr(pred, key, value)
        return pred
    # If it's some unsupported object, fall back to dict
    return {key: value}


def get_pred_instances_3d(item: Any) -> Pred3D:
    if isinstance(item, dict) and "pred_instances_3d" in item:
        return item["pred_instances_3d"]
    if hasattr(item, "pred_instances_3d"):
        return getattr(item, "pred_instances_3d")
    raise KeyError("Could not find pred_instances_3d in result item.")


def set_pred_instances_3d(item: Any, pred3d: Pred3D) -> Any:
    if isinstance(item, dict):
        item["pred_instances_3d"] = pred3d
        return item
    if hasattr(item, "pred_instances_3d"):
        setattr(item, "pred_instances_3d", pred3d)
        return item
    raise TypeError("Unsupported result item type for writing pred_instances_3d.")


def _ensure_boxes3d(x: Any, origin_fallback=(0.5, 0.5, 0.5)) -> LiDARInstance3DBoxes:
    if x is None:
        return LiDARInstance3DBoxes(torch.zeros((0, 7), dtype=torch.float32), box_dim=7, origin=origin_fallback)
    if isinstance(x, LiDARInstance3DBoxes):
        return x
    # sometimes it can be a tensor / ndarray Nx7
    arr = _to_numpy(x).astype(np.float32)
    if arr.size == 0:
        return LiDARInstance3DBoxes(torch.zeros((0, 7), dtype=torch.float32), box_dim=7, origin=origin_fallback)
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise ValueError(f"bboxes_3d must be Nx7-like, got shape {arr.shape}")
    arr7 = arr[:, :7]
    return LiDARInstance3DBoxes(torch.from_numpy(arr7), box_dim=7, origin=origin_fallback)


def _normalize_pred3d(pred3d: Pred3D) -> Dict[str, Any]:
    """
    Return a normalized dict with keys:
      - bboxes_3d: LiDARInstance3DBoxes
      - scores_3d: torch.FloatTensor [N]
      - labels_3d: torch.LongTensor [N]
    """
    boxes = _pred_get(pred3d, "bboxes_3d")
    scores = _pred_get(pred3d, "scores_3d")
    labels = _pred_get(pred3d, "labels_3d")

    origin = getattr(boxes, "origin", (0.5, 0.5, 0.5)) if boxes is not None else (0.5, 0.5, 0.5)
    boxes3d = _ensure_boxes3d(boxes, origin_fallback=origin)

    scores_t = _to_torch_1d(scores, dtype=torch.float32)
    labels_t = _to_torch_1d(labels, dtype=torch.long)

    # If lengths mismatch (rare but can happen with malformed dumps), clip to min
    n = min(len(boxes3d), int(scores_t.numel()), int(labels_t.numel()))
    if n != len(boxes3d) or n != scores_t.numel() or n != labels_t.numel():
        if n == 0:
            boxes3d = LiDARInstance3DBoxes(torch.zeros((0, 7), dtype=torch.float32), box_dim=7, origin=boxes3d.origin)
            scores_t = torch.zeros((0,), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)
        else:
            boxes3d = LiDARInstance3DBoxes(boxes3d.tensor[:n, :], box_dim=7, origin=boxes3d.origin)
            scores_t = scores_t[:n]
            labels_t = labels_t[:n]

    return {"bboxes_3d": boxes3d, "scores_3d": scores_t, "labels_3d": labels_t}


# -----------------------
# Geometry and fusion
# -----------------------

def transform_boxes_i2v(boxes: LiDARInstance3DBoxes, T_i2v: np.ndarray) -> LiDARInstance3DBoxes:
    """
    Transform LiDAR boxes from infra frame to vehicle frame.
    Boxes are (x,y,z,dx,dy,dz,yaw). Yaw is rotated by applying R to heading vector.
    """
    if boxes is None or len(boxes) == 0:
        return boxes

    t = boxes.tensor
    if not torch.is_tensor(t):
        t = torch.tensor(_to_numpy(t), dtype=torch.float32)

    xyz = t[:, 0:3].detach().cpu().numpy()
    dims = t[:, 3:6].detach().cpu().numpy()
    yaw = t[:, 6].detach().cpu().numpy()

    R = T_i2v[0:3, 0:3]
    p = T_i2v[0:3, 3]

    xyz_v = (R @ xyz.T).T + p[None, :]

    head_i = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1)
    head_v = (R @ head_i.T).T
    yaw_v = np.arctan2(head_v[:, 1], head_v[:, 0])

    out = np.concatenate([xyz_v, dims, yaw_v[:, None]], axis=1).astype(np.float32)
    out_t = torch.from_numpy(out)

    return LiDARInstance3DBoxes(out_t, box_dim=7, origin=getattr(boxes, "origin", (0.5, 0.5, 0.5)))


def euclidean_cost_xy(a_xyz: np.ndarray, b_xyz: np.ndarray) -> np.ndarray:
    if a_xyz.size == 0 or b_xyz.size == 0:
        return np.zeros((a_xyz.shape[0], b_xyz.shape[0]), dtype=np.float64)
    ax = a_xyz[:, 0:2][:, None, :]
    bx = b_xyz[:, 0:2][None, :, :]
    d = ax - bx
    return np.sqrt((d ** 2).sum(axis=2))


def merge_pair_weighted(box_a7: np.ndarray, score_a: float,
                        box_b7: np.ndarray, score_b: float,
                        score_mode: str) -> Tuple[np.ndarray, float]:
    wa = float(score_a)
    wb = float(score_b)
    wsum = max(wa + wb, 1e-6)

    pa = box_a7[0:3]
    pb = box_b7[0:3]
    p = (wa * pa + wb * pb) / wsum

    ya = float(box_a7[6])
    yb = float(box_b7[6])
    vx = wa * math.cos(ya) + wb * math.cos(yb)
    vy = wa * math.sin(ya) + wb * math.sin(yb)
    y = math.atan2(vy, vx)

    dims = box_a7[3:6] if score_a >= score_b else box_b7[3:6]
    out = np.array([p[0], p[1], p[2], dims[0], dims[1], dims[2], y], dtype=np.float32)

    if score_mode == "max":
        s = max(score_a, score_b)
    elif score_mode == "prob_union":
        # interpret scores as probabilities in [0,1]
        sa = float(np.clip(score_a, 0.0, 1.0))
        sb = float(np.clip(score_b, 0.0, 1.0))
        s = 1.0 - (1.0 - sa) * (1.0 - sb)
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    return out, float(s)


def late_fuse_one_sample(
    veh_pred3d: Dict[str, Any],
    inf_pred3d_v: Dict[str, Any],
    dist_thresh: float,
    per_class: bool,
    merge_mode: str,
    score_mode: str
) -> Dict[str, Any]:
    veh_boxes: LiDARInstance3DBoxes = veh_pred3d["bboxes_3d"]
    veh_scores_t: torch.Tensor = veh_pred3d["scores_3d"]
    veh_labels_t: torch.Tensor = veh_pred3d["labels_3d"]

    inf_boxes: LiDARInstance3DBoxes = inf_pred3d_v["bboxes_3d"]
    inf_scores_t: torch.Tensor = inf_pred3d_v["scores_3d"]
    inf_labels_t: torch.Tensor = inf_pred3d_v["labels_3d"]

    veh_scores = _to_numpy(veh_scores_t).astype(np.float32)
    veh_labels = _to_numpy(veh_labels_t).astype(np.int64)
    inf_scores = _to_numpy(inf_scores_t).astype(np.float32)
    inf_labels = _to_numpy(inf_labels_t).astype(np.int64)

    veh7 = _to_numpy(veh_boxes.tensor).astype(np.float32) if len(veh_boxes) else np.zeros((0, 7), np.float32)
    inf7 = _to_numpy(inf_boxes.tensor).astype(np.float32) if len(inf_boxes) else np.zeros((0, 7), np.float32)

    # Degenerate cases
    if veh7.shape[0] == 0 and inf7.shape[0] == 0:
        return {
            "bboxes_3d": LiDARInstance3DBoxes(torch.zeros((0, 7), dtype=torch.float32), box_dim=7, origin=getattr(veh_boxes, "origin", (0.5, 0.5, 0.5))),
            "scores_3d": torch.zeros((0,), dtype=torch.float32),
            "labels_3d": torch.zeros((0,), dtype=torch.long),
        }
    if veh7.shape[0] == 0:
        return {
            "bboxes_3d": inf_boxes,
            "scores_3d": inf_scores_t.to(dtype=torch.float32),
            "labels_3d": inf_labels_t.to(dtype=torch.long),
        }
    if inf7.shape[0] == 0:
        return {
            "bboxes_3d": veh_boxes,
            "scores_3d": veh_scores_t.to(dtype=torch.float32),
            "labels_3d": veh_labels_t.to(dtype=torch.long),
        }

    fused_boxes: List[np.ndarray] = []
    fused_scores: List[float] = []
    fused_labels: List[int] = []

    classes = np.unique(np.concatenate([veh_labels, inf_labels], axis=0)) if per_class else np.array([-1], dtype=np.int64)

    BIG = 1e6

    for cls in classes:
        if per_class:
            v_mask = (veh_labels == cls)
            i_mask = (inf_labels == cls)
        else:
            v_mask = np.ones_like(veh_labels, dtype=bool)
            i_mask = np.ones_like(inf_labels, dtype=bool)

        v_idx = np.where(v_mask)[0]
        i_idx = np.where(i_mask)[0]

        if v_idx.size == 0 and i_idx.size == 0:
            continue

        if v_idx.size == 0:
            for j in i_idx:
                fused_boxes.append(inf7[j])
                fused_scores.append(float(inf_scores[j]))
                fused_labels.append(int(inf_labels[j]) if per_class else int(inf_labels[j]))
            continue

        if i_idx.size == 0:
            for i in v_idx:
                fused_boxes.append(veh7[i])
                fused_scores.append(float(veh_scores[i]))
                fused_labels.append(int(veh_labels[i]) if per_class else int(veh_labels[i]))
            continue

        v_xyz = veh7[v_idx, 0:3]
        i_xyz = inf7[i_idx, 0:3]
        C = euclidean_cost_xy(v_xyz, i_xyz)

        C_gated = C.copy()
        C_gated[C_gated > dist_thresh] = BIG

        row_ind, col_ind = linear_sum_assignment(C_gated)

        matched_v = set()
        matched_i = set()

        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if C_gated[r, c] >= BIG:
                continue
            vi = int(v_idx[r])
            ij = int(i_idx[c])
            matched_v.add(vi)
            matched_i.add(ij)

            if merge_mode == "weighted":
                box, sc = merge_pair_weighted(veh7[vi], float(veh_scores[vi]),
                                              inf7[ij], float(inf_scores[ij]),
                                              score_mode=score_mode)
                fused_boxes.append(box)
                fused_scores.append(sc)
                fused_labels.append(int(cls) if per_class else int(veh_labels[vi]))
            elif merge_mode == "best_score":
                if veh_scores[vi] >= inf_scores[ij]:
                    fused_boxes.append(veh7[vi])
                    fused_scores.append(float(veh_scores[vi]))
                    fused_labels.append(int(cls) if per_class else int(veh_labels[vi]))
                else:
                    fused_boxes.append(inf7[ij])
                    fused_scores.append(float(inf_scores[ij]))
                    fused_labels.append(int(cls) if per_class else int(inf_labels[ij]))
            else:
                raise ValueError(f"Unknown merge_mode: {merge_mode}")

        # Append unmatched vehicle
        for vi in v_idx.tolist():
            if int(vi) in matched_v:
                continue
            fused_boxes.append(veh7[int(vi)])
            fused_scores.append(float(veh_scores[int(vi)]))
            fused_labels.append(int(cls) if per_class else int(veh_labels[int(vi)]))

        # Append unmatched infra
        for ij in i_idx.tolist():
            if int(ij) in matched_i:
                continue
            fused_boxes.append(inf7[int(ij)])
            fused_scores.append(float(inf_scores[int(ij)]))
            fused_labels.append(int(cls) if per_class else int(inf_labels[int(ij)]))

    out7 = np.stack(fused_boxes, axis=0).astype(np.float32) if fused_boxes else np.zeros((0, 7), np.float32)
    out_scores = np.array(fused_scores, dtype=np.float32) if fused_scores else np.zeros((0,), np.float32)
    out_labels = np.array(fused_labels, dtype=np.int64) if fused_labels else np.zeros((0,), np.int64)

    origin = getattr(veh_boxes, "origin", (0.5, 0.5, 0.5))
    out_boxes = LiDARInstance3DBoxes(torch.from_numpy(out7), box_dim=7, origin=origin)

    return {
        "bboxes_3d": out_boxes,
        "scores_3d": torch.from_numpy(out_scores),
        "labels_3d": torch.from_numpy(out_labels),
    }


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veh-pkl", required=True, type=str, help="Vehicle detector prediction PKL (MMEngine DumpResults output).")
    ap.add_argument("--inf-pkl", required=True, type=str, help="Infrastructure detector prediction PKL (MMEngine DumpResults output).")
    ap.add_argument("--out-pkl", required=True, type=str, help="Output fused PKL.")
    ap.add_argument("--i2v-json", default=None, type=str,
                    help="JSON mapping sample_idx -> 4x4 T_i2v (infra LiDAR -> vehicle LiDAR).")
    ap.add_argument("--i2v-dir", default=None, type=str,
                    help="Directory with per-sample transforms: <id>.txt or <id>.json (4x4 T_i2v).")
    ap.add_argument("--dist-thresh", default=2.0, type=float,
                    help="Euclidean XY distance threshold (meters) for matching.")
    ap.add_argument("--no-per-class", action="store_true",
                    help="If set, match across all classes together (not recommended).")
    ap.add_argument("--merge-mode", default="weighted", choices=["weighted", "best_score"],
                    help="How to merge matched pairs.")
    ap.add_argument("--score-mode", default="max", choices=["max", "prob_union"],
                    help="How to compute fused score when merge-mode=weighted.")
    args = ap.parse_args()

    veh_pkl = Path(args.veh_pkl)
    inf_pkl = Path(args.inf_pkl)
    out_pkl = Path(args.out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)

    i2v_map = load_i2v_from_json(Path(args.i2v_json)) if args.i2v_json else None
    i2v_dir = Path(args.i2v_dir) if args.i2v_dir else None

    veh_results = mmload(str(veh_pkl))
    inf_results = mmload(str(inf_pkl))

    if not isinstance(veh_results, list) or not isinstance(inf_results, list):
        raise TypeError("Expected both PKLs to contain a list of per-sample results.")

    if len(veh_results) != len(inf_results):
        raise ValueError(
            f"Vehicle and infra PKLs have different lengths: {len(veh_results)} vs {len(inf_results)}. "
            "They must be the same split in the same order."
        )

    fused_results = copy.deepcopy(veh_results)

    for i, (veh_item, inf_item) in enumerate(zip(veh_results, inf_results)):
        sample_idx = get_sample_idx(veh_item, i)

        veh_pred_raw = get_pred_instances_3d(veh_item)
        inf_pred_raw = get_pred_instances_3d(inf_item)

        veh_pred = _normalize_pred3d(veh_pred_raw)
        inf_pred = _normalize_pred3d(inf_pred_raw)

        T_i2v = load_i2v_for_sample(sample_idx, i2v_map, i2v_dir)

        # Transform infra boxes into vehicle frame
        inf_boxes_v = transform_boxes_i2v(inf_pred["bboxes_3d"], T_i2v)
        inf_pred_v = {
            "bboxes_3d": inf_boxes_v,
            "scores_3d": inf_pred["scores_3d"],
            "labels_3d": inf_pred["labels_3d"],
        }

        fused_pred = late_fuse_one_sample(
            veh_pred3d=veh_pred,
            inf_pred3d_v=inf_pred_v,
            dist_thresh=float(args.dist_thresh),
            per_class=(not args.no_per_class),
            merge_mode=args.merge_mode,
            score_mode=args.score_mode,
        )

        # Write fused pred back, keeping the vehicle entry structure intact
        fused_item = fused_results[i]
        set_pred_instances_3d(fused_item, fused_pred)

    mmdump(fused_results, str(out_pkl))
    print(f"[OK] Wrote fused predictions: {out_pkl}")


if __name__ == "__main__":
    main()
