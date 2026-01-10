#!/usr/bin/env python3
"""
tools/dump_pred_gt_lidar.py

Dump predictions + GT boxes in LiDAR frame for the ENTIRE split in cfg.test_dataloader.dataset.

No transforms. No KITTI camera conversion. No calib.

Output is a pickle with:
  {
    "classes": [...],
    "samples": [
      {
        "sample_id": "000002",
        "pred_boxes": (N,7) float32 [x,y,z,dx,dy,dz,yaw]
        "pred_scores": (N,) float32
        "pred_labels": (N,) int64 in [0..C-1]
        "gt_boxes": (M,7) float32
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
from typing import Any, Dict, Optional

import numpy as np
import torch

from mmengine.config import Config
from mmengine.registry import DATASETS
from mmengine.dataset import pseudo_collate

from mmdet3d.apis import init_model
from mmdet3d.utils import register_all_modules


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
    # fallback
    if hasattr(dataset, "data_list") and isinstance(dataset.data_list, list):
        if 0 <= idx < len(dataset.data_list):
            return dataset.data_list[idx] if isinstance(dataset.data_list[idx], dict) else {}
    if hasattr(dataset, "infos") and isinstance(dataset.infos, list):
        if 0 <= idx < len(dataset.infos):
            return dataset.infos[idx] if isinstance(dataset.infos[idx], dict) else {}
    return {}


def to_np(x) -> np.ndarray:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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

    # label key name varies across versions
    gt_labels = ea.get("gt_labels_3d", None)
    if gt_labels is None:
        gt_labels = ea.get("gt_labels", None)
    if gt_labels is None:
        return gt_boxes, np.zeros((gt_boxes.shape[0],), dtype=np.int64)

    gt_labels = to_np(gt_labels).astype(np.int64).reshape(-1)
    return gt_boxes, gt_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", type=str)
    ap.add_argument("ckpt", type=str)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--min-score", type=float, default=0.0, help="Optional speed filter. Default keeps all preds.")
    ap.add_argument("--max-samples", type=int, default=-1, help="Debug: dump only first K samples.")
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

        with torch.no_grad():
            outputs = model.test_step(data_batch)
        pred_sample = outputs[0]

        pb, ps, pl = extract_pred(pred_sample, min_score=float(args.min_score))
        gb, gl = extract_gt_from_eval_ann_info(info)

        samples_out.append(
            dict(
                sample_id=sid,
                pred_boxes=pb,
                pred_scores=ps,
                pred_labels=pl,
                gt_boxes=gb,
                gt_labels=gl,
            )
        )

        if (idx + 1) % 50 == 0 or idx == n - 1:
            print(f"[INFO] dumped {idx+1}/{n}")

    out = dict(classes=class_names, samples=samples_out)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(out_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[INFO] Wrote dump: {out_path}")


if __name__ == "__main__":
    main()
