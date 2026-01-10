#!/usr/bin/env python3
"""
tools/eval_lidar_ap_from_dump.py

Compute BEV + 3D AP11/AP40 in pure LiDAR frame.
No transforms, no calib, no KITTI camera conversion.

Requires mmcv.ops.box_iou_rotated.

Usage:
  python3 tools/eval_lidar_ap_from_dump.py dump.pkl
"""

import argparse
import pickle
import numpy as np


def ap_from_pr(rec, prec, npoints: int) -> float:
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    if npoints == 11:
        rs = np.linspace(0.0, 1.0, 11)
    elif npoints == 40:
        rs = np.linspace(0.0, 1.0, 40)
    else:
        raise ValueError("npoints must be 11 or 40")

    ap = 0.0
    for r in rs:
        ap += (mpre[mrec >= r].max() if np.any(mrec >= r) else 0.0)
    return ap / float(len(rs))


def boxes7_to_bev5(boxes7: np.ndarray) -> np.ndarray:
    # (x,y,z,dx,dy,dz,yaw) -> (cx,cy,w,h,angle)
    out = np.zeros((boxes7.shape[0], 5), dtype=np.float32)
    out[:, 0] = boxes7[:, 0]
    out[:, 1] = boxes7[:, 1]
    out[:, 2] = boxes7[:, 3]
    out[:, 3] = boxes7[:, 4]
    out[:, 4] = boxes7[:, 6]
    return out


def bev_iou_rotated_mmcv(a5: np.ndarray, b5: np.ndarray, device: str = "cuda:0") -> np.ndarray:
    try:
        import torch
        from mmcv.ops import box_iou_rotated
    except Exception as e:
        raise RuntimeError("mmcv.ops.box_iou_rotated not available in this env.") from e

    if a5.shape[0] == 0 or b5.shape[0] == 0:
        return np.zeros((a5.shape[0], b5.shape[0]), dtype=np.float32)

    a = torch.from_numpy(a5).float().to(device)
    b = torch.from_numpy(b5).float().to(device)
    iou = box_iou_rotated(a, b)  # (Na, Nb)
    return iou.detach().cpu().numpy().astype(np.float32)


def iou3d_from_bev_and_height(a7: np.ndarray, b7: np.ndarray, bev_iou: np.ndarray) -> np.ndarray:
    # upright 3D: IoU3D = inter_area * inter_h / union_vol
    a_area = (a7[:, 3] * a7[:, 4]).astype(np.float32)[:, None]  # (Na,1)
    b_area = (b7[:, 3] * b7[:, 4]).astype(np.float32)[None, :]  # (1,Nb)

    # inter_area = iou*(A+B)/(1+iou)
    inter_area = np.where(
        bev_iou > 0,
        bev_iou * (a_area + b_area) / (1.0 + bev_iou),
        0.0
    ).astype(np.float32)

    a_z0 = (a7[:, 2] - 0.5 * a7[:, 5]).astype(np.float32)[:, None]
    a_z1 = (a7[:, 2] + 0.5 * a7[:, 5]).astype(np.float32)[:, None]
    b_z0 = (b7[:, 2] - 0.5 * b7[:, 5]).astype(np.float32)[None, :]
    b_z1 = (b7[:, 2] + 0.5 * b7[:, 5]).astype(np.float32)[None, :]

    inter_h = np.maximum(0.0, np.minimum(a_z1, b_z1) - np.maximum(a_z0, b_z0)).astype(np.float32)
    inter_vol = inter_area * inter_h

    a_vol = (a7[:, 3] * a7[:, 4] * a7[:, 5]).astype(np.float32)[:, None]
    b_vol = (b7[:, 3] * b7[:, 4] * b7[:, 5]).astype(np.float32)[None, :]
    union = a_vol + b_vol - inter_vol

    return np.where(union > 0, inter_vol / union, 0.0).astype(np.float32)


def match_class(samples, cls_id: int, iou_thr: float, use_3d: bool, device: str):
    # collect all preds globally, sorted by score desc
    preds = []
    gts = []
    total_gt = 0

    for si, s in enumerate(samples):
        gt_mask = (s["gt_labels"] == cls_id)
        gt_boxes = s["gt_boxes"][gt_mask].astype(np.float32)
        gts.append(dict(boxes=gt_boxes, matched=np.zeros((gt_boxes.shape[0],), dtype=bool)))
        total_gt += gt_boxes.shape[0]

        pred_mask = (s["pred_labels"] == cls_id)
        pb = s["pred_boxes"][pred_mask].astype(np.float32)
        ps = s["pred_scores"][pred_mask].astype(np.float32)
        for k in range(pb.shape[0]):
            preds.append((float(ps[k]), si, pb[k]))

    preds.sort(key=lambda x: -x[0])

    if total_gt == 0:
        return None, None

    tp = np.zeros((len(preds),), dtype=np.float32)
    fp = np.zeros((len(preds),), dtype=np.float32)

    for i, (score, si, pbox) in enumerate(preds):
        gt = gts[si]
        gt_boxes = gt["boxes"]
        if gt_boxes.shape[0] == 0:
            fp[i] = 1.0
            continue

        unmatched = ~gt["matched"]
        if not np.any(unmatched):
            fp[i] = 1.0
            continue

        pb7 = np.asarray(pbox, dtype=np.float32).reshape(1, 7)
        gb7 = gt_boxes

        pb5 = boxes7_to_bev5(pb7)
        gb5 = boxes7_to_bev5(gb7)

        bev_iou = bev_iou_rotated_mmcv(pb5, gb5, device=device)  # (1,M)

        if use_3d:
            iou = iou3d_from_bev_and_height(pb7, gb7, bev_iou)[0]
        else:
            iou = bev_iou[0]

        iou_sel = iou.copy()
        iou_sel[~unmatched] = -1.0
        j = int(np.argmax(iou_sel))

        if iou_sel[j] >= float(iou_thr):
            tp[i] = 1.0
            gt["matched"][j] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / float(total_gt)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return rec, prec


def eval_set(name: str, classes, samples, thr_map, device: str):
    print(f"\n=== {name} ===")
    for cls_name in classes:
        cls_id = classes.index(cls_name)
        thr = float(thr_map[cls_name])

        rec_bev, prec_bev = match_class(samples, cls_id, thr, use_3d=False, device=device)
        rec_3d, prec_3d = match_class(samples, cls_id, thr, use_3d=True, device=device)

        if rec_bev is None:
            print(f"{cls_name}: no GT")
            continue

        bev_ap11 = 100.0 * ap_from_pr(rec_bev, prec_bev, 11)
        bev_ap40 = 100.0 * ap_from_pr(rec_bev, prec_bev, 40)
        d3_ap11 = 100.0 * ap_from_pr(rec_3d, prec_3d, 11)
        d3_ap40 = 100.0 * ap_from_pr(rec_3d, prec_3d, 40)

        print(
            f"{cls_name} IoU={thr:.2f} | "
            f"BEV AP11={bev_ap11:.4f} AP40={bev_ap40:.4f} | "
            f"3D AP11={d3_ap11:.4f} AP40={d3_ap40:.4f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_pkl", type=str)
    ap.add_argument("--device", type=str, default="cuda:0", help="Device for rotated IoU (mmcv op).")
    args = ap.parse_args()

    dump = pickle.load(open(args.dump_pkl, "rb"))
    classes = list(dump["classes"])
    samples = dump["samples"]

    # KITTI-like strict/loose thresholds (same as what MMDet3D prints in your log)
    thr_strict = {"Pedestrian": 0.50, "Cyclist": 0.50, "Car": 0.70}
    thr_loose  = {"Pedestrian": 0.25, "Cyclist": 0.25, "Car": 0.50}

    # If your classes list differs, default to 0.5
    for c in classes:
        if c not in thr_strict:
            thr_strict[c] = 0.50
        if c not in thr_loose:
            thr_loose[c] = 0.25

    eval_set("STRICT", classes, samples, thr_strict, device=args.device)
    eval_set("LOOSE", classes, samples, thr_loose, device=args.device)


if __name__ == "__main__":
    main()
