#!/usr/bin/env python3
"""
tools/eval_lidar_ap_dairv2x_style.py

DAIR-V2X style AP evaluation that reads existing pkl dumps from dump_pred_gt_dairv2x_v2.py.

Key differences from eval_lidar_ap_from_dump.py:
1. Uses 8-corner convex hull IoU (not axis-aligned BEV IoU)
2. Uses continuous AP (not KITTI AP11/AP40)
3. Uses DAIR-V2X IoU thresholds: Car [0.3, 0.5, 0.7], Pedestrian [0.25, 0.5]

Output txt naming follows the same convention as eval_lidar_ap_from_dump.py.
"""

import argparse
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.spatial import ConvexHull

# ==========================================================
# IN-CODE PARAM: where to save metrics txt files
# ==========================================================
METRICS_DIR = Path("/home/dellg16ssg/mmdetection3d/tools/metrics")
# ==========================================================

# DAIR-V2X IoU thresholds per class
IOU_THRESHOLDS = {
    'Car': [0.3, 0.5, 0.7],
    'Pedestrian': [0.25, 0.5],
    'Cyclist': [0.25, 0.5],
}


# ============================================================
# Box7 to 8-corner conversion
# ============================================================

def box7_to_corners(box7: np.ndarray) -> np.ndarray:
    """
    Convert box7 (x, y, z, dx, dy, dz, yaw) to 8-corner representation.

    Returns: (N, 8, 3) array of corner points

    Corner order:
    - Corners 0-3: bottom face (z = center_z - dz/2), counter-clockwise
    - Corners 4-7: top face (z = center_z + dz/2), counter-clockwise
    """
    box7 = np.asarray(box7, dtype=np.float64).reshape(-1, 7)
    n = box7.shape[0]
    if n == 0:
        return np.zeros((0, 8, 3), dtype=np.float64)

    corners = np.zeros((n, 8, 3), dtype=np.float64)

    for i in range(n):
        x, y, z, dx, dy, dz, yaw = box7[i]

        # Half dimensions
        hdx, hdy, hdz = dx / 2, dy / 2, dz / 2

        # Rotation matrix (around z-axis)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # Local corners (before rotation)
        # Bottom face (z=-hdz): 0-3 in counter-clockwise order
        # Top face (z=+hdz): 4-7 in counter-clockwise order
        local = np.array([
            [-hdx, -hdy, -hdz],  # 0: back-left-bottom
            [ hdx, -hdy, -hdz],  # 1: front-left-bottom
            [ hdx,  hdy, -hdz],  # 2: front-right-bottom
            [-hdx,  hdy, -hdz],  # 3: back-right-bottom
            [-hdx, -hdy,  hdz],  # 4: back-left-top
            [ hdx, -hdy,  hdz],  # 5: front-left-top
            [ hdx,  hdy,  hdz],  # 6: front-right-top
            [-hdx,  hdy,  hdz],  # 7: back-right-top
        ])

        # Rotate and translate
        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])

        rotated = (R @ local.T).T
        corners[i] = rotated + np.array([x, y, z])

    return corners


# ============================================================
# 8-corner convex hull IoU (DAIR-V2X style)
# ============================================================

def polygon_clip(subjectPolygon, clipPolygon):
    """Sutherland-Hodgman polygon clipping."""
    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        denom = dc[0] * dp[1] - dc[1] * dp[0]
        if abs(denom) < 1e-10:
            return s  # Return current point if parallel
        n3 = 1.0 / denom
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = list(subjectPolygon)
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        if len(inputList) == 0:
            return None
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList


def poly_area(x, y):
    """Compute polygon area using shoelace formula."""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convex_hull_intersection(p1, p2):
    """Compute intersection area of two convex polygons."""
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None and len(inter_p) >= 3:
        try:
            hull_inter = ConvexHull(inter_p)
            return inter_p, hull_inter.volume
        except Exception:
            return None, 0.0
    return None, 0.0


def box3d_vol(corners):
    """Compute volume of 3D box from 8 corners."""
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def box3d_iou_corners(corners1: np.ndarray, corners2: np.ndarray) -> Tuple[float, float]:
    """
    Compute 3D and BEV IoU between two boxes given their 8 corners.

    Args:
        corners1: (8, 3) array
        corners2: (8, 3) array

    Returns:
        iou_3d, iou_bev
    """
    # BEV rectangle (first 4 corners projected to xy plane)
    rect1 = [(corners1[i, 0], corners1[i, 1]) for i in range(4)]
    rect2 = [(corners2[i, 0], corners2[i, 1]) for i in range(4)]

    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])

    inter, inter_area = convex_hull_intersection(rect1, rect2)

    if area1 + area2 - inter_area <= 0:
        iou_2d = 0.0
    else:
        iou_2d = inter_area / (area1 + area2 - inter_area)

    # 3D IoU
    zmax = min(corners1[4, 2], corners2[4, 2])
    zmin = max(corners1[0, 2], corners2[0, 2])

    inter_vol = inter_area * max(0.0, zmax - zmin)

    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)

    if vol1 + vol2 - inter_vol <= 0:
        iou_3d = 0.0
    else:
        iou_3d = inter_vol / (vol1 + vol2 - inter_vol)

    return iou_3d, iou_2d


# ============================================================
# DAIR-V2X style continuous AP
# ============================================================

def compute_ap_continuous(pred_results: List[Dict], num_gt: int) -> float:
    """
    Compute continuous AP (DAIR-V2X style).

    Args:
        pred_results: list of {'score': float, 'type': 'tp'/'fp'}
        num_gt: number of ground truth boxes

    Returns:
        AP value (0-1)
    """
    if num_gt == 0:
        return 0.0

    if len(pred_results) == 0:
        return 0.0

    # Sort by score descending
    pred_results = sorted(pred_results, key=lambda x: -x['score'])

    num_tp = np.zeros(len(pred_results))
    for i in range(len(pred_results)):
        num_tp[i] = (0 if i == 0 else num_tp[i - 1])
        if pred_results[i]['type'] == 'tp':
            num_tp[i] += 1

    precision = num_tp / np.arange(1, len(pred_results) + 1)
    recall = num_tp / num_gt

    # Smooth precision (take max from right)
    for i in range(len(pred_results) - 1, 0, -1):
        precision[i - 1] = max(precision[i], precision[i - 1])

    # Compute AP as area under PR curve
    index = np.where(recall[1:] != recall[:-1])[0]
    if len(index) == 0:
        return 0.0

    ap = np.sum((recall[index + 1] - recall[index]) * precision[index + 1])
    return float(ap)


def match_class_dairv2x(samples, cls_id: int, iou_thr: float, use_3d: bool) -> Tuple[List[Dict], int]:
    """
    DAIR-V2X style TP/FP matching using 8-corner convex hull IoU.

    Returns:
        pred_results: list of {'score': float, 'type': 'tp'/'fp'}
        total_gt: number of GT boxes
    """
    all_preds = []
    all_gts = []
    total_gt = 0

    for si, s in enumerate(samples):
        gt_mask = (s["gt_labels"] == cls_id)
        gt_boxes = s["gt_boxes"][gt_mask].astype(np.float32)
        all_gts.append({
            'boxes': gt_boxes,
            'corners': box7_to_corners(gt_boxes),
            'matched': np.zeros((gt_boxes.shape[0],), dtype=bool)
        })
        total_gt += gt_boxes.shape[0]

        pred_mask = (s["pred_labels"] == cls_id)
        pb = s["pred_boxes"][pred_mask].astype(np.float32)
        ps = s["pred_scores"][pred_mask].astype(np.float32)
        pc = box7_to_corners(pb)

        for k in range(pb.shape[0]):
            all_preds.append({
                'score': float(ps[k]),
                'si': si,
                'corners': pc[k]
            })

    # Sort predictions by score descending
    all_preds.sort(key=lambda x: -x['score'])

    if total_gt == 0:
        return [], 0

    pred_results = []

    for pred in all_preds:
        result = {'score': pred['score'], 'type': 'fp'}
        gt = all_gts[pred['si']]

        if gt['boxes'].shape[0] == 0:
            pred_results.append(result)
            continue

        unmatched = ~gt['matched']
        if not np.any(unmatched):
            pred_results.append(result)
            continue

        # Find best matching GT using 8-corner IoU
        best_iou = iou_thr
        best_j = -1

        for j in range(gt['boxes'].shape[0]):
            if gt['matched'][j]:
                continue

            try:
                iou_3d, iou_bev = box3d_iou_corners(gt['corners'][j], pred['corners'])
                iou = iou_3d if use_3d else iou_bev
            except Exception:
                iou = 0.0

            if iou >= best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0:
            result['type'] = 'tp'
            gt['matched'][best_j] = True

        pred_results.append(result)

    return pred_results, total_gt


def infer_run_tag_from_path(dump_path: Path) -> str:
    s = dump_path.as_posix().lower()
    if "late_fusion" in s or "late-fusion" in s or "lidar_late_fusion" in s:
        return "late_fusion"
    if "/infra" in s or "_infra" in s or "infrastructure" in s:
        return "infra"
    if "/vehicle" in s or "_vehicle" in s:
        return "vehicle"
    return "unknown"


def eval_set_dairv2x(name: str, classes: List[str], samples, device: str = None):
    """
    DAIR-V2X style evaluation with multiple IoU thresholds per class.

    Returns list of output lines for txt file.
    """
    print(f"\n=== {name} (DAIR-V2X Style) ===")
    lines = [f"=== {name} (DAIR-V2X Style) ==="]

    for cls_name in classes:
        cls_id = classes.index(cls_name)
        thresholds = IOU_THRESHOLDS.get(cls_name, [0.5])

        for iou_thr in thresholds:
            # BEV evaluation
            pred_results_bev, total_gt = match_class_dairv2x(samples, cls_id, iou_thr, use_3d=False)
            ap_bev = 100.0 * compute_ap_continuous(pred_results_bev, total_gt) if total_gt > 0 else 0.0

            # 3D evaluation
            pred_results_3d, total_gt = match_class_dairv2x(samples, cls_id, iou_thr, use_3d=True)
            ap_3d = 100.0 * compute_ap_continuous(pred_results_3d, total_gt) if total_gt > 0 else 0.0

            if total_gt == 0:
                msg = f"{cls_name} IoU={iou_thr:.2f}: no GT"
            else:
                msg = f"{cls_name} IoU={iou_thr:.2f} | BEV AP={ap_bev:.4f} | 3D AP={ap_3d:.4f}"

            print(msg)
            lines.append(msg)

    return lines


def main():
    ap = argparse.ArgumentParser(description="DAIR-V2X style AP evaluation from pkl dump")
    ap.add_argument("dump_pkl", type=str, help="Path to dump pkl file from dump_pred_gt_dairv2x_v2.py")
    ap.add_argument("--classes", nargs="+", default=None, help="Subset of classes to eval, e.g., Car Pedestrian")
    args = ap.parse_args()

    dump_path = Path(args.dump_pkl)
    dump = pickle.load(open(dump_path, "rb"))
    classes = list(dump["classes"])

    if args.classes:
        wanted = [str(c) for c in args.classes]
        classes = [c for c in classes if c in wanted]
        if not classes:
            raise ValueError(f"No matching classes in dump. dump classes={dump['classes']}, wanted={wanted}")

    samples = dump["samples"]

    run_tag = infer_run_tag_from_path(dump_path)

    header = []
    def h(msg):
        print(msg)
        header.append(msg)

    h(f"[INFO] dump_pkl: {dump_path}")
    h(f"[INFO] run_tag: {run_tag}")
    h(f"[INFO] eval_method: DAIR-V2X style (8-corner IoU, continuous AP)")
    h(f"[INFO] timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    h(f"[INFO] num_samples: {len(samples)}")
    h(f"[INFO] classes: {classes}")
    h(f"[INFO] IoU thresholds: {IOU_THRESHOLDS}")

    # Run evaluation
    eval_lines = eval_set_dairv2x("DAIR-V2X Style Evaluation", classes, samples)

    # Write output txt
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    dump_stem = dump_path.with_suffix("").name
    out_txt = METRICS_DIR / f"{dump_stem}_eval_{run_tag}_dairv2x_style.txt"

    out_txt.write_text("\n".join(header + [""] + eval_lines) + "\n")
    print(f"\n[INFO] Wrote metrics txt: {out_txt}")


if __name__ == "__main__":
    main()
