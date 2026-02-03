#!/usr/bin/env python3
"""
tools/eval_lidar_ap_from_dump.py

Compute BEV + 3D AP11/AP40 in pure LiDAR frame.
No transforms, no calib, no KITTI camera conversion.

Also writes the printed metrics to a .txt file.

Output txt naming is inferred from the dump path:
- contains "late_fusion" / "lidar_late_fusion" -> late_fusion
- else contains "infra" / "infrastructure"    -> infra
- else contains "vehicle"                      -> vehicle
- else -> unknown
"""

import argparse
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np

# ==========================================================
# IN-CODE PARAM: where to save metrics txt files
# ==========================================================
METRICS_DIR = Path("/home/dellg16ssg/mmdetection3d/tools/metrics")
# ==========================================================


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


def infer_run_tag_from_path(dump_path: Path) -> str:
    s = dump_path.as_posix().lower()
    if "late_fusion" in s or "late-fusion" in s or "lidar_late_fusion" in s:
        return "late_fusion"
    if "/infra" in s or "_infra" in s or "infrastructure" in s:
        return "infra"
    if "/vehicle" in s or "_vehicle" in s:
        return "vehicle"
    return "unknown"


def eval_set(name: str, classes, samples, thr_map, device: str):
    print(f"\n=== {name} ===")
    lines = [f"=== {name} ==="]
    for cls_name in classes:
        cls_id = classes.index(cls_name)
        thr = float(thr_map[cls_name])

        rec_bev, prec_bev = match_class(samples, cls_id, thr, use_3d=False, device=device)
        rec_3d, prec_3d = match_class(samples, cls_id, thr, use_3d=True, device=device)

        if rec_bev is None:
            msg = f"{cls_name}: no GT"
            print(msg)
            lines.append(msg)
            continue

        bev_ap11 = 100.0 * ap_from_pr(rec_bev, prec_bev, 11)
        bev_ap40 = 100.0 * ap_from_pr(rec_bev, prec_bev, 40)
        d3_ap11 = 100.0 * ap_from_pr(rec_3d, prec_3d, 11)
        d3_ap40 = 100.0 * ap_from_pr(rec_3d, prec_3d, 40)

        msg = (
            f"{cls_name} IoU={thr:.2f} | "
            f"BEV AP11={bev_ap11:.4f} AP40={bev_ap40:.4f} | "
            f"3D AP11={d3_ap11:.4f} AP40={d3_ap40:.4f}"
        )
        print(msg)
        lines.append(msg)
    return lines


def parse_range_buckets(spec: str):
    if not spec:
        return []
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Bad range bucket: {part}")
        a, b = part.split("-", 1)
        out.append((float(a), float(b)))
    return out


def filter_samples_by_range(samples, r0: float, r1: float, axis: str = "radial"):
    filt = []
    for s in samples:
        gt = np.asarray(s["gt_boxes"], dtype=np.float32).reshape(-1, 7)
        pr = np.asarray(s["pred_boxes"], dtype=np.float32).reshape(-1, 7)

        if axis == "x":
            gt_d = gt[:, 0]
            pr_d = pr[:, 0]
        else:
            gt_d = np.sqrt(gt[:, 0] ** 2 + gt[:, 1] ** 2)
            pr_d = np.sqrt(pr[:, 0] ** 2 + pr[:, 1] ** 2)

        gt_keep = (gt_d >= r0) & (gt_d < r1)
        pr_keep = (pr_d >= r0) & (pr_d < r1)

        s2 = dict(s)
        s2["gt_boxes"] = gt[gt_keep]
        s2["gt_labels"] = s["gt_labels"][gt_keep]
        s2["pred_boxes"] = pr[pr_keep]
        s2["pred_scores"] = s["pred_scores"][pr_keep]
        s2["pred_labels"] = s["pred_labels"][pr_keep]
        filt.append(s2)
    return filt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_pkl", type=str)
    ap.add_argument("--device", type=str, default="cuda:0", help="Device for rotated IoU (mmcv op).")
    ap.add_argument("--classes", nargs="+", default=None, help="Subset of classes to eval, e.g., Car Pedestrian")
    ap.add_argument("--uniform-iou-thrs", default="", type=str,
                    help="Comma list of IoU thresholds to apply to ALL classes, e.g. '0.7,0.5,0.25'. "
                         "If set, overrides STRICT/LOOSE mixed IoUs.")
    ap.add_argument("--range-buckets", default="", type=str,
                    help="Comma list like '0-30,30-50,50-100'.")
    ap.add_argument("--range-axis", default="radial", choices=["radial", "x"],
                    help="Distance axis for range buckets.")
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
    h(f"[INFO] timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    h(f"[INFO] num_samples: {len(samples)}")
    h(f"[INFO] classes: {classes}")

    fcfg = dump.get("fov_filter", None)
    if isinstance(fcfg, dict):
        h("[INFO] Dump contains FOV filter settings:")
        for k in sorted(fcfg.keys()):
            h(f"  - {k}: {fcfg[k]}")

    thr_strict = {"Pedestrian": 0.50, "Cyclist": 0.50, "Car": 0.70}
    thr_loose  = {"Pedestrian": 0.25, "Cyclist": 0.25, "Car": 0.50}
    for c in classes:
        if c not in thr_strict:
            thr_strict[c] = 0.50
        if c not in thr_loose:
            thr_loose[c] = 0.25

    uniform_thrs: List[float] = []
    if args.uniform_iou_thrs:
        for part in str(args.uniform_iou_thrs).split(","):
            part = part.strip()
            if not part:
                continue
            uniform_thrs.append(float(part))

    buckets = parse_range_buckets(args.range_buckets)
    strict_lines = []
    loose_lines = []
    if uniform_thrs:
        for thr in uniform_thrs:
            thr_map = {c: float(thr) for c in classes}
            if not buckets:
                strict_lines += eval_set(f"IOU={thr:.2f}", classes, samples, thr_map, device=args.device)
                strict_lines.append("")
            else:
                for r0, r1 in buckets:
                    bucket_samples = filter_samples_by_range(samples, r0, r1, axis=args.range_axis)
                    strict_lines += eval_set(f"IOU={thr:.2f} [{r0:.0f}-{r1:.0f}]", classes, bucket_samples, thr_map, device=args.device)
                    strict_lines.append("")
    else:
        if not buckets:
            strict_lines = eval_set("STRICT", classes, samples, thr_strict, device=args.device)
            loose_lines = eval_set("LOOSE", classes, samples, thr_loose, device=args.device)
        else:
            for r0, r1 in buckets:
                bucket_samples = filter_samples_by_range(samples, r0, r1, axis=args.range_axis)
                strict_lines += eval_set(f"STRICT [{r0:.0f}-{r1:.0f}]", classes, bucket_samples, thr_strict, device=args.device)
                strict_lines.append("")
                loose_lines += eval_set(f"LOOSE [{r0:.0f}-{r1:.0f}]", classes, bucket_samples, thr_loose, device=args.device)
                loose_lines.append("")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    dump_stem = dump_path.with_suffix("").name
    out_txt = METRICS_DIR / f"{dump_stem}_eval_{run_tag}.txt"

    out_txt.write_text("\n".join(header + [""] + strict_lines + ([""] + loose_lines if loose_lines else [])) + "\n")
    print(f"\n[INFO] Wrote metrics txt: {out_txt}")


if __name__ == "__main__":
    main()
