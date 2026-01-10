#!/usr/bin/env python3
"""
tools/vis_dump_pkl_open3d.py

Visualize ONLY what is in dump_val_lidar.pkl produced by dump_pred_gt_lidar.py.
No dataset loading, no calib, no transforms, no pointcloud file access.

Shows:
  - GT boxes (green)
  - Pred boxes (red), optionally filtered by --score-thr

Browse:
  N: next sample
  B: prev sample
  S: save screenshot (if --out-dir set)
  H: help

Usage:
  python3 tools/vis_dump_pkl_open3d.py work_dirs/pp_infra_synth_3class/dump_val_lidar.pkl \
    --score-thr 0.1 --out-dir /tmp/vis_dump
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception as e:
    raise RuntimeError("Open3D is required. Try: pip install open3d") from e

import pickle


# -------------------------
# Geometry helpers
# -------------------------

def make_corners_from_box7(box7: np.ndarray) -> np.ndarray:
    """
    box7: [x, y, z, dx, dy, dz, yaw] in LiDAR frame, yaw around +Z.
    Returns (8,3) corners.
    """
    x, y, z, dx, dy, dz, yaw = [float(v) for v in box7]
    hx, hy, hz = dx * 0.5, dy * 0.5, dz * 0.5

    # 8 corners in the box local frame
    # bottom (z - hz), then top (z + hz)
    local = np.array(
        [
            [-hx, -hy, -hz],
            [ hx, -hy, -hz],
            [ hx,  hy, -hz],
            [-hx,  hy, -hz],
            [-hx, -hy,  hz],
            [ hx, -hy,  hz],
            [ hx,  hy,  hz],
            [-hx,  hy,  hz],
        ],
        dtype=np.float64,
    )

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    corners = (local @ R.T) + np.array([x, y, z], dtype=np.float64)
    return corners


def corners_to_lineset(corners_8x3: np.ndarray, color_rgb: Tuple[float, float, float]) -> o3d.geometry.LineSet:
    corners = np.asarray(corners_8x3, dtype=np.float64).reshape(8, 3)
    lines = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # top
            [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
        ],
        dtype=np.int32,
    )
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color_rgb, dtype=np.float64), (lines.shape[0], 1)))
    return ls


def print_help():
    print("Keys:")
    print("  N : next sample (wrap)")
    print("  B : previous sample (wrap)")
    print("  S : save screenshot to --out-dir")
    print("  H : print help + current status")
    print("  ESC / close window: exit")


# -------------------------
# Dump loading
# -------------------------

def load_dump(path: Path) -> Dict:
    with open(path, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict) or "samples" not in d:
        raise RuntimeError("Unexpected dump format: expected dict with key 'samples'.")
    return d


def build_geoms_for_sample(
    sample: Dict,
    score_thr: float,
    show_gt: bool,
    show_pred: bool,
    max_preds: int,
) -> Tuple[List[o3d.geometry.Geometry], Dict]:
    geoms: List[o3d.geometry.Geometry] = []
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0]))

    gt_boxes = sample.get("gt_boxes", None)
    gt_labels = sample.get("gt_labels", None)
    pred_boxes = sample.get("pred_boxes", None)
    pred_scores = sample.get("pred_scores", None)
    pred_labels = sample.get("pred_labels", None)

    if gt_boxes is None:
        gt_boxes = np.zeros((0, 7), dtype=np.float32)
    if pred_boxes is None:
        pred_boxes = np.zeros((0, 7), dtype=np.float32)
    if pred_scores is None:
        pred_scores = np.zeros((pred_boxes.shape[0],), dtype=np.float32)

    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 7)
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 7)
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)

    # Filter preds by score and optionally cap count
    keep = pred_scores >= float(score_thr)
    pb = pred_boxes[keep]
    ps = pred_scores[keep]

    if max_preds > 0 and pb.shape[0] > max_preds:
        order = np.argsort(-ps)
        order = order[:max_preds]
        pb = pb[order]
        ps = ps[order]

    if show_gt and gt_boxes.shape[0] > 0:
        for k in range(gt_boxes.shape[0]):
            corners = make_corners_from_box7(gt_boxes[k])
            geoms.append(corners_to_lineset(corners, (0.2, 1.0, 0.2)))

    if show_pred and pb.shape[0] > 0:
        for k in range(pb.shape[0]):
            corners = make_corners_from_box7(pb[k])
            geoms.append(corners_to_lineset(corners, (1.0, 0.2, 0.2)))

    dbg = dict(
        sample_id=str(sample.get("sample_id", "")),
        gt_count=int(gt_boxes.shape[0]),
        pred_total=int(pred_boxes.shape[0]),
        pred_kept=int(pb.shape[0]),
        score_thr=float(score_thr),
    )
    return geoms, dbg


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_pkl", type=str)
    ap.add_argument("--score-thr", type=float, default=0.0)
    ap.add_argument("--point-size", type=float, default=2.0)  # not used (no pointcloud), kept for consistency
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--start", type=int, default=0, help="Start at dataset index (0-based).")
    ap.add_argument("--show-gt", action="store_true", help="Show GT boxes.")
    ap.add_argument("--show-pred", action="store_true", help="Show predicted boxes.")
    ap.add_argument("--max-preds", type=int, default=0, help="Cap drawn preds per sample (0 = no cap).")
    args = ap.parse_args()

    dump_path = Path(args.dump_pkl)
    d = load_dump(dump_path)
    samples = d["samples"]
    n = len(samples)

    if n == 0:
        raise RuntimeError("Dump has zero samples.")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    show_gt = bool(args.show_gt) if (args.show_gt or args.show_pred) else True
    show_pred = bool(args.show_pred) if (args.show_gt or args.show_pred) else True

    state = {
        "idx": int(args.start) % n,
        "n": n,
        "dbg": None,
    }

    print(f"[INFO] Loaded dump: {dump_path} | samples={n} | classes={d.get('classes', None)}")
    print_help()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Dump PKL viewer (boxes only)", width=1600, height=900, visible=True)

    def redraw(i: int, save: bool = False):
        i = int(i) % n
        state["idx"] = i
        sample = samples[i]

        geoms, dbg = build_geoms_for_sample(
            sample=sample,
            score_thr=float(args.score_thr),
            show_gt=show_gt,
            show_pred=show_pred,
            max_preds=int(args.max_preds),
        )
        state["dbg"] = dbg

        vis.clear_geometries()
        for g in geoms:
            vis.add_geometry(g)

        vis.poll_events()
        vis.update_renderer()

        print(
            f"[INFO] idx={i}/{n-1} sid={dbg['sample_id']} | "
            f"GT={dbg['gt_count']} | Pred kept={dbg['pred_kept']}/{dbg['pred_total']} | score_thr={dbg['score_thr']}"
        )

        if out_dir and save:
            sid = dbg["sample_id"] if dbg["sample_id"] else f"{i:06d}"
            png = out_dir / f"{sid}_boxes_only.png"
            vis.capture_screen_image(str(png), do_render=True)
            print(f"[INFO] Wrote screenshot: {png}")

    def cb_next(v):
        redraw(state["idx"] + 1, save=False)
        return False

    def cb_prev(v):
        redraw(state["idx"] - 1, save=False)
        return False

    def cb_save(v):
        if not out_dir:
            print("[WARN] No --out-dir set; cannot save screenshot.")
            return False
        redraw(state["idx"], save=True)
        return False

    def cb_help(v):
        print_help()
        dbg = state.get("dbg", None)
        if isinstance(dbg, dict):
            print(f"[INFO] current idx={state['idx']} sid={dbg['sample_id']}")
        return False

    vis.register_key_callback(ord("N"), cb_next)
    vis.register_key_callback(ord("B"), cb_prev)
    vis.register_key_callback(ord("S"), cb_save)
    vis.register_key_callback(ord("H"), cb_help)

    redraw(state["idx"], save=False)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
