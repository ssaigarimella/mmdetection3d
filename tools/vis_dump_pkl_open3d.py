#!/usr/bin/env python3
"""
tools/vis_dump_pkl_open3d.py

Visualize ONLY what is in dump_val_lidar.pkl produced by dump_pred_gt_lidar.py.
No dataset loading, no calib, no transforms, no pointcloud file access.

Modes:
- 2D BEV (Matplotlib): draw BEV rectangles (orthogonal projection onto XY)
- 3D (Open3D): draw 3D cuboids with THICK edges (cylinders), since OpenGL line width
  is often clamped to 1px in Open3D's classic visualizer on Linux.

Browse (both modes):
  N: next sample
  B: prev sample
  S: save screenshot (if --out-dir set)
  H: help
  Q / ESC: quit
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pickle


# ============================================================
# IN-CODE SETTINGS (no CLI args)
# ============================================================

# If True, use Matplotlib for a true 2D BEV plot.
# If False, use Open3D 3D cuboids viewer.
USE_MATPLOTLIB_BEV_2D = False

# -------- BEV plot toggles (Matplotlib only) --------
BEV_SHOW_GRID = False
BEV_SHOW_AXES = False          # draw vehicle-frame X/Y arrows
BEV_SHOW_LABELS = False        # title + axis labels + X/Y text
BEV_SHOW_TICKS = False         # tick marks + tick labels

# BEV plot cosmetics
BEV_DARK_GT_COLOR = (0.0, 0.35, 0.0)   # dark green
BEV_DARK_PR_COLOR = (0.55, 0.0, 0.0)   # dark red
BEV_LINEWIDTH_GT = 2.5
BEV_LINEWIDTH_PR = 2.5
BEV_AXES_LINEWIDTH = 2.0

# Vehicle frame axes length (meters) to draw in BEV
BEV_AXIS_LEN_M = 8.0

# Auto-zoom margin around boxes in BEV
BEV_MARGIN_M = 5.0

# If you want a fixed BEV window instead of autoscaling, set to a number (half-range).
# Example: 40.0 gives xlim/ylim roughly centered at origin: [-40, 40].
BEV_FIXED_HALF_RANGE: Optional[float] = None

# -------- Open3D 3D cosmetics --------
# Thick "line" edges are drawn as cylinders with this radius (meters).
# Increase to make box edges thicker in 3D.
O3D_EDGE_RADIUS = 0.1         # try 0.02 .. 0.10
O3D_EDGE_RESOLUTION = 12       # cylinder quality; lower if too slow

# Coordinate frame size in Open3D
O3D_COORD_FRAME_SIZE = 2.0


# ============================================================
# Helpers: parsing + box geometry
# ============================================================

def load_dump(path: Path) -> Dict:
    with open(path, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict) or "samples" not in d:
        raise RuntimeError("Unexpected dump format: expected dict with key 'samples'.")
    return d


def bev_rect_corners_xy(box7: np.ndarray) -> np.ndarray:
    """
    Orthogonal BEV projection of a 3D box onto XY plane: rotated rectangle.

    box7: [x, y, z, dx, dy, dz, yaw] in LiDAR/vehicle frame
    Returns (4,2) corners in order around the rectangle.
    """
    x, y, _, dx, dy, _, yaw = [float(v) for v in box7]
    hx, hy = 0.5 * dx, 0.5 * dy

    local = np.array(
        [
            [-hx, -hy],
            [ hx, -hy],
            [ hx,  hy],
            [-hx,  hy],
        ],
        dtype=np.float64,
    )

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float64)

    world = (local @ R.T) + np.array([x, y], dtype=np.float64)
    return world


def filter_and_cap_preds(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    score_thr: float,
    max_preds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if pred_boxes is None:
        pred_boxes = np.zeros((0, 7), dtype=np.float64)
    if pred_scores is None:
        pred_scores = np.zeros((pred_boxes.shape[0],), dtype=np.float64)

    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 7)
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)

    keep = pred_scores >= float(score_thr)
    pb = pred_boxes[keep]
    ps = pred_scores[keep]

    if max_preds > 0 and pb.shape[0] > max_preds:
        order = np.argsort(-ps)[:max_preds]
        pb = pb[order]
        ps = ps[order]

    return pb, ps


def print_help():
    print("Keys:")
    print("  N : next sample (wrap)")
    print("  B : previous sample (wrap)")
    print("  S : save screenshot to --out-dir")
    print("  H : print help + current status")
    print("  Q / ESC : quit")


# ============================================================
# Matplotlib BEV viewer
# ============================================================

def run_matplotlib_bev_viewer(
    samples: List[Dict],
    dump_path: Path,
    classes,
    score_thr: float,
    out_dir: Optional[Path],
    start_idx: int,
    show_gt: bool,
    show_pred: bool,
    max_preds: int,
) -> None:
    import matplotlib.pyplot as plt

    n = len(samples)
    idx = int(start_idx) % n

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.canvas.manager.set_window_title("Dump PKL viewer (Matplotlib BEV 2D)")

    state = {"idx": idx}

    def _axes_limits_from_boxes(gt7: np.ndarray, pb7: np.ndarray) -> Tuple[float, float, float, float]:
        if BEV_FIXED_HALF_RANGE is not None:
            r = float(BEV_FIXED_HALF_RANGE)
            return (-r, r, -r, r)

        pts = []
        if gt7 is not None and gt7.size > 0:
            gt7 = np.asarray(gt7, dtype=np.float64).reshape(-1, 7)
            for k in range(gt7.shape[0]):
                pts.append(bev_rect_corners_xy(gt7[k]))
        if pb7 is not None and pb7.size > 0:
            pb7 = np.asarray(pb7, dtype=np.float64).reshape(-1, 7)
            for k in range(pb7.shape[0]):
                pts.append(bev_rect_corners_xy(pb7[k]))

        if not pts:
            r = 20.0
            return (-r, r, -r, r)

        P = np.concatenate(pts, axis=0)  # (N,2)
        xmin, ymin = P.min(axis=0)
        xmax, ymax = P.max(axis=0)
        xmin -= BEV_MARGIN_M
        xmax += BEV_MARGIN_M
        ymin -= BEV_MARGIN_M
        ymax += BEV_MARGIN_M

        # keep square view
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        half = 0.5 * max((xmax - xmin), (ymax - ymin))
        return (cx - half, cx + half, cy - half, cy + half)

    def _draw_rect(corners_xy: np.ndarray, color_rgb: Tuple[float, float, float], lw: float) -> None:
        c = np.asarray(corners_xy, dtype=np.float64).reshape(4, 2)
        closed = np.vstack([c, c[0:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=color_rgb, linewidth=lw)

    def _draw_vehicle_axes() -> None:
        L = float(BEV_AXIS_LEN_M)
        ax.plot([0.0, L], [0.0, 0.0], color=(0.0, 0.0, 0.0), linewidth=BEV_AXES_LINEWIDTH)
        if BEV_SHOW_LABELS:
            ax.text(L, 0.0, "X", fontsize=12, ha="left", va="center")
        ax.plot([0.0, 0.0], [0.0, L], color=(0.0, 0.0, 0.0), linewidth=BEV_AXES_LINEWIDTH)
        if BEV_SHOW_LABELS:
            ax.text(0.0, L, "Y", fontsize=12, ha="center", va="bottom")
        ax.scatter([0.0], [0.0], color=(0.0, 0.0, 0.0), s=20)

    def _apply_axis_visibility():
        ax.set_xticks([] if not BEV_SHOW_TICKS else ax.get_xticks())
        ax.set_yticks([] if not BEV_SHOW_TICKS else ax.get_yticks())
        ax.tick_params(
            bottom=BEV_SHOW_TICKS,
            left=BEV_SHOW_TICKS,
            labelbottom=BEV_SHOW_TICKS,
            labelleft=BEV_SHOW_TICKS,
        )

        if not BEV_SHOW_LABELS:
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")

        if (not BEV_SHOW_TICKS) and (not BEV_SHOW_LABELS):
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            for spine in ax.spines.values():
                spine.set_visible(True)

    def redraw(save: bool = False) -> None:
        i = state["idx"]
        sample = samples[i]

        gt_boxes = sample.get("gt_boxes", None)
        pred_boxes = sample.get("pred_boxes", None)
        pred_scores = sample.get("pred_scores", None)

        if gt_boxes is None:
            gt_boxes = np.zeros((0, 7), dtype=np.float64)
        gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 7)

        pb, _ps = filter_and_cap_preds(pred_boxes, pred_scores, score_thr, max_preds)

        ax.clear()

        if BEV_SHOW_GRID:
            ax.grid(True, linewidth=0.6)
        else:
            ax.grid(False)

        if BEV_SHOW_AXES:
            _draw_vehicle_axes()

        if show_gt and gt_boxes.shape[0] > 0:
            for k in range(gt_boxes.shape[0]):
                _draw_rect(bev_rect_corners_xy(gt_boxes[k]), BEV_DARK_GT_COLOR, BEV_LINEWIDTH_GT)

        if show_pred and pb.shape[0] > 0:
            for k in range(pb.shape[0]):
                _draw_rect(bev_rect_corners_xy(pb[k]), BEV_DARK_PR_COLOR, BEV_LINEWIDTH_PR)

        xmin, xmax, ymin, ymax = _axes_limits_from_boxes(gt_boxes, pb)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")

        sid = str(sample.get("sample_id", ""))

        if BEV_SHOW_LABELS:
            pred_total = int(np.asarray(pred_boxes).shape[0]) if pred_boxes is not None else 0
            ax.set_title(
                f"BEV 2D | idx={i}/{n-1} sid={sid} | GT={gt_boxes.shape[0]} | "
                f"Pred kept={pb.shape[0]}/{pred_total} | score_thr={score_thr}",
                fontsize=12,
            )
            ax.set_xlabel("X (vehicle frame)")
            ax.set_ylabel("Y (vehicle frame)")

        _apply_axis_visibility()

        fig.canvas.draw_idle()

        if out_dir and save:
            sid_out = sid if sid else f"{i:06d}"
            png = out_dir / f"{sid_out}_bev2d.png"
            fig.savefig(str(png), dpi=200, bbox_inches="tight", pad_inches=0.02)
            print(f"[INFO] Wrote screenshot: {png}")

        pred_total = int(np.asarray(pred_boxes).shape[0]) if pred_boxes is not None else 0
        print(
            f"[INFO] idx={i}/{n-1} sid={sid} | GT={gt_boxes.shape[0]} | "
            f"Pred kept={pb.shape[0]}/{pred_total} | score_thr={score_thr} | mode=BEV-2D"
        )

    def on_key(event):
        k = (event.key or "").lower()
        if k == "n":
            state["idx"] = (state["idx"] + 1) % n
            redraw(save=False)
        elif k == "b":
            state["idx"] = (state["idx"] - 1) % n
            redraw(save=False)
        elif k == "s":
            if not out_dir:
                print("[WARN] No --out-dir set; cannot save screenshot.")
            else:
                redraw(save=True)
        elif k == "h":
            print_help()
            print(
                f"[INFO] current idx={state['idx']} | grid={BEV_SHOW_GRID} axes={BEV_SHOW_AXES} "
                f"labels={BEV_SHOW_LABELS} ticks={BEV_SHOW_TICKS}"
            )
        elif k in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    print(f"[INFO] Loaded dump: {dump_path} | samples={n} | classes={classes}")
    print_help()
    redraw(save=False)
    plt.show()


# ============================================================
# Open3D 3D viewer (thick edges via cylinders)
# ============================================================

def run_open3d_3d_viewer(
    samples: List[Dict],
    dump_path: Path,
    classes,
    score_thr: float,
    out_dir: Optional[Path],
    start_idx: int,
    show_gt: bool,
    show_pred: bool,
    max_preds: int,
) -> None:
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError("Open3D is required for 3D mode. Try: pip install open3d") from e

    def make_corners_from_box7_3d(box7: np.ndarray) -> np.ndarray:
        x, y, z, dx, dy, dz, yaw = [float(v) for v in box7]
        hx, hy, hz = dx * 0.5, dy * 0.5, dz * 0.5
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
        return (local @ R.T) + np.array([x, y, z], dtype=np.float64)

    def _rotation_matrix_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64).reshape(3)
        b = np.asarray(b, dtype=np.float64).reshape(3)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return np.eye(3, dtype=np.float64)

        a = a / na
        b = b / nb

        v = np.cross(a, b)
        c = float(np.dot(a, b))
        s = float(np.linalg.norm(v))

        if s < 1e-12:
            if c > 0.0:
                return np.eye(3, dtype=np.float64)
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(a[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = axis - a * float(np.dot(axis, a))
            axis /= (np.linalg.norm(axis) + 1e-12)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]], dtype=np.float64)
            return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)

        axis = v / s
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float64)
        R = np.eye(3, dtype=np.float64) + K * s + (K @ K) * (1.0 - c)
        return R

    def _cylinder_between(
        p0: np.ndarray,
        p1: np.ndarray,
        radius: float,
        color_rgb: Tuple[float, float, float],
    ) -> "o3d.geometry.TriangleMesh":
        p0 = np.asarray(p0, dtype=np.float64).reshape(3)
        p1 = np.asarray(p1, dtype=np.float64).reshape(3)
        v = p1 - p0
        L = float(np.linalg.norm(v))
        if L < 1e-9:
            m = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=O3D_EDGE_RESOLUTION)
            m.compute_vertex_normals()
            m.paint_uniform_color(color_rgb)
            m.translate(p0)
            return m

        cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius,
            height=L,
            resolution=O3D_EDGE_RESOLUTION,
            split=1,
        )
        cyl.compute_vertex_normals()
        cyl.paint_uniform_color(color_rgb)

        cyl.translate(-cyl.get_center())

        z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        R = _rotation_matrix_from_a_to_b(z, v)
        cyl.rotate(R, center=(0.0, 0.0, 0.0))

        mid = 0.5 * (p0 + p1)
        cyl.translate(mid)
        return cyl

    def corners_to_thick_edges_mesh(
        corners_8x3: np.ndarray,
        color_rgb: Tuple[float, float, float],
        radius: float,
    ) -> "o3d.geometry.TriangleMesh":
        corners = np.asarray(corners_8x3, dtype=np.float64).reshape(8, 3)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        m = o3d.geometry.TriangleMesh()
        for i0, i1 in edges:
            m += _cylinder_between(corners[i0], corners[i1], radius, color_rgb)
        return m

    n = len(samples)
    if n == 0:
        raise RuntimeError("Dump has zero samples.")

    state = {"idx": int(start_idx) % n}

    print(f"[INFO] Loaded dump: {dump_path} | samples={n} | classes={classes}")
    print_help()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Dump PKL viewer (Open3D 3D)", width=1600, height=900, visible=True)

    def redraw(i: int, save: bool = False):
        i = int(i) % n
        state["idx"] = i
        sample = samples[i]

        gt_boxes = sample.get("gt_boxes", None)
        pred_boxes = sample.get("pred_boxes", None)
        pred_scores = sample.get("pred_scores", None)

        if gt_boxes is None:
            gt_boxes = np.zeros((0, 7), dtype=np.float64)
        gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 7)

        pb, _ps = filter_and_cap_preds(pred_boxes, pred_scores, score_thr, max_preds)

        geoms: List["o3d.geometry.Geometry"] = []
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(O3D_COORD_FRAME_SIZE), origin=[0, 0, 0]))

        if show_gt and gt_boxes.shape[0] > 0:
            for k in range(gt_boxes.shape[0]):
                geoms.append(
                    corners_to_thick_edges_mesh(
                        make_corners_from_box7_3d(gt_boxes[k]),
                        (0.2, 1.0, 0.2),
                        radius=float(O3D_EDGE_RADIUS),
                    )
                )

        if show_pred and pb.shape[0] > 0:
            for k in range(pb.shape[0]):
                geoms.append(
                    corners_to_thick_edges_mesh(
                        make_corners_from_box7_3d(pb[k]),
                        (1.0, 0.2, 0.2),
                        radius=float(O3D_EDGE_RADIUS),
                    )
                )

        vis.clear_geometries()
        for g in geoms:
            vis.add_geometry(g)

        vis.poll_events()
        vis.update_renderer()

        sid = str(sample.get("sample_id", ""))
        pred_total = int(np.asarray(pred_boxes).shape[0]) if pred_boxes is not None else 0
        print(
            f"[INFO] idx={i}/{n-1} sid={sid} | GT={gt_boxes.shape[0]} | "
            f"Pred kept={pb.shape[0]}/{pred_total} | score_thr={score_thr} | mode=3D"
        )

        if out_dir and save:
            sid_out = sid if sid else f"{i:06d}"
            png = out_dir / f"{sid_out}_3d.png"
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
        print(f"[INFO] current idx={state['idx']} | edge_radius={O3D_EDGE_RADIUS}")
        return False

    def cb_quit(v):
        vis.close()
        return False

    vis.register_key_callback(ord("N"), cb_next)
    vis.register_key_callback(ord("B"), cb_prev)
    vis.register_key_callback(ord("S"), cb_save)
    vis.register_key_callback(ord("H"), cb_help)
    vis.register_key_callback(ord("Q"), cb_quit)
    vis.register_key_callback(256, cb_quit)  # ESC (best-effort; may vary by backend)

    redraw(state["idx"], save=False)
    vis.run()
    vis.destroy_window()


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_pkl", type=str)
    ap.add_argument("--score-thr", type=float, default=0.0)
    ap.add_argument("--point-size", type=float, default=2.0)  # unused in both modes, kept for consistency
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--start", type=int, default=0, help="Start at dataset index (0-based).")
    ap.add_argument("--show-gt", action="store_true", help="Show GT boxes.")
    ap.add_argument("--show-pred", action="store_true", help="Show predicted boxes.")
    ap.add_argument("--max-preds", type=int, default=0, help="Cap drawn preds per sample (0 = no cap).")
    args = ap.parse_args()

    dump_path = Path(args.dump_pkl)
    d = load_dump(dump_path)
    samples = d["samples"]
    if len(samples) == 0:
        raise RuntimeError("Dump has zero samples.")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    show_gt = bool(args.show_gt) if (args.show_gt or args.show_pred) else True
    show_pred = bool(args.show_pred) if (args.show_gt or args.show_pred) else True

    classes = d.get("classes", None)

    if USE_MATPLOTLIB_BEV_2D:
        run_matplotlib_bev_viewer(
            samples=samples,
            dump_path=dump_path,
            classes=classes,
            score_thr=float(args.score_thr),
            out_dir=out_dir,
            start_idx=int(args.start),
            show_gt=show_gt,
            show_pred=show_pred,
            max_preds=int(args.max_preds),
        )
    else:
        run_open3d_3d_viewer(
            samples=samples,
            dump_path=dump_path,
            classes=classes,
            score_thr=float(args.score_thr),
            out_dir=out_dir,
            start_idx=int(args.start),
            show_gt=show_gt,
            show_pred=show_pred,
            max_preds=int(args.max_preds),
        )


if __name__ == "__main__":
    main()
