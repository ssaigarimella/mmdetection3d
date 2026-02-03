#!/usr/bin/env python3
"""
tools/vis_dump_pkl_open3d.py

Visualize ONLY what is in dump_val_lidar.pkl produced by dump_pred_gt_lidar.py.
No dataset loading, no calib, no transforms, no pointcloud file access.
This one includes both the GT boxes and the predicted boxes.

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
BEV_DARK_GT_COLOR = (0.0, 0.35, 0.0)   # dark green: GROUND TRUTH
BEV_DARK_PR_COLOR = (0.55, 0.0, 0.0)   # dark red: PREDICTION
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
O3D_EDGE_RADIUS = 0.1          # thickness in meters
O3D_EDGE_RESOLUTION = 12       # cylinder quality; lower if too slow
O3D_COORD_FRAME_SIZE = 4.0

# -------- Open3D camera focus (THIS controls how zoomed-in it is) --------
O3D_AUTO_FOCUS_ON_REDRAW = True

# Padding added around the box AABB before framing (meters). Smaller = tighter.
O3D_FOCUS_PADDING_M = 0.25

# Distance from camera to lookat, relative to the box AABB diagonal.
# Smaller => more zoomed-in. Try 0.35 .. 0.80
O3D_CAM_DIST_SCALE = 0.45

# Hard minimum distance so the camera doesn't go inside the box cluster (meters)
O3D_CAM_DIST_MIN_M = 3.0

# Camera orientation (in your LiDAR frame)
O3D_CAM_UP = (0.0, 0.0, 1.0)
# "front" is direction from camera toward the lookat point.
O3D_CAM_FRONT = (1.0, 0.0, -0.35)


# ============================================================
# Helpers: parsing + box geometry
# ============================================================

def load_dump(path: Path) -> Dict:
    with open(path, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict) or "samples" not in d:
        raise RuntimeError("Unexpected dump format: expected dict with key 'samples'.")
    return d


def _parse_class_list(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _resolve_class_ids(names: List[str], classes: Optional[List[str]]) -> List[int]:
    if not names:
        return []
    ids: List[int] = []
    if classes is None:
        # Allow numeric ids only if class list is unknown.
        for n in names:
            if n.isdigit():
                ids.append(int(n))
            else:
                raise RuntimeError(f"Class name '{n}' provided but dump has no class list.")
        return ids

    name_to_id = {str(c).lower(): i for i, c in enumerate(classes)}
    for n in names:
        key = str(n).lower()
        if key.isdigit():
            ids.append(int(key))
        elif key in name_to_id:
            ids.append(int(name_to_id[key]))
        else:
            raise RuntimeError(f"Unknown class '{n}'. Available: {classes}")
    return ids


def _sample_has_any_gt_class(sample: Dict, class_ids: List[int]) -> bool:
    if not class_ids:
        return True
    labels = sample.get("gt_labels", sample.get("gt_labels_3d", None))
    if labels is None:
        return False
    arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return False
    return bool(np.isin(arr, np.asarray(class_ids, dtype=np.int64)).any())


def bev_rect_corners_xy(box7: np.ndarray) -> np.ndarray:
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
    print("  Q / ESC: quit")


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

        P = np.concatenate(pts, axis=0)
        xmin, ymin = P.min(axis=0)
        xmax, ymax = P.max(axis=0)
        xmin -= BEV_MARGIN_M
        xmax += BEV_MARGIN_M
        ymin -= BEV_MARGIN_M
        ymax += BEV_MARGIN_M

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
        ax.grid(bool(BEV_SHOW_GRID), linewidth=0.6)

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
            ax.set_title(f"BEV 2D | idx={i}/{n-1} sid={sid}", fontsize=12)
            ax.set_xlabel("X (vehicle frame)")
            ax.set_ylabel("Y (vehicle frame)")

        _apply_axis_visibility()
        fig.canvas.draw_idle()

        if out_dir and save:
            sid_out = sid if sid else f"{i:06d}"
            png = out_dir / f"{sid_out}_bev2d.png"
            fig.savefig(str(png), dpi=200, bbox_inches="tight", pad_inches=0.02)
            print(f"[INFO] Wrote screenshot: {png}")

        # MOD: print only GT and prediction counts for this frame (from PKL content for the frame)
        gt_count = int(gt_boxes.shape[0])
        pred_count = int(pb.shape[0]) if show_pred else 0
        print(f"[COUNT] idx={i}/{n-1} sid={sid} | GT={gt_count} | Pred={pred_count}")

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
# Open3D 3D viewer (thick edges via cylinders) + deterministic camera distance
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

    def _aabb_of_geoms(geoms: List["o3d.geometry.Geometry"]) -> Optional["o3d.geometry.AxisAlignedBoundingBox"]:
        aabb = None
        for g in geoms:
            try:
                bb = g.get_axis_aligned_bounding_box()
                aabb = bb if aabb is None else (aabb + bb)
            except Exception:
                continue
        return aabb

    def _normalize(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return np.array([1.0, 0.0, -0.35], dtype=np.float64)
        return v / n

    def _lookat_extrinsic(eye: np.ndarray, lookat: np.ndarray, up: np.ndarray) -> np.ndarray:
        """
        Return Open3D pinhole extrinsic (world -> camera) using a standard look-at.
        """
        eye = np.asarray(eye, dtype=np.float64).reshape(3)
        lookat = np.asarray(lookat, dtype=np.float64).reshape(3)
        up = np.asarray(up, dtype=np.float64).reshape(3)

        f = lookat - eye
        f = _normalize(f)
        upn = _normalize(up)

        s = np.cross(f, upn)
        s = _normalize(s)
        u = np.cross(s, f)  # already normalized

        M = np.eye(4, dtype=np.float64)
        M[0, 0:3] = s
        M[1, 0:3] = u
        M[2, 0:3] = -f
        M[0, 3] = -float(np.dot(s, eye))
        M[1, 3] = -float(np.dot(u, eye))
        M[2, 3] = float(np.dot(f, eye))
        return M

    def _focus_camera_on_boxes(
        vis: "o3d.visualization.VisualizerWithKeyCallback",
        box_geoms: List["o3d.geometry.Geometry"],
        fallback_geoms: List["o3d.geometry.Geometry"],
    ) -> None:
        vc = vis.get_view_control()
        if vc is None:
            return

        aabb = _aabb_of_geoms(box_geoms) if len(box_geoms) > 0 else _aabb_of_geoms(fallback_geoms)
        if aabb is None:
            return

        minb = np.asarray(aabb.min_bound, dtype=np.float64)
        maxb = np.asarray(aabb.max_bound, dtype=np.float64)

        pad = float(O3D_FOCUS_PADDING_M)
        minb = minb - pad
        maxb = maxb + pad

        center = 0.5 * (minb + maxb)
        extent = (maxb - minb)
        diag = float(np.linalg.norm(extent))
        if diag < 1e-6:
            diag = 10.0

        front = _normalize(np.array(O3D_CAM_FRONT, dtype=np.float64))
        up = _normalize(np.array(O3D_CAM_UP, dtype=np.float64))

        dist = max(float(O3D_CAM_DIST_MIN_M), float(O3D_CAM_DIST_SCALE) * diag)
        eye = center - front * dist

        try:
            pin = vc.convert_to_pinhole_camera_parameters()
            pin.extrinsic = _lookat_extrinsic(eye, center, up)
            try:
                vc.convert_from_pinhole_camera_parameters(pin, allow_arbitrary=True)
            except TypeError:
                vc.convert_from_pinhole_camera_parameters(pin)
        except Exception:
            try:
                vc.set_lookat(center.tolist())
                vc.set_front(front.tolist())
                vc.set_up(up.tolist())
            except Exception:
                pass

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

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(O3D_COORD_FRAME_SIZE), origin=[0, 0, 0])

        box_geoms: List["o3d.geometry.Geometry"] = []
        if show_gt and gt_boxes.shape[0] > 0:
            for k in range(gt_boxes.shape[0]):
                box_geoms.append(
                    corners_to_thick_edges_mesh(
                        make_corners_from_box7_3d(gt_boxes[k]),
                        (0.2, 1.0, 0.2),
                        radius=float(O3D_EDGE_RADIUS),
                    )
                )

        if show_pred and pb.shape[0] > 0:
            for k in range(pb.shape[0]):
                box_geoms.append(
                    corners_to_thick_edges_mesh(
                        make_corners_from_box7_3d(pb[k]),
                        (1.0, 0.2, 0.2),
                        radius=float(O3D_EDGE_RADIUS),
                    )
                )

        geoms: List["o3d.geometry.Geometry"] = [coord] + box_geoms

        vis.clear_geometries()
        for gi, g in enumerate(geoms):
            try:
                vis.add_geometry(g, reset_bounding_box=(gi == 0))
            except TypeError:
                vis.add_geometry(g)

        vis.poll_events()
        vis.update_renderer()

        try:
            vis.reset_view_point(True)
        except Exception:
            pass

        if O3D_AUTO_FOCUS_ON_REDRAW:
            _focus_camera_on_boxes(vis, box_geoms=box_geoms, fallback_geoms=geoms)

        vis.poll_events()
        vis.update_renderer()

        sid = str(sample.get("sample_id", ""))

        # MOD: print only GT and prediction counts for this frame (from PKL content for the frame)
        gt_count = int(gt_boxes.shape[0])
        pred_count = int(pb.shape[0]) if show_pred else 0
        print(f"[COUNT] idx={i}/{n-1} sid={sid} | GT={gt_count} | Pred={pred_count}")

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
        print(
            f"[INFO] current idx={state['idx']} | edge_radius={O3D_EDGE_RADIUS} | "
            f"pad={O3D_FOCUS_PADDING_M} | dist_scale={O3D_CAM_DIST_SCALE} | dist_min={O3D_CAM_DIST_MIN_M}"
        )
        return False

    def cb_quit(v):
        try:
            vis.close()
        except Exception:
            pass
        return False

    vis.register_key_callback(ord("N"), cb_next)
    vis.register_key_callback(ord("B"), cb_prev)
    vis.register_key_callback(ord("S"), cb_save)
    vis.register_key_callback(ord("H"), cb_help)
    vis.register_key_callback(ord("Q"), cb_quit)

    try:
        vis.register_key_callback(256, cb_quit)  # ESC best-effort
    except Exception:
        pass

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
    ap.add_argument(
        "--only-gt-class",
        type=str,
        default=None,
        help="Comma-separated class names (or ids) to filter samples by GT labels (e.g., Pedestrian).",
    )
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

    only_gt_names = _parse_class_list(args.only_gt_class)
    only_gt_ids = _resolve_class_ids(only_gt_names, classes)
    if only_gt_ids:
        before_n = len(samples)
        samples = [s for s in samples if _sample_has_any_gt_class(s, only_gt_ids)]
        after_n = len(samples)
        if after_n == 0:
            raise RuntimeError(f"No samples matched --only-gt-class={args.only_gt_class}")
        print(f"[INFO] Filtered samples by GT class {only_gt_names} -> {after_n}/{before_n}")

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
