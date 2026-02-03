#!/usr/bin/env python3
"""
tools/v2x_vis_open3d_sanity_dairv2x.py

Open3D sanity viewer for DAIR-V2X dataset (vehicle + infrastructure).
This version is specifically for the official DAIR-V2X dataset format.

What it visualizes (in VEHICLE LiDAR frame):
  - vehicle LiDAR points (red)
  - infra LiDAR points transformed into vehicle LiDAR (blue)
  - vehicle native LiDAR GT boxes (green)
  - infra native virtuallidar GT boxes transformed into vehicle LiDAR (yellow)

Key differences from synthetic version:
  - Uses .pcd files instead of .bin
  - Reads data_info.json for transforms
  - Handles DAIR-V2X label structure
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d is not installed in this environment. Try: pip install open3d")
    sys.exit(1)

# ============================================================
# USER SETTINGS (edit these variables only)
# ============================================================

# Root of DAIR-V2X dataset (should contain vehicle-side/, infrastructure-side/, cooperative/)
DATA_ROOT = Path(
    "/home/dellg16ssg/multi-robot-coordination/collaborative-perception-BEVP/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure"
)

SAMPLE_ID = "000001"  # DAIR-V2X sample ID

# Open3D visualization tuning
AXIS_SIZE_M = 2.0
VOXEL_DOWNSAMPLE_M = 0.10  # set to 0.0 to disable
MAX_POINTS_PER_CLOUD = 200_000  # set to 0 to disable
O3D_POINT_SIZE = 2.0

# Which GT source to use for 3D boxes
SHOW_NATIVE_LIDAR_GT = True

# Filter classes to show (set to None to show all)
SHOW_CLASSES = None  # e.g. {"Car", "Pedestrian"}

# Point cloud colors (RGB in [0,1])
VEH_COLOR = (1.0, 0.2, 0.2)
INF_COLOR = (0.2, 0.6, 1.0)

# Box colors
VEH_BOX_COLOR_NATIVE = (0.2, 1.0, 0.2)     # green
INF_BOX_COLOR_NATIVE = (1.0, 0.85, 0.2)    # yellow

# ============================================================
# Helpers: point clouds
# ============================================================

def load_pcd_xyzi(path: Path) -> np.ndarray:
    """Load DAIR-V2X .pcd file and return Nx3 xyz points."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing pcd: {path}")

    try:
        # Try pypcd first (more accurate for structured PCDs)
        from pypcd import pypcd
        pc = pypcd.PointCloud.from_path(str(path))
        arr = pc.pc_data
        x = arr["x"].astype(np.float32)
        y = arr["y"].astype(np.float32)
        z = arr["z"].astype(np.float32)
        pts = np.stack([x, y, z], axis=1)
        pts = pts[np.isfinite(pts).all(axis=1)]
        return pts
    except Exception:
        # Fallback to open3d
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float32)
        pts = pts[np.isfinite(pts).all(axis=1)]
        return pts

def maybe_limit_points(pts: np.ndarray, max_points: int) -> np.ndarray:
    if max_points is None or max_points <= 0:
        return pts
    if pts.shape[0] <= max_points:
        return pts
    idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
    return pts[idx]

def o3d_cloud(pts_xyz: np.ndarray, color_rgb) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_xyz.astype(np.float64))
    pcd.paint_uniform_color(color_rgb)
    return pcd

def maybe_downsample(pcd: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    if voxel is None or voxel <= 0:
        return pcd
    return pcd.voxel_down_sample(voxel_size=float(voxel))

# ============================================================
# Helpers: rigid transforms
# ============================================================

def _as_np(x):
    return np.array(x, dtype=np.float64)

def get_rot_trans(d: Dict):
    """Extract rotation and translation from DAIR-V2X calibration dict."""
    if "rotation" in d and "translation" in d:
        return d["rotation"], d["translation"]
    if "transform" in d and isinstance(d["transform"], dict):
        t = d["transform"]
        if "rotation" in t and "translation" in t:
            return t["rotation"], t["translation"]
    if "rotation_matrix" in d and "translation_vector" in d:
        return d["rotation_matrix"], d["translation_vector"]
    raise KeyError("rotation/translation not found in calibration json")

def make_T(rot, trans) -> np.ndarray:
    """Build 4x4 transform matrix from rotation and translation."""
    R = np.array(rot, dtype=np.float64).reshape(3, 3)
    t = np.array(trans, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def apply_T(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    N = pts_xyz.shape[0]
    pts_h = np.ones((N, 4), dtype=np.float64)
    pts_h[:, :3] = pts_xyz.astype(np.float64)
    out = (T @ pts_h.T).T
    return out[:, :3]

def load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

# ============================================================
# Helpers: native LiDAR GT labels
# ============================================================

def _as_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.number)):
            return float(x)
        if isinstance(x, str):
            return float(x.strip())
        if isinstance(x, dict):
            for k in ("value", "val", "data", "v"):
                if k in x:
                    return _as_float(x[k], default)
    except Exception:
        return default
    return default

def load_native_lidar_labels(data_root: Path, side: str, sample_id: str, z_is_bottom: bool = True) -> List[Dict]:
    """
    Load DAIR-V2X native LiDAR labels.

    Args:
        data_root: Path to cooperative-vehicle-infrastructure/
        side: "vehicle-side" or "infrastructure-side"
        sample_id: Sample ID (e.g., "000001")
        z_is_bottom: If True, treat z as bottom of box; else as center

    Returns:
        List of dicts with keys: type, center, hwl, yaw
    """
    if side == "vehicle-side":
        label_path = data_root / side / "label" / "lidar" / f"{sample_id}.json"
    elif side == "infrastructure-side":
        label_path = data_root / side / "label" / "virtuallidar" / f"{sample_id}.json"
    else:
        raise ValueError("side must be 'vehicle-side' or 'infrastructure-side'")

    if not label_path.is_file():
        print(f"[WARN] Label file not found: {label_path}")
        return []

    data = load_json(label_path)
    recs = data if isinstance(data, list) else data.get("labels", data.get("annotations", []))

    out = []
    for g in recs or []:
        typ = g.get("type", "Car")
        if SHOW_CLASSES is not None and typ not in SHOW_CLASSES:
            continue

        dims = g.get("3d_dimensions", {}) or {}
        loc = g.get("3d_location", {}) or {}

        h = _as_float(dims.get("h"), 0.0)
        w = _as_float(dims.get("w"), 0.0)
        l = _as_float(dims.get("l"), 0.0)

        x = _as_float(loc.get("x"), 0.0)
        y = _as_float(loc.get("y"), 0.0)
        z = _as_float(loc.get("z"), 0.0)

        # DAIR-V2X: z is typically bottom of box
        if z_is_bottom:
            z = z + h / 2.0  # Convert to center

        yaw = _as_float(g.get("rotation", g.get("yaw", 0.0)), 0.0)

        out.append({
            "type": typ,
            "center": np.array([x, y, z], dtype=np.float64),
            "hwl": (h, w, l),
            "yaw": float(yaw),
        })
    return out

def lidar_box_corners_center(center: np.ndarray, h: float, w: float, l: float, yaw: float) -> np.ndarray:
    """
    Build 8 corners in LiDAR frame (x fwd, y left, z up), centered box.
    yaw rotates about +Z.
    """
    x_c = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2], dtype=np.float64)
    y_c = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2], dtype=np.float64)
    z_c = np.array([ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2], dtype=np.float64)

    c = math.cos(yaw)
    s = math.sin(yaw)
    R = np.array([[ c, -s, 0.0],
                  [ s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    corners = np.stack([x_c, y_c, z_c], axis=0)  # (3,8)
    corners = (R @ corners).T  # (8,3)
    corners += center.reshape(1, 3)
    return corners

# ============================================================
# Helpers: Open3D box drawing
# ============================================================

def make_lineset_from_corners(corners_xyz: np.ndarray, color_rgb) -> o3d.geometry.LineSet:
    corners_xyz = np.asarray(corners_xyz, dtype=np.float64).reshape(8, 3)
    lines = np.array([
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners_xyz)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color_rgb, dtype=np.float64), (lines.shape[0], 1)))
    return ls

# ============================================================
# Main
# ============================================================

def main():
    print(f"[INFO] DAIR-V2X sanity check for sample: {SAMPLE_ID}")
    print(f"[INFO] Data root: {DATA_ROOT}")

    # Load data_info.json files
    veh_info_path = DATA_ROOT / "vehicle-side" / "data_info.json"
    inf_info_path = DATA_ROOT / "infrastructure-side" / "data_info.json"

    if not veh_info_path.exists():
        print(f"[ERROR] Vehicle data_info.json not found: {veh_info_path}")
        sys.exit(1)
    if not inf_info_path.exists():
        print(f"[ERROR] Infrastructure data_info.json not found: {inf_info_path}")
        sys.exit(1)

    veh_info_list = load_json(veh_info_path)
    inf_info_list = load_json(inf_info_path)

    # Build lookup by pointcloud basename
    veh_lookup = {}
    for item in veh_info_list:
        pc_path = item.get("pointcloud_path", "")
        basename = Path(pc_path).stem
        veh_lookup[basename] = item

    inf_lookup = {}
    for item in inf_info_list:
        pc_path = item.get("pointcloud_path", "")
        basename = Path(pc_path).stem
        inf_lookup[basename] = item

    # Find sample
    if SAMPLE_ID not in veh_lookup:
        print(f"[ERROR] Vehicle sample {SAMPLE_ID} not found in data_info.json")
        print(f"[INFO] Available samples: {sorted(list(veh_lookup.keys()))[:10]}...")
        sys.exit(1)
    if SAMPLE_ID not in inf_lookup:
        print(f"[ERROR] Infrastructure sample {SAMPLE_ID} not found in data_info.json")
        sys.exit(1)

    veh_item = veh_lookup[SAMPLE_ID]
    inf_item = inf_lookup[SAMPLE_ID]

    # Load point clouds
    veh_pc_path = DATA_ROOT / "vehicle-side" / veh_item["pointcloud_path"]
    inf_pc_path = DATA_ROOT / "infrastructure-side" / inf_item["pointcloud_path"]

    print(f"[INFO] Vehicle PC: {veh_pc_path}")
    print(f"[INFO] Infra PC: {inf_pc_path}")

    veh_pts_v = load_pcd_xyzi(veh_pc_path)
    inf_pts_i = load_pcd_xyzi(inf_pc_path)

    veh_pts_v = maybe_limit_points(veh_pts_v, MAX_POINTS_PER_CLOUD)
    inf_pts_i = maybe_limit_points(inf_pts_i, MAX_POINTS_PER_CLOUD)

    # Load transforms
    veh_n2w_path = DATA_ROOT / "vehicle-side" / veh_item["calib_novatel_to_world_path"]
    veh_l2n_path = DATA_ROOT / "vehicle-side" / veh_item["calib_lidar_to_novatel_path"]
    inf_l2w_path = DATA_ROOT / "infrastructure-side" / inf_item["calib_virtuallidar_to_world_path"]

    print(f"[INFO] Vehicle novatel_to_world: {veh_n2w_path}")
    print(f"[INFO] Vehicle lidar_to_novatel: {veh_l2n_path}")
    print(f"[INFO] Infra virtuallidar_to_world: {inf_l2w_path}")

    veh_n2w = load_json(veh_n2w_path)
    veh_l2n = load_json(veh_l2n_path)
    inf_l2w = load_json(inf_l2w_path)

    v_n2w_r, v_n2w_t = get_rot_trans(veh_n2w)
    v_l2n_r, v_l2n_t = get_rot_trans(veh_l2n)
    i_l2w_r, i_l2w_t = get_rot_trans(inf_l2w)

    T_world_from_veh_novatel = make_T(v_n2w_r, v_n2w_t)
    T_veh_novatel_from_veh_lidar = make_T(v_l2n_r, v_l2n_t)
    T_world_from_veh_lidar = T_world_from_veh_novatel @ T_veh_novatel_from_veh_lidar

    T_world_from_inf_lidar = make_T(i_l2w_r, i_l2w_t)

    T_veh_from_world = inv_T(T_world_from_veh_lidar)
    T_veh_from_inf = T_veh_from_world @ T_world_from_inf_lidar

    # Transform infra points into vehicle lidar frame
    inf_pts_v = apply_T(T_veh_from_inf, inf_pts_i)

    # --- Open3D geometries ---
    geometries: List[o3d.geometry.Geometry] = []

    # Vehicle frame axes
    veh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SIZE_M, origin=[0.0, 0.0, 0.0])
    geometries.append(veh_frame)

    # Infra frame axes (transformed)
    inf_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SIZE_M, origin=[0.0, 0.0, 0.0])
    inf_frame.transform(T_veh_from_inf.astype(np.float64))
    geometries.append(inf_frame)

    # Point clouds
    veh_pcd = o3d_cloud(veh_pts_v, VEH_COLOR)
    inf_pcd = o3d_cloud(inf_pts_v, INF_COLOR)

    veh_pcd = maybe_downsample(veh_pcd, VOXEL_DOWNSAMPLE_M)
    inf_pcd = maybe_downsample(inf_pcd, VOXEL_DOWNSAMPLE_M)

    geometries.extend([veh_pcd, inf_pcd])

    # --- Native LiDAR GT boxes ---
    if SHOW_NATIVE_LIDAR_GT:
        veh_native = load_native_lidar_labels(DATA_ROOT, "vehicle-side", SAMPLE_ID)
        inf_native = load_native_lidar_labels(DATA_ROOT, "infrastructure-side", SAMPLE_ID)

        print(f"[INFO] Native LiDAR GT: vehicle={len(veh_native)} infra={len(inf_native)}")

        # Vehicle boxes (already in vehicle frame)
        for o in veh_native:
            h, w, l = o["hwl"]
            c = o["center"]
            yaw = o["yaw"]
            corners = lidar_box_corners_center(c, h, w, l, yaw)
            ls = make_lineset_from_corners(corners, VEH_BOX_COLOR_NATIVE)
            geometries.append(ls)

        # Infra boxes (transform to vehicle frame)
        for o in inf_native:
            h, w, l = o["hwl"]
            c = o["center"]
            yaw = o["yaw"]
            corners_inf = lidar_box_corners_center(c, h, w, l, yaw)
            corners_veh = apply_T(T_veh_from_inf, corners_inf)
            ls = make_lineset_from_corners(corners_veh, INF_BOX_COLOR_NATIVE)
            geometries.append(ls)

    # --- Visualize ---
    print(f"[INFO] Vehicle points shown: {np.asarray(veh_pcd.points).shape[0]}")
    print(f"[INFO] Infra points shown: {np.asarray(inf_pcd.points).shape[0]}")
    print("[INFO] Controls: left-drag rotate, shift+left-drag translate, wheel zoom")
    print("[INFO] Close the window to exit.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"DAIR-V2X sanity (veh frame) id={SAMPLE_ID}", width=1600, height=900, visible=True)
    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = float(O3D_POINT_SIZE)

    for g in geometries:
        vis.add_geometry(g)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}\n")
        import traceback
        traceback.print_exc()
        print("\nQuick checklist:")
        print("  1) DATA_ROOT points to cooperative-vehicle-infrastructure/")
        print("  2) SAMPLE_ID exists in both vehicle-side and infrastructure-side data_info.json")
        print("  3) Point cloud files (.pcd) exist for the sample")
        print("  4) Calibration JSON files exist for the sample")
        sys.exit(1)
