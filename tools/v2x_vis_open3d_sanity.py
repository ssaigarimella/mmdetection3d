#!/usr/bin/env python3
"""
tools/v2x_vis_open3d_sanity.py

Open3D sanity viewer for DAIR-V2X-style two-LiDAR setup (vehicle + infrastructure).

What it visualizes (in VEHICLE LiDAR frame):
  - vehicle KITTI velodyne points (red)
  - infra KITTI velodyne points transformed into vehicle LiDAR (blue)
  - vehicle native LiDAR GT boxes (green)  [from NON_KITTI_ROOT label/lidar JSON]
  - infra native LiDAR GT boxes transformed into vehicle LiDAR (yellow) [same transform as points]

Optional overlays (for debugging only):
  - vehicle KITTI label_2 boxes converted cam->lidar (cyan)
  - infra KITTI label_2 boxes converted cam->lidar then inf->veh (magenta)

Key point:
  Your collection code writes native LiDAR labels in the LiDAR sensor frame with a
  specific AirSim->KITTI axis conversion (y'=-y, z'=-z, yaw'=-yaw). Use these
  for correct 3D boxes in LiDAR space.
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

KITTI_ROOT = Path(
    "/home/dellg16ssg/multi-robot-coordination/collaborative-perception-BEVP/datasets/dair_v2x_synth_FULL_kitti"
)

NON_KITTI_ROOT = Path(
    "/home/dellg16ssg/multi-robot-coordination/collaborative-perception-BEVP/datasets/dair_v2x_synth_FULL"
)

SAMPLE_ID = "000000"
SPLIT = "training"  # "training" or "testing"

# Open3D visualization tuning
AXIS_SIZE_M = 2.0
VOXEL_DOWNSAMPLE_M = 0.10  # set to 0.0 to disable
MAX_POINTS_PER_CLOUD = 200_000  # set to 0 to disable
O3D_POINT_SIZE = 2.0

# Which GT source to use for 3D boxes
# Recommended: native lidar JSON GT (these should align with points)
SHOW_NATIVE_LIDAR_GT = True

# Optional: also draw boxes derived from KITTI label_2 (camera) for comparison
# These can look "tilted" depending on how camera extrinsics are defined.
SHOW_KITTI_LABEL2_GT = False

# Filter classes to show (set to None to show all)
SHOW_CLASSES = None  # e.g. {"Car", "Pedestrian"}

# Skip DontCare from KITTI label_2 parsing
SKIP_DONTCARE = True

# Point cloud colors (RGB in [0,1])
VEH_COLOR = (1.0, 0.2, 0.2)
INF_COLOR = (0.2, 0.6, 1.0)

# Box colors
VEH_BOX_COLOR_NATIVE = (0.2, 1.0, 0.2)     # green
INF_BOX_COLOR_NATIVE = (1.0, 0.85, 0.2)    # yellow

VEH_BOX_COLOR_KITTI = (0.2, 1.0, 1.0)      # cyan
INF_BOX_COLOR_KITTI = (1.0, 0.2, 1.0)      # magenta

# ============================================================
# Helpers: point clouds
# ============================================================

def load_kitti_bin(bin_path: Path) -> np.ndarray:
    """KITTI velodyne .bin is float32, typically Nx4 (x,y,z,intensity). Returns Nx3 xyz."""
    if not bin_path.is_file():
        raise FileNotFoundError(f"Missing bin: {bin_path}")
    pts = np.fromfile(str(bin_path), dtype=np.float32)
    if pts.size % 4 != 0:
        raise ValueError(f"Unexpected KITTI bin size (not divisible by 4): {bin_path} (floats={pts.size})")
    pts = pts.reshape(-1, 4)[:, :3]
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
# Helpers: rigid transforms (NON_KITTI_ROOT JSONs)
# ============================================================

def _as_np(x):
    return np.array(x, dtype=np.float64)

def quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n <= 0:
        raise ValueError("Invalid quaternion norm")
    qx, qy, qz, qw = q / n

    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)]
    ], dtype=np.float64)

def parse_rigid_json(json_path: Path) -> np.ndarray:
    """
    Parse common DAIR-V2X-like rigid transform JSON into 4x4.
    Supports:
      - {rotation: [[3x3]], translation: [3]}
      - flat 9-list rotation
      - quaternion + translation
      - optional nesting under common keys
    """
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing transform json: {json_path}")

    obj = json.loads(json_path.read_text())

    candidates = [obj]
    for k in ["transform", "Tr", "calib", "data", "extrinsic", "extrinsics"]:
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
            candidates.append(obj[k])

    def find_key(d, keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    R = None
    t = None

    for d in candidates:
        if not isinstance(d, dict):
            continue

        R_raw = find_key(d, ["rotation", "rot", "R"])
        t_raw = find_key(d, ["translation", "trans", "t"])

        if R_raw is not None and t_raw is not None:
            Rn = _as_np(R_raw)
            tn = _as_np(t_raw).reshape(-1)
            if Rn.size == 9:
                Rn = Rn.reshape(3, 3)
            if Rn.shape == (3, 3) and tn.size == 3:
                R, t = Rn, tn
                break

        q_raw = find_key(d, ["quaternion", "quat", "q"])
        if q_raw is not None and t_raw is not None:
            qn = _as_np(q_raw).reshape(-1)
            tn = _as_np(t_raw).reshape(-1)
            if tn.size == 3 and qn.size == 4:
                # accept either [qx,qy,qz,qw] or [qw,qx,qy,qz]
                qx, qy, qz, qw = qn
                if abs(qn[0]) > abs(qn[3]):
                    qw, qx, qy, qz = qn
                R = quat_to_R(qx, qy, qz, qw)
                t = tn
                break

    if R is None or t is None:
        raise ValueError(f"Could not parse rigid transform JSON into (R,t): {json_path}")

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

def find_transform_json(root: Path, sample_id: str, must_contain: str) -> Path:
    sample_name = f"{sample_id}.json"
    hits = []
    for p in root.rglob(sample_name):
        s = str(p).lower()
        if must_contain.lower() in s:
            hits.append(p)
    hits = sorted(hits)
    if not hits:
        raise FileNotFoundError(
            f"Could not find {sample_name} under {root} with path containing '{must_contain}'."
        )
    return hits[0]

def get_coop_root(non_kitti_root: Path) -> Path:
    """Return the directory that contains vehicle-side/, infrastructure-side/, cooperative/."""
    cand = non_kitti_root / "cooperative-vehicle-infrastructure"
    if cand.is_dir():
        return cand
    return non_kitti_root

# ============================================================
# Helpers: native LiDAR GT labels (NON_KITTI_ROOT label/lidar/*.json)
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

def load_native_lidar_labels(coop_root: Path, side: str, sample_id: str) -> List[Dict]:
    """
    Reads the native-format LiDAR labels written by your collection pipeline:
      <coop_root>/<side>/label/lidar/<id>.json

    Each entry provides:
      - 3d_location (x,y,z) as CENTER in LiDAR frame
      - 3d_dimensions (h,w,l)
      - rotation yaw (radians) about +Z in KITTI LiDAR basis (x fwd, y left, z up)
    """
    if side not in ("vehicle-side", "infrastructure-side"):
        raise ValueError("side must be 'vehicle-side' or 'infrastructure-side'")

    p = coop_root / side / "label" / "lidar" / f"{sample_id}.json"
    if not p.is_file():
        # fallback: search (in case your tree differs)
        hits = sorted(coop_root.rglob(f"{sample_id}.json"))
        hits = [h for h in hits if ("/label/lidar/" in h.as_posix().lower()) and (side in h.as_posix())]
        if not hits:
            raise FileNotFoundError(f"Could not find native lidar label json for {side} id={sample_id} under {coop_root}")
        p = hits[0]

    data = json.loads(p.read_text())
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

        cx = _as_float(loc.get("x"), 0.0)
        cy = _as_float(loc.get("y"), 0.0)
        cz = _as_float(loc.get("z"), 0.0)

        yaw = _as_float(g.get("rotation", g.get("yaw", 0.0)), 0.0)

        out.append({
            "type": typ,
            "center": np.array([cx, cy, cz], dtype=np.float64),
            "hwl": (h, w, l),
            "yaw": float(yaw),
            "src_path": str(p),
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
# Helpers: KITTI label_2 (optional overlay)
# ============================================================

def _parse_calib_txt(calib_path: Path) -> dict:
    if not calib_path.is_file():
        raise FileNotFoundError(f"Missing calib: {calib_path}")
    out = {}
    for line in calib_path.read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        vals = [float(x) for x in v.strip().split()]
        out[k] = np.array(vals, dtype=np.float64)
    return out

def _find_first_key(d: dict, keys) -> str:
    for k in keys:
        if k in d:
            return k
    return ""

def _mat3x4(arr: np.ndarray) -> np.ndarray:
    arr = arr.reshape(-1)
    if arr.size != 12:
        raise ValueError(f"Expected 12 floats for 3x4, got {arr.size}")
    return arr.reshape(3, 4)

def _mat3x3(arr: np.ndarray) -> np.ndarray:
    arr = arr.reshape(-1)
    if arr.size != 9:
        raise ValueError(f"Expected 9 floats for 3x3, got {arr.size}")
    return arr.reshape(3, 3)

def kitti_cam_to_velo_rect(calib_dict: dict) -> np.ndarray:
    rect_key = _find_first_key(calib_dict, ["R0_rect", "R_rect", "R0", "Rect", "R_rect0"])
    R0 = _mat3x3(calib_dict[rect_key]) if rect_key else np.eye(3, dtype=np.float64)

    # Accept several variants, including infra virtual lidar exports
    tr_key = _find_first_key(
        calib_dict,
        [
            "Tr_velo_to_cam", "Tr_velo_to_camera",
            "Tr_lidar_to_cam", "Tr_lidar_to_camera",
            "Tr_virtuallidar_to_camera", "Tr_virtuallidar_to_cam",
        ],
    )
    if not tr_key:
        raise ValueError("Could not find a lidar->cam extrinsic in calib txt.")

    Tr = _mat3x4(calib_dict[tr_key])

    T_cam_from_velo = np.eye(4, dtype=np.float64)
    T_cam_from_velo[:3, :] = Tr

    T_rect = np.eye(4, dtype=np.float64)
    T_rect[:3, :3] = R0

    # cam_rect = T_rect * T_cam_from_velo * velo
    # velo = inv(T_cam_from_velo) * inv(T_rect) * cam_rect
    return inv_T(T_cam_from_velo) @ inv_T(T_rect)

def read_kitti_label2(label_path: Path) -> List[Dict]:
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label: {label_path}")
    objs = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 15:
            continue
        obj_type = parts[0]
        if SKIP_DONTCARE and obj_type.lower() == "dontcare":
            continue
        if SHOW_CLASSES is not None and obj_type not in SHOW_CLASSES:
            continue
        h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
        x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
        ry = float(parts[14])
        objs.append({"type": obj_type, "hwl": (h, w, l), "loc_cam": (x, y, z), "ry": ry})
    return objs

def kitti_box_corners_cam(loc_cam, hwl, ry) -> np.ndarray:
    """
    KITTI camera rect:
      x right, y down, z forward
      location is bottom-center
      ry around +Y (down)
    """
    x, y, z = loc_cam
    h, w, l = hwl

    x_c = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2], dtype=np.float64)
    y_c = np.array([   0,    0,    0,    0,  -h,  -h,   -h,   -h], dtype=np.float64)
    z_c = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2], dtype=np.float64)

    c = math.cos(ry)
    s = math.sin(ry)
    R = np.array([[ c, 0.0,  s],
                  [0.0, 1.0, 0.0],
                  [-s, 0.0,  c]], dtype=np.float64)

    corners = np.stack([x_c, y_c, z_c], axis=0)  # (3,8)
    corners = (R @ corners).T
    corners[:, 0] += x
    corners[:, 1] += y
    corners[:, 2] += z
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
    veh_root_k = KITTI_ROOT / "vehicle-side" / SPLIT
    inf_root_k = KITTI_ROOT / "infrastructure-side" / SPLIT

    veh_bin = veh_root_k / "velodyne" / f"{SAMPLE_ID}.bin"
    inf_bin = inf_root_k / "velodyne" / f"{SAMPLE_ID}.bin"

    print(f"[INFO] vehicle bin: {veh_bin}")
    print(f"[INFO] infra   bin: {inf_bin}")

    veh_pts_v = load_kitti_bin(veh_bin)
    inf_pts_i = load_kitti_bin(inf_bin)

    veh_pts_v = maybe_limit_points(veh_pts_v, MAX_POINTS_PER_CLOUD)
    inf_pts_i = maybe_limit_points(inf_pts_i, MAX_POINTS_PER_CLOUD)

    coop_root = get_coop_root(NON_KITTI_ROOT)

    # --- Load transforms (native JSON) ---
    # Vehicle chain: novatel_to_world ∘ lidar_to_novatel
    veh_novatel_to_world_json = find_transform_json(coop_root, SAMPLE_ID, "novatel_to_world")
    veh_lidar_to_novatel_json = find_transform_json(coop_root, SAMPLE_ID, "lidar_to_novatel")
    # Infra: virtuallidar_to_world
    inf_lidar_to_world_json = find_transform_json(coop_root, SAMPLE_ID, "virtuallidar_to_world")

    print(f"[INFO] veh novatel_to_world: {veh_novatel_to_world_json}")
    print(f"[INFO] veh lidar_to_novatel: {veh_lidar_to_novatel_json}")
    print(f"[INFO] inf virtuallidar_to_world: {inf_lidar_to_world_json}")

    T_world_from_veh_novatel = parse_rigid_json(veh_novatel_to_world_json)        # novatel -> world
    T_veh_novatel_from_veh_lidar = parse_rigid_json(veh_lidar_to_novatel_json)    # lidar -> novatel
    T_world_from_veh_lidar = T_world_from_veh_novatel @ T_veh_novatel_from_veh_lidar

    T_world_from_inf_lidar = parse_rigid_json(inf_lidar_to_world_json)

    T_veh_from_world = inv_T(T_world_from_veh_lidar)
    T_veh_from_inf = T_veh_from_world @ T_world_from_inf_lidar

    # Transform infra points into vehicle lidar
    inf_pts_v = apply_T(T_veh_from_inf, inf_pts_i)

    # --- Open3D geometries ---
    geometries: List[o3d.geometry.Geometry] = []

    veh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SIZE_M, origin=[0.0, 0.0, 0.0])
    geometries.append(veh_frame)

    inf_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SIZE_M, origin=[0.0, 0.0, 0.0])
    inf_frame.transform(T_veh_from_inf.astype(np.float64))
    geometries.append(inf_frame)

    veh_pcd = o3d_cloud(veh_pts_v, VEH_COLOR)
    inf_pcd = o3d_cloud(inf_pts_v, INF_COLOR)

    veh_pcd = maybe_downsample(veh_pcd, VOXEL_DOWNSAMPLE_M)
    inf_pcd = maybe_downsample(inf_pcd, VOXEL_DOWNSAMPLE_M)

    geometries.extend([veh_pcd, inf_pcd])

    # Render options
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"V2X sanity (veh frame) id={SAMPLE_ID}", width=1600, height=900, visible=True)
    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = float(O3D_POINT_SIZE)

    for g in geometries:
        vis.add_geometry(g)

    # --- Native LiDAR GT (recommended) ---
    if SHOW_NATIVE_LIDAR_GT:
        veh_native = load_native_lidar_labels(coop_root, "vehicle-side", SAMPLE_ID)
        inf_native = load_native_lidar_labels(coop_root, "infrastructure-side", SAMPLE_ID)

        print(f"[INFO] native lidar GT: vehicle={len(veh_native)} infra={len(inf_native)}")
        if len(veh_native) > 0:
            print(f"[INFO] vehicle native label source: {veh_native[0]['src_path']}")
        if len(inf_native) > 0:
            print(f"[INFO] infra   native label source: {inf_native[0]['src_path']}")

        for o in veh_native:
            h, w, l = o["hwl"]
            c = o["center"]
            yaw = o["yaw"]
            corners = lidar_box_corners_center(c, h, w, l, yaw)
            ls = make_lineset_from_corners(corners, VEH_BOX_COLOR_NATIVE)
            vis.add_geometry(ls)

        for o in inf_native:
            h, w, l = o["hwl"]
            c = o["center"]
            yaw = o["yaw"]
            corners_inf = lidar_box_corners_center(c, h, w, l, yaw)
            corners_veh = apply_T(T_veh_from_inf, corners_inf)
            ls = make_lineset_from_corners(corners_veh, INF_BOX_COLOR_NATIVE)
            vis.add_geometry(ls)

    # --- Optional: KITTI label_2 overlay (debug only) ---
    if SHOW_KITTI_LABEL2_GT:
        veh_calib = veh_root_k / "calib" / f"{SAMPLE_ID}.txt"
        inf_calib = inf_root_k / "calib" / f"{SAMPLE_ID}.txt"
        veh_label = veh_root_k / "label_2" / f"{SAMPLE_ID}.txt"
        inf_label = inf_root_k / "label_2" / f"{SAMPLE_ID}.txt"

        print(f"[INFO] KITTI label_2 overlay enabled.")
        print(f"[INFO] vehicle calib: {veh_calib}")
        print(f"[INFO] infra   calib: {inf_calib}")
        print(f"[INFO] vehicle label: {veh_label}")
        print(f"[INFO] infra   label: {inf_label}")

        veh_cal = _parse_calib_txt(veh_calib)
        inf_cal = _parse_calib_txt(inf_calib)
        T_veh_velo_from_cam = kitti_cam_to_velo_rect(veh_cal)
        T_inf_velo_from_cam = kitti_cam_to_velo_rect(inf_cal)

        veh_objs = read_kitti_label2(veh_label)
        inf_objs = read_kitti_label2(inf_label)

        print(f"[INFO] KITTI label_2 objects kept: vehicle={len(veh_objs)} infra={len(inf_objs)}")

        for o in veh_objs:
            corners_cam = kitti_box_corners_cam(o["loc_cam"], o["hwl"], o["ry"])
            corners_velo = apply_T(T_veh_velo_from_cam, corners_cam)
            ls = make_lineset_from_corners(corners_velo, VEH_BOX_COLOR_KITTI)
            vis.add_geometry(ls)

        for o in inf_objs:
            corners_cam = kitti_box_corners_cam(o["loc_cam"], o["hwl"], o["ry"])
            corners_inf_velo = apply_T(T_inf_velo_from_cam, corners_cam)
            corners_veh_velo = apply_T(T_veh_from_inf, corners_inf_velo)
            ls = make_lineset_from_corners(corners_veh_velo, INF_BOX_COLOR_KITTI)
            vis.add_geometry(ls)

    print(f"[INFO] vehicle points shown: {np.asarray(veh_pcd.points).shape[0]}")
    print(f"[INFO] infra   points shown: {np.asarray(inf_pcd.points).shape[0]}")
    print("[INFO] Controls: left-drag rotate, shift+left-drag translate, wheel zoom")
    print("[INFO] Close the window to exit.")

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}\n")
        print("Quick checklist:")
        print("  1) KITTI_ROOT points to your *_kitti folder containing vehicle-side/ and infrastructure-side/.")
        print("  2) NON_KITTI_ROOT points to the folder that contains cooperative-vehicle-infrastructure/,")
        print("     or is itself the cooperative-vehicle-infrastructure folder.")
        print("  3) SAMPLE_ID exists on BOTH sides for KITTI velodyne, and for native label/lidar JSONs.")
        print("  4) If your transform folder names differ, adjust find_transform_json must_contain strings.")
        sys.exit(1)
