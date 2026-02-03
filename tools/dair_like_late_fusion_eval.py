#!/usr/bin/env python3
"""
DAIR-V2X-like late-fusion evaluation inside mmdetection3d.

Replicates DAIR-V2X eval logic (cooperative GT in vehicle frame, DAIR IoU/AP),
but runs mmdet3d models from this repo in the mmdet3d_pp env.

No time compensation (by default) and no changes to existing scripts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import open3d as o3d

from mmdet3d.apis.inference import init_model, inference_detector


# -------------------------
# DAIR label mappings
# -------------------------
NAME2ID = {
    "car": 2,
    "van": 2,
    "truck": 2,
    "bus": 2,
    "cyclist": 1,
    "tricyclist": 3,
    "motorcyclist": 3,
    "barrow": 3,
    "barrowlist": 3,
    "pedestrian": 0,
    "trafficcone": 3,
    "pedestrianignore": 3,
    "carignore": 3,
    "otherignore": 3,
    "unknowns_unmovable": 3,
    "unknowns_movable": 3,
    "unknown_unmovable": 3,
    "unknown_movable": 3,
}

SUPERCLASS = {
    -1: "ignore",
    0: "pedestrian",
    1: "cyclist",
    2: "car",
    3: "ignore",
}

IOU_THRESHOLDS = {
    "car": [0.3, 0.5, 0.7],
    "cyclist": [0.25, 0.5],
    "pedestrian": [0.25, 0.5],
}


# -------------------------
# Geometry utils (ported from DAIR)
# -------------------------
def range2box(box_range):
    box_range = np.array(box_range, dtype=np.float64)
    idxs = [
        [0, 1, 2],
        [3, 1, 2],
        [3, 4, 2],
        [0, 4, 2],
        [0, 1, 5],
        [3, 1, 5],
        [3, 4, 5],
        [0, 4, 5],
    ]
    return np.array([[box_range[i] for i in idxs]], dtype=np.float64)


def _dot(p1, p2):
    return p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]


def _cross(p1, p2):
    return [
        p1[1] * p2[2] - p1[2] * p2[1],
        p1[2] * p2[0] - p1[0] * p2[2],
        p1[0] * p2[1] - p2[0] * p1[1],
    ]


def _above_plane(point, plane):
    norm = _cross(plane[1] - plane[0], plane[2] - plane[0])
    d = _dot(plane[0], norm)
    z_intersec = (d - norm[0] * point[0] - norm[1] * point[1]) / norm[2]
    t = (norm[0] * point[0] + norm[1] * point[1] + norm[2] * point[2] - d) / (
        norm[0] ** 2 + norm[1] ** 2 + norm[2] ** 2
    )
    point_x = point[0] - norm[0] * t
    point_y = point[1] - norm[1] * t

    def _is_inside(x1, y1, x2, y2, x3, y3, x4, y4, x, y):
        def _get_cross(x1, y1, x2, y2, x, y):
            a = (x2 - x1, y2 - y1)
            b = (x - x1, y - y1)
            return a[0] * b[1] - a[1] * b[0]

        return (
            _get_cross(x1, y1, x2, y2, x, y) * _get_cross(x3, y3, x4, y4, x, y) >= 0
            and _get_cross(x2, y2, x3, y3, x, y) * _get_cross(x4, y4, x1, y1, x, y) >= 0
        )

    if z_intersec <= point[2] and _is_inside(
        plane[0][0], plane[0][1],
        plane[1][0], plane[1][1],
        plane[2][0], plane[2][1],
        plane[3][0], plane[3][1],
        point_x, point_y,
    ):
        return 1
    return 0


def point_in_box(point, box):
    return _above_plane(point, box[:4]) + _above_plane(point, box[4:]) == 1


class RectFilter:
    def __init__(self, bbox):
        self.bbox = bbox

    def __call__(self, box, **kwargs):
        for corner in box:
            if point_in_box(corner, self.bbox):
                return True
        return False


# -------------------------
# DAIR eval utils (ported)
# -------------------------
from functools import cmp_to_key
from scipy.spatial import ConvexHull


def polygon_clip(subjectPolygon, clipPolygon):
    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]
    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
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


def convex_hull_intersection(p1, p2):
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    return None, 0


def box3d_iou(corners1, corners2):
    # 3d IoU + BEV IoU from DAIR
    rect1 = [(corners1[i, 0], corners1[i, 1]) for i in range(4)]
    rect2 = [(corners2[i, 0], corners2[i, 1]) for i in range(4)]
    area1 = ConvexHull(rect1).volume
    area2 = ConvexHull(rect2).volume
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area) if inter_area > 0 else 0
    zmax = min(corners1[0, 2], corners2[0, 2])
    zmin = max(corners1[4, 2], corners2[4, 2])
    inter_vol = inter_area * max(0.0, zmax - zmin)
    vol1 = area1 * (corners1[0, 2] - corners1[4, 2])
    vol2 = area2 * (corners2[0, 2] - corners2[4, 2])
    iou_3d = inter_vol / (vol1 + vol2 - inter_vol) if inter_vol > 0 else 0
    return iou_3d, iou_2d


perm_pred = [0, 4, 7, 3, 1, 5, 6, 2]
perm_label = [3, 2, 1, 0, 7, 6, 5, 4]


def cmp_pred(p1, p2):
    return -1 if p1["score"] > p2["score"] else (1 if p1["score"] < p2["score"] else 0)


def build_label_list(annos, filt):
    result_list = []
    for i in range(len(annos["labels_3d"])):
        if SUPERCLASS[annos["labels_3d"][i]] == filt:
            result_list.append({"box": annos["boxes_3d"][i], "score": annos["scores_3d"][i]})
    return result_list


def compute_type(gt_annos, pred_annos, cla, iou_threshold, view):
    gt_annos = build_label_list(gt_annos, filt=cla)
    pred_annos = build_label_list(pred_annos, filt=cla)
    pred_annos = sorted(pred_annos, key=cmp_to_key(cmp_pred))
    result_pred_annos = []
    for i in range(len(pred_annos)):
        pred_annos[i]["id"] = i
    for gt_anno in gt_annos:
        mx = iou_threshold
        mx_pred = None
        for i in range(len(pred_annos)):
            pred_anno = pred_annos[i]
            try:
                iou, iou_2d = box3d_iou(gt_anno["box"][perm_label], pred_anno["box"][perm_pred])
            except Exception:
                iou, iou_2d = 0, 0
            if view == "bev":
                iou = iou_2d
            if iou >= mx:
                mx = iou
                mx_pred = i
        if mx_pred is not None:
            result_pred_annos.append(pred_annos[mx_pred])
            del pred_annos[mx_pred]
            result_pred_annos[-1]["type"] = "tp"
    for pred_anno in pred_annos:
        pred_anno["type"] = "fp"
        result_pred_annos.append(pred_anno)
    return result_pred_annos, len(gt_annos)


def compute_ap(pred_annos, num_gt):
    pred_annos = sorted(pred_annos, key=cmp_to_key(cmp_pred))
    num_tp = np.zeros(len(pred_annos))
    for i in range(len(pred_annos)):
        num_tp[i] = 0 if i == 0 else num_tp[i - 1]
        if pred_annos[i]["type"] == "tp":
            num_tp[i] += 1
    precision = num_tp / np.arange(1, len(pred_annos) + 1)
    recall = num_tp / num_gt if num_gt > 0 else np.zeros_like(num_tp)
    for i in range(len(pred_annos) - 1, 0, -1):
        precision[i - 1] = max(precision[i], precision[i - 1])
    idx = np.where(recall[1:] != recall[:-1])[0]
    return np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1]) if len(idx) > 0 else 0.0


class Evaluator:
    def __init__(self, pred_classes):
        self.pred_classes = pred_classes
        self.all_preds = {"3d": {}, "bev": {}}
        self.gt_num = {}
        for pred_class in self.pred_classes:
            self.all_preds["3d"][pred_class] = {}
            self.all_preds["bev"][pred_class] = {}
            self.gt_num[pred_class] = {}
            for iou in IOU_THRESHOLDS[pred_class]:
                self.all_preds["3d"][pred_class][iou] = []
                self.all_preds["bev"][pred_class][iou] = []
                self.gt_num[pred_class][iou] = 0

    def add_frame(self, pred, label):
        for pred_class in self.pred_classes:
            for iou in IOU_THRESHOLDS[pred_class]:
                pred_result, num_label = compute_type(label, pred, pred_class, iou, "3d")
                self.all_preds["3d"][pred_class][iou] += pred_result
                self.all_preds["bev"][pred_class][iou] += compute_type(label, pred, pred_class, iou, "bev")[0]
                self.gt_num[pred_class][iou] += num_label

    def print_ap(self, view):
        for pred_class in self.pred_classes:
            for iou in IOU_THRESHOLDS[pred_class]:
                ap = compute_ap(self.all_preds[view][pred_class][iou], self.gt_num[pred_class][iou])
                print(f"{pred_class} {view} IoU threshold {iou:.2f}, Average Precision = {ap * 100:.2f}")


# -------------------------
# Late fusion (ported)
# -------------------------
class BBoxList:
    def __init__(self, boxes, label, score):
        self.num_boxes = boxes.shape[0]
        self.num_dims = boxes.shape[2]
        self.boxes = boxes
        self.label = label
        self.confidence = score
        self.center = np.mean(self.boxes, axis=1) if self.num_boxes > 0 else np.zeros((0, 3))


def diff_label_filt(frame1, frame2, i, j):
    size = np.maximum(np.abs(frame1.center[i] - frame1.center[i]) + 1.0, 1.0)
    diff = np.abs(frame1.center[i] - frame2.center[j]) / size
    return diff[0] <= 1 and diff[1] <= 1 and diff[2] <= 1 and frame1.label[i] == frame2.label[j]


class EuclidianMatcher:
    def __init__(self, filter_func=None):
        self.filter_func = filter_func

    def match(self, frame1, frame2):
        from scipy.optimize import linear_sum_assignment
        cost_matrix = np.zeros((frame1.num_boxes, frame2.num_boxes))
        for i in range(frame1.num_boxes):
            for j in range(frame2.num_boxes):
                cost_matrix[i][j] = np.linalg.norm(frame1.center[i] - frame2.center[j])
                if self.filter_func is not None and not self.filter_func(frame1, frame2, i, j):
                    cost_matrix[i][j] = 1e6
        index1, index2 = linear_sum_assignment(cost_matrix)
        accepted = []
        for i in range(len(index1)):
            if cost_matrix[index1[i]][index2[i]] < 1e5:
                accepted.append(i)
        return index1[accepted], index2[accepted]


class BasicFuser:
    def __init__(self, perspective="vehicle", trust_type="main", retain_type="all"):
        self.perspective = perspective
        self.trust_type = trust_type
        self.retain_type = retain_type

    def fuse(self, frame_r, frame_v, ind_r, ind_v):
        if self.perspective == "vehicle":
            frame1, frame2 = frame_v, frame_r
            ind1, ind2 = ind_v, ind_r
        else:
            frame1, frame2 = frame_r, frame_v
            ind1, ind2 = ind_r, ind_v

        if len(ind1) == 0:
            return {
                "boxes_3d": frame2.boxes,
                "labels_3d": frame2.label,
                "scores_3d": frame2.confidence,
            }

        confidence1 = np.array(frame1.confidence[ind1])
        confidence2 = np.array(frame2.confidence[ind2])
        if self.trust_type == "main":
            confidence1 = np.ones_like(confidence1)
            confidence2 = 1 - confidence1

        center = frame1.center[ind1] * confidence1[:, None] + frame2.center[ind2] * confidence2[:, None]
        boxes = frame1.boxes[ind1] + center[:, None, :] - frame1.center[ind1][:, None, :]
        label = frame1.label[ind1]
        confidence = frame1.confidence[ind1] * confidence1 + frame2.confidence[ind2] * confidence2

        boxes_u, label_u, confidence_u = [], [], []
        if self.retain_type in ["all", "main"]:
            for i in range(frame1.num_boxes):
                if i not in ind1 and frame1.label[i] != -1:
                    boxes_u.append(frame1.boxes[i])
                    label_u.append(frame1.label[i])
                    confidence_u.append(frame1.confidence[i])
        if self.retain_type in ["all"]:
            for i in range(frame2.num_boxes):
                if i not in ind2 and frame2.label[i] != -1:
                    boxes_u.append(frame2.boxes[i])
                    label_u.append(frame2.label[i])
                    confidence_u.append(frame2.confidence[i] * 0.4)

        if len(boxes_u) == 0:
            return {"boxes_3d": boxes, "labels_3d": label, "scores_3d": confidence}
        return {
            "boxes_3d": np.concatenate((boxes, np.array(boxes_u)), axis=0),
            "labels_3d": np.concatenate((label, np.array(label_u)), axis=0),
            "scores_3d": np.concatenate((confidence, np.array(confidence_u)), axis=0),
        }


# -------------------------
# Calibration utils
# -------------------------
def load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def get_rot_trans(d: Dict):
    if "rotation" in d and "translation" in d:
        return d["rotation"], d["translation"]
    if "transform" in d and isinstance(d["transform"], dict):
        t = d["transform"]
        if "rotation" in t and "translation" in t:
            return t["rotation"], t["translation"]
    if "rotation_matrix" in d and "translation_vector" in d:
        return d["rotation_matrix"], d["translation_vector"]
    raise KeyError("rotation/translation not found in calibration json")


def make_T(rot, trans):
    R = np.array(rot, dtype=np.float64).reshape(3, 3)
    t = np.array(trans, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_T_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    out = (T @ pts_h.T).T[:, :3]
    return out


def apply_T_boxes(T: np.ndarray, boxes_8x3: np.ndarray) -> np.ndarray:
    n = boxes_8x3.shape[0]
    out = boxes_8x3.reshape(-1, 3)
    out = apply_T_points(T, out)
    return out.reshape(n, 8, 3)


def T_veh_from_inf(data_root: Path, veh_info: Dict, inf_info: Dict) -> np.ndarray:
    veh_n2w = load_json(data_root / "vehicle-side" / veh_info["calib_novatel_to_world_path"])
    veh_l2n = load_json(data_root / "vehicle-side" / veh_info["calib_lidar_to_novatel_path"])
    inf_l2w = load_json(data_root / "infrastructure-side" / inf_info["calib_virtuallidar_to_world_path"])

    v_n2w_r, v_n2w_t = get_rot_trans(veh_n2w)
    v_l2n_r, v_l2n_t = get_rot_trans(veh_l2n)
    i_l2w_r, i_l2w_t = get_rot_trans(inf_l2w)
    T_w_from_veh = make_T(v_n2w_r, v_n2w_t) @ make_T(v_l2n_r, v_l2n_t)
    T_veh_from_w = np.linalg.inv(T_w_from_veh)
    T_w_from_inf = make_T(i_l2w_r, i_l2w_t)
    return T_veh_from_w @ T_w_from_inf


def T_veh_from_world(data_root: Path, veh_info: Dict) -> np.ndarray:
    veh_n2w = load_json(data_root / "vehicle-side" / veh_info["calib_novatel_to_world_path"])
    veh_l2n = load_json(data_root / "vehicle-side" / veh_info["calib_lidar_to_novatel_path"])
    v_n2w_r, v_n2w_t = get_rot_trans(veh_n2w)
    v_l2n_r, v_l2n_t = get_rot_trans(veh_l2n)
    T_w_from_veh = make_T(v_n2w_r, v_n2w_t) @ make_T(v_l2n_r, v_l2n_t)
    return np.linalg.inv(T_w_from_veh)


# -------------------------
# Label loading (cooperative GT)
# -------------------------
def get_3d_8points(obj_size, yaw_lidar, center_lidar):
    import math
    liadr_r = np.matrix(
        [
            [math.cos(yaw_lidar), -math.sin(yaw_lidar), 0],
            [math.sin(yaw_lidar), math.cos(yaw_lidar), 0],
            [0, 0, 1],
        ]
    )
    l, w, h = obj_size
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = liadr_r * corners_3d_lidar + np.matrix(center_lidar).T
    return corners_3d_lidar.T


def load_coop_label(path: Path, filt: RectFilter):
    raw = load_json(path)
    boxes = []
    labels = []
    for label in raw:
        size = label["3d_dimensions"]
        if size["l"] == 0 or size["w"] == 0 or size["h"] == 0:
            continue
        if "world_8_points" in label:
            box = label["world_8_points"]
        else:
            pos = label["3d_location"]
            box = get_3d_8points(
                [float(size["l"]), float(size["w"]), float(size["h"])],
                float(label["rotation"]),
                [float(pos["x"]), float(pos["y"]), float(pos["z"]) - float(size["h"]) / 2],
            ).tolist()
        if filt is None or filt(box):
            boxes.append(box)
            labels.append(NAME2ID[label["type"].lower()])
    boxes = np.array(boxes, dtype=np.float64)
    labels = np.array(labels, dtype=np.int64)
    scores = np.ones_like(labels, dtype=np.float32)
    return boxes, labels, scores


# -------------------------
# Dataset helpers
# -------------------------
def build_path_to_info(prefix: str, data: List[Dict]) -> Dict[str, Dict]:
    path2info = {}
    for elem in data:
        p = elem.get("pointcloud_path", "")
        if not p:
            continue
        path = os.path.join(prefix, p)
        path2info[path] = elem
    return path2info


def load_split_list(split_json: Path, split: str) -> List[str]:
    d = load_json(split_json)
    return d["cooperative_split"][split]


def filter_frame_pairs(frame_pairs: List[Dict], split_ids: List[str]) -> List[Dict]:
    out = []
    for fp in frame_pairs:
        vid = os.path.splitext(os.path.basename(fp["vehicle_image_path"]))[0]
        if vid in split_ids:
            out.append(fp)
    return out


def prev_inf_frame(inf_path2info: Dict[str, Dict], inf_info: Dict, k: int) -> Tuple[Dict, float]:
    cur_id = os.path.splitext(os.path.basename(inf_info["pointcloud_path"]))[0]
    cur_full = f"infrastructure-side/velodyne/{cur_id}.pcd"
    cur = inf_path2info.get(cur_full)
    if cur is None:
        return None, None
    batch_start = int(cur["batch_start_id"])
    prev_id = int(cur_id) - int(k)
    if prev_id < batch_start:
        return None, None
    prev_full = f"infrastructure-side/velodyne/{prev_id:06d}.pcd"
    prev = inf_path2info.get(prev_full)
    if prev is None:
        return None, None
    delta_t = 0.0
    try:
        delta_t = (int(cur["pointcloud_timestamp"]) - int(prev["pointcloud_timestamp"])) / 1000.0
    except Exception:
        delta_t = 0.0
    return prev, delta_t


def load_pcd_xyz(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points, dtype=np.float32)


def mmdet3d_pred_to_dair(pred_sample, class_names: List[str], score_thr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inst = pred_sample.pred_instances_3d
    if inst is None or inst.bboxes_3d is None or len(inst.bboxes_3d) == 0:
        return np.zeros((0, 8, 3)), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    scores = inst.scores_3d.detach().cpu().numpy()
    labels = inst.labels_3d.detach().cpu().numpy()
    keep = scores >= float(score_thr)
    scores = scores[keep]
    labels = labels[keep]
    corners = inst.bboxes_3d.corners.detach().cpu().numpy()[keep]

    # map to DAIR label ids
    dair_labels = []
    for li in labels:
        cls = class_names[int(li)].lower()
        dair_labels.append(NAME2ID.get(cls, 3))
    return corners.astype(np.float64), np.array(dair_labels, dtype=np.int64), scores.astype(np.float32)


def filter_by_range(boxes, labels, scores, filt: RectFilter):
    if boxes.shape[0] == 0:
        return boxes, labels, scores
    keep = []
    for i in range(boxes.shape[0]):
        if filt(boxes[i]):
            keep.append(i)
    keep = np.array(keep, dtype=np.int64)
    if keep.size == 0:
        return np.zeros((0, 8, 3)), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    return boxes[keep], labels[keep], scores[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=str,
                    help="Path to cooperative-vehicle-infrastructure root.")
    ap.add_argument("--split-json", required=True, type=str)
    ap.add_argument("--split", default="val", type=str)
    ap.add_argument("--dataset", default="vic-async", choices=["vic-async", "vic-sync"])
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--fusion-method", default="late_fusion", choices=["late_fusion", "veh_only", "inf_only"])
    ap.add_argument("--veh-config", required=True, type=str)
    ap.add_argument("--veh-ckpt", required=True, type=str)
    ap.add_argument("--inf-config", required=True, type=str)
    ap.add_argument("--inf-ckpt", required=True, type=str)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--pred-classes", nargs="+", default=["car"])
    ap.add_argument("--score-thr", type=float, default=0.1)
    ap.add_argument("--extended-range", nargs="+", type=float, required=True)
    ap.add_argument("--output-dir", type=str, default="", help="Optional: save per-frame pkls like DAIR.")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    coop_info = load_json(data_root / "cooperative" / "data_info.json")
    veh_info = load_json(data_root / "vehicle-side" / "data_info.json")
    inf_info = load_json(data_root / "infrastructure-side" / "data_info.json")

    split_ids = load_split_list(Path(args.split_json), args.split)
    frame_pairs = filter_frame_pairs(coop_info, split_ids)

    veh_path2info = build_path_to_info("vehicle-side", veh_info)
    inf_path2info = build_path_to_info("infrastructure-side", inf_info)

    model_v = init_model(args.veh_config, args.veh_ckpt, device=args.device)
    model_i = init_model(args.inf_config, args.inf_ckpt, device=args.device)

    class_names_v = model_v.dataset_meta.get("classes", [])
    class_names_i = model_i.dataset_meta.get("classes", [])

    pred_classes = [c.lower() for c in args.pred_classes]
    evaluator = Evaluator(pred_classes)

    box_range = np.array(args.extended_range, dtype=np.float64)
    filt_box = range2box(box_range)[0]
    filt = RectFilter(filt_box)

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        (out_dir / "result").mkdir(parents=True, exist_ok=True)

    for fp in frame_pairs:
        veh_pc = data_root / fp["vehicle_pointcloud_path"]
        inf_pc = data_root / fp["infrastructure_pointcloud_path"]
        veh_id = os.path.splitext(os.path.basename(fp["vehicle_pointcloud_path"]))[0]

        veh_info_item = veh_path2info[fp["vehicle_pointcloud_path"]]
        inf_info_item = inf_path2info[fp["infrastructure_pointcloud_path"]]

        # async: replace infra with previous frame if needed
        if args.dataset == "vic-async" and int(args.k) > 0:
            prev_inf, _ = prev_inf_frame(inf_path2info, inf_info_item, args.k)
            if prev_inf is None:
                continue
            inf_info_item = prev_inf
            inf_pc = data_root / inf_info_item["pointcloud_path"]

        pts_v = load_pcd_xyz(veh_pc)
        pts_i = load_pcd_xyz(inf_pc)

        pred_v, _ = inference_detector(model_v, pts_v)
        pred_i, _ = inference_detector(model_i, pts_i)

        v_boxes, v_labels, v_scores = mmdet3d_pred_to_dair(pred_v, class_names_v, args.score_thr)
        i_boxes, i_labels, i_scores = mmdet3d_pred_to_dair(pred_i, class_names_i, args.score_thr)

        # transform infra -> vehicle
        T_vi = T_veh_from_inf(data_root, veh_info_item, inf_info_item)
        if i_boxes.shape[0] > 0:
            i_boxes = apply_T_boxes(T_vi, i_boxes)

        # filter by range
        v_boxes, v_labels, v_scores = filter_by_range(v_boxes, v_labels, v_scores, filt)
        i_boxes, i_labels, i_scores = filter_by_range(i_boxes, i_labels, i_scores, filt)

        if args.fusion_method == "veh_only":
            pred = {"boxes_3d": v_boxes, "labels_3d": v_labels, "scores_3d": v_scores}
        elif args.fusion_method == "inf_only":
            pred = {"boxes_3d": i_boxes, "labels_3d": i_labels, "scores_3d": i_scores}
        else:
            matcher = EuclidianMatcher(diff_label_filt)
            pred_inf = BBoxList(i_boxes, i_labels, i_scores)
            pred_veh = BBoxList(v_boxes, v_labels, v_scores)
            ind_inf, ind_veh = matcher.match(pred_inf, pred_veh)
            fuser = BasicFuser(perspective="vehicle", trust_type="main", retain_type="all")
            pred = fuser.fuse(pred_inf, pred_veh, ind_inf, ind_veh)

        # load cooperative GT (world) -> vehicle
        coop_label_path = data_root / fp["cooperative_label_path"]
        gt_boxes_w, gt_labels, gt_scores = load_coop_label(coop_label_path, filt=None)
        T_vw = T_veh_from_world(data_root, veh_info_item)
        if gt_boxes_w.shape[0] > 0:
            gt_boxes = apply_T_boxes(T_vw, gt_boxes_w)
        else:
            gt_boxes = gt_boxes_w
        gt_boxes, gt_labels, gt_scores = filter_by_range(gt_boxes, gt_labels, gt_scores, filt)
        label = {"boxes_3d": gt_boxes, "labels_3d": gt_labels, "scores_3d": gt_scores}

        evaluator.add_frame(pred, label)

        if out_dir:
            save = {
                "boxes_3d": pred["boxes_3d"],
                "labels_3d": pred["labels_3d"],
                "scores_3d": pred["scores_3d"],
            }
            with open(out_dir / "result" / f"{veh_id}.pkl", "wb") as f:
                import pickle
                pickle.dump(save, f)

    evaluator.print_ap("3d")
    evaluator.print_ap("bev")


if __name__ == "__main__":
    main()
