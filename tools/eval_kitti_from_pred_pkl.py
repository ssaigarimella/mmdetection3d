#!/usr/bin/env python3
import argparse
import numpy as np
import mmengine
import torch
from mmengine.config import Config
from mmengine.structures import InstanceData

from mmdet3d.evaluation.metrics import KittiMetric
from mmdet3d.structures import LiDARInstance3DBoxes


# MMDet3D LiDAR boxes are (x, y, z, dx, dy, dz, yaw)
# For KITTI LiDAR, the box origin in z should be bottom (origin_z = 0.0).
# KITTI Velodyne: x forward, y left, z up; object coord in z is at bottom contact point.
# See KITTI paper. :contentReference[oaicite:1]{index=1}


def get_classes_from_cfg(cfg: Config):
    if hasattr(cfg, "metainfo") and cfg.metainfo and "classes" in cfg.metainfo:
        return list(cfg.metainfo["classes"])
    if hasattr(cfg, "class_names"):
        return list(cfg.class_names)
    try:
        ds_meta = cfg.test_dataloader["dataset"].get("metainfo", None)
        if ds_meta and "classes" in ds_meta:
            return list(ds_meta["classes"])
    except Exception:
        pass
    raise RuntimeError("Could not find class names in cfg (metainfo/classes).")


def to_tensor(x, dtype=None):
    if torch.is_tensor(x):
        t = x
    else:
        t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype)
    return t


def _ensure_nx7(t: torch.Tensor) -> torch.Tensor:
    t = to_tensor(t, dtype=torch.float32)
    if t.numel() == 0:
        return t.reshape(-1, 7)
    if t.ndim == 1:
        if t.numel() != 7:
            raise ValueError(f"Expected 7 numbers for one box, got {t.numel()}")
        return t.reshape(1, 7)
    if t.ndim == 2 and t.shape[1] == 7:
        return t
    raise ValueError(f"Expected shape (N,7) or (7,), got {tuple(t.shape)}")


def to_lidar_boxes(b, out_origin_z: float, assume_in_origin_z: float):
    """
    Convert bboxes_3d to LiDARInstance3DBoxes with requested origin_z.
    Only adjusts z to account for origin changes along height (dz).
    """
    out_origin = (0.5, 0.5, float(out_origin_z))

    def _convert_tensor(t, in_origin_z):
        t = _ensure_nx7(t)
        if t.numel() == 0:
            return LiDARInstance3DBoxes(t, box_dim=7, origin=out_origin)

        t = t.clone()
        dz = t[:, 5]
        # shift along z to match desired origin
        t[:, 2] = t[:, 2] + (out_origin[2] - float(in_origin_z)) * dz
        return LiDARInstance3DBoxes(t, box_dim=7, origin=out_origin)

    if isinstance(b, LiDARInstance3DBoxes):
        in_origin_z = getattr(b, "origin", (0.5, 0.5, assume_in_origin_z))[2]
        return _convert_tensor(b.tensor, in_origin_z)

    if isinstance(b, dict):
        t = b.get("tensor", None)
        if t is None:
            raise KeyError("bboxes_3d dict missing 'tensor'")
        in_origin_z = b.get("origin", (0.5, 0.5, assume_in_origin_z))[2]
        return _convert_tensor(t, in_origin_z)

    return _convert_tensor(b, assume_in_origin_z)


def to_instance_data(pred_instances_3d, out_origin_z: float, assume_in_origin_z: float, z_shift: float):
    if isinstance(pred_instances_3d, InstanceData):
        inst = pred_instances_3d
        # still normalize boxes origin if needed
        b = to_lidar_boxes(inst.bboxes_3d, out_origin_z=out_origin_z, assume_in_origin_z=assume_in_origin_z)
        if z_shift != 0.0 and b.tensor.numel() > 0:
            tt = b.tensor.clone()
            tt[:, 2] += float(z_shift)
            b = LiDARInstance3DBoxes(tt, box_dim=7, origin=b.origin)
        inst.bboxes_3d = b
        return inst

    if not isinstance(pred_instances_3d, dict):
        raise TypeError(f"pred_instances_3d must be dict or InstanceData, got {type(pred_instances_3d)}")

    bboxes = to_lidar_boxes(
        pred_instances_3d["bboxes_3d"],
        out_origin_z=out_origin_z,
        assume_in_origin_z=assume_in_origin_z,
    )

    if z_shift != 0.0 and bboxes.tensor.numel() > 0:
        tt = bboxes.tensor.clone()
        tt[:, 2] += float(z_shift)
        bboxes = LiDARInstance3DBoxes(tt, box_dim=7, origin=bboxes.origin)

    scores = to_tensor(pred_instances_3d["scores_3d"], dtype=torch.float32).reshape(-1)
    labels = to_tensor(pred_instances_3d["labels_3d"], dtype=torch.long).reshape(-1)

    return InstanceData(bboxes_3d=bboxes, scores_3d=scores, labels_3d=labels)


def load_info_list(path: str):
    obj = mmengine.load(path)
    if isinstance(obj, dict):
        if "data_list" in obj:
            return obj["data_list"]
        if "infos" in obj:
            return obj["infos"]
    if isinstance(obj, list):
        return obj
    raise TypeError(f"Unsupported ann_file type: {type(obj)}")


def extract_gt_from_info(sample, out_origin_z: float, assume_in_origin_z: float):
    """
    Try hard to extract GT boxes from a KITTI info entry.

    Supports:
    - old-style: sample['annos']['gt_bboxes_3d'] (or dict/tensor)
    - new-style: sample['instances'][i]['bbox_3d'] with labels in bbox_label_3d
    """
    # Old style
    if isinstance(sample, dict) and "annos" in sample:
        annos = sample["annos"]
        if isinstance(annos, dict) and "gt_bboxes_3d" in annos:
            gt = annos["gt_bboxes_3d"]
            gt_boxes = to_lidar_boxes(gt, out_origin_z=out_origin_z, assume_in_origin_z=assume_in_origin_z)
            gt_labels = None
            if "gt_labels_3d" in annos:
                gt_labels = to_tensor(annos["gt_labels_3d"], dtype=torch.long).reshape(-1)
            return gt_boxes, gt_labels

    # New style
    if isinstance(sample, dict) and "instances" in sample and isinstance(sample["instances"], list):
        boxes = []
        labels = []
        for inst in sample["instances"]:
            if not isinstance(inst, dict):
                continue
            if "bbox_3d" not in inst:
                continue
            boxes.append(inst["bbox_3d"])
            if "bbox_label_3d" in inst:
                labels.append(int(inst["bbox_label_3d"]))
            elif "label_3d" in inst:
                labels.append(int(inst["label_3d"]))
            else:
                labels.append(-1)

        if len(boxes) == 0:
            gt_t = torch.zeros((0, 7), dtype=torch.float32)
        else:
            gt_t = _ensure_nx7(torch.tensor(boxes, dtype=torch.float32))
        gt_boxes = LiDARInstance3DBoxes(gt_t, box_dim=7, origin=(0.5, 0.5, float(out_origin_z)))
        gt_labels = torch.tensor(labels, dtype=torch.long) if len(labels) > 0 else None
        return gt_boxes, gt_labels

    return None, None


def bottom_z(boxes: LiDARInstance3DBoxes) -> torch.Tensor:
    """
    Return z at bottom of box, independent of origin.
    For LiDARInstance3DBoxes: tensor[:,2] is at origin_z.
    bottom_z = z + (0.0 - origin_z) * dz
    """
    if boxes.tensor.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    origin_z = float(getattr(boxes, "origin", (0.5, 0.5, 0.0))[2])
    z = boxes.tensor[:, 2]
    dz = boxes.tensor[:, 5]
    return z + (0.0 - origin_z) * dz


def estimate_constant_z_shift(pred: InstanceData, gt_boxes: LiDARInstance3DBoxes, gt_labels: torch.Tensor, xy_gate: float = 2.0):
    """
    Heuristic: for each pred, find nearest GT of same class in XY (within xy_gate meters),
    compute (gt_bottom - pred_bottom), return median.
    """
    if pred.bboxes_3d.tensor.numel() == 0 or gt_boxes.tensor.numel() == 0 or gt_labels is None:
        return None

    p = pred.bboxes_3d.tensor
    g = gt_boxes.tensor
    pl = pred.labels_3d
    gl = gt_labels

    pb = bottom_z(pred.bboxes_3d)
    gb = bottom_z(gt_boxes)

    deltas = []
    for i in range(p.shape[0]):
        cls = int(pl[i].item())
        mask = (gl == cls)
        if mask.sum().item() == 0:
            continue
        gxy = g[mask][:, :2]
        pxy = p[i, :2].reshape(1, 2)
        d2 = ((gxy - pxy) ** 2).sum(dim=1)
        j = int(torch.argmin(d2).item())
        dist = float(torch.sqrt(d2[j]).item())
        if dist <= float(xy_gate):
            gb_j = float(gb[mask][j].item())
            pb_i = float(pb[i].item())
            deltas.append(gb_j - pb_i)

    if len(deltas) == 0:
        return None
    return float(np.median(np.array(deltas, dtype=np.float32)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Config with correct class list.")
    ap.add_argument("--pred-pkl", required=True, help="DumpResults-style list of preds.")
    ap.add_argument("--ann-file", required=True, help="KITTI info pkl, eg data/kitti/kitti_infos_val.pkl")
    ap.add_argument("--origin-z", type=float, default=0.0,
                    help="Target origin_z for LiDARInstance3DBoxes (0.0 bottom for KITTI).")
    ap.add_argument("--assume-in-origin-z", type=float, default=0.0,
                    help="If pred boxes are raw tensors, assume their origin_z is this.")
    ap.add_argument("--z-shift", type=float, default=0.0,
                    help="Add a constant shift (meters) to pred z AFTER origin normalization. Debug can estimate this.")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-n", type=int, default=3)
    ap.add_argument("--debug-xy-gate", type=float, default=2.0,
                    help="XY gating distance for z-shift estimation (meters).")
    args = ap.parse_args()

    cfg = Config.fromfile(args.cfg)
    classes = get_classes_from_cfg(cfg)

    preds = mmengine.load(args.pred_pkl)
    if not isinstance(preds, list):
        raise TypeError("Expected pred pkl to be a list (one entry per sample).")

    metric = KittiMetric(ann_file=args.ann_file, metric="bbox")
    metric.dataset_meta = {"classes": classes}

    info_list = None
    if args.debug:
        info_list = load_info_list(args.ann_file)

    packed = []
    if args.debug:
        print(f"\n=== DEBUG (first {min(args.debug_n, len(preds))} samples) ===")

    for i, item in enumerate(preds):
        if not isinstance(item, dict) or "pred_instances_3d" not in item:
            raise RuntimeError(f"Entry {i} missing pred_instances_3d (not DumpResults-style).")

        pi3d = to_instance_data(
            item["pred_instances_3d"],
            out_origin_z=args.origin_z,
            assume_in_origin_z=args.assume_in_origin_z,
            z_shift=args.z_shift,
        )

        if args.debug and i < args.debug_n:
            gt_boxes, gt_labels = extract_gt_from_info(
                info_list[i],
                out_origin_z=args.origin_z,
                assume_in_origin_z=args.assume_in_origin_z,
            )

            # pred stats
            pN = int(pi3d.bboxes_3d.tensor.shape[0])
            pin_origin_z = float(getattr(pi3d.bboxes_3d, "origin", (0.5, 0.5, args.assume_in_origin_z))[2])
            pz = pi3d.bboxes_3d.tensor[:, 2] if pN > 0 else torch.zeros((0,))
            ph = pi3d.bboxes_3d.tensor[:, 5] if pN > 0 else torch.zeros((0,))
            pbz = bottom_z(pi3d.bboxes_3d)

            print(f"[{i}] pred boxes: N={pN} origin_z={pin_origin_z}")
            if pN > 0:
                print(f"     pred z(origin): mean={pz.mean().item():.4f} min={pz.min().item():.4f} max={pz.max().item():.4f}")
                print(f"     pred dz(height): mean={ph.mean().item():.4f} min={ph.min().item():.4f} max={ph.max().item():.4f}")
                print(f"     pred bottom_z:  mean={pbz.mean().item():.4f} min={pbz.min().item():.4f} max={pbz.max().item():.4f}")
            else:
                print("     pred: empty")

            if gt_boxes is None:
                print("     gt: could not extract from ann_file entry (debug extraction failed)")
            else:
                gN = int(gt_boxes.tensor.shape[0])
                gz = gt_boxes.tensor[:, 2] if gN > 0 else torch.zeros((0,))
                gh = gt_boxes.tensor[:, 5] if gN > 0 else torch.zeros((0,))
                gbz = bottom_z(gt_boxes)

                print(f"     gt   boxes: N={gN} origin_z={float(getattr(gt_boxes,'origin',(0.5,0.5,0.0))[2])}")
                if gN > 0:
                    print(f"     gt   z(origin): mean={gz.mean().item():.4f} min={gz.min().item():.4f} max={gz.max().item():.4f}")
                    print(f"     gt   dz(height): mean={gh.mean().item():.4f} min={gh.min().item():.4f} max={gh.max().item():.4f}")
                    print(f"     gt   bottom_z:  mean={gbz.mean().item():.4f} min={gbz.min().item():.4f} max={gbz.max().item():.4f}")

                    est = estimate_constant_z_shift(pi3d, gt_boxes, gt_labels, xy_gate=args.debug_xy_gate) if gt_labels is not None else None
                    if est is not None:
                        print(f"     estimated constant z_shift (gt_bottom - pred_bottom) ~= {est:.4f} m  (xy_gate={args.debug_xy_gate} m)")
                else:
                    print("     gt: empty")

        packed.append({"sample_idx": i, "pred_instances_3d": pi3d})

    if args.debug:
        print("=== END DEBUG ===\n")

    results = metric.compute_metrics(packed)

    print("\n===== KITTI AP results (KittiMetric) =====")
    for k in sorted(results.keys()):
        print(f"{k}: {results[k]}")
    print("=========================================\n")


if __name__ == "__main__":
    main()
