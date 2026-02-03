# DAIR-V2X 2-Class (Car + Pedestrian) Training Guide

Complete workflow to train individual vehicle and infrastructure detectors on DAIR-V2X dataset with only Car and Pedestrian classes.

## Prerequisites

- DAIR-V2X dataset at: `/home/dellg16ssg/multi-robot-coordination/collaborative-perception-BEVP/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure/`
- Existing 3-class KITTI PKLs (already created)
- MMDetection3D environment active

## Step 0: Setup

```bash
cd ~/mmdetection3d
conda activate mmdet3d_pp
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

## Step 1: Backup and Create 2-Class Dataset PKLs

This will:
- Backup all 3-class PKLs with suffix `_3class_backup`
- Create new 2-class PKLs (Car + Pedestrian only) in place
- Filter out all Cyclist annotations

```bash
python tools/create_dairv2x_2class_dataset.py \
  --vehicle-root /home/dellg16ssg/multi-robot-coordination/collaborative-perception-BEVP/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure/vehicle-side \
  --infra-root /home/dellg16ssg/multi-robot-coordination/collaborative-perception-BEVP/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure/infrastructure-side \
  --old-classes Pedestrian Cyclist Car \
  --new-classes Pedestrian Car \
  --backup-suffix _3class_backup
```

### Dry run first (recommended):

```bash
python tools/create_dairv2x_2class_dataset.py \
  --vehicle-root /home/dellg16ssg/multi-robot-coordination/collaborative-perception-BEVP/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure/vehicle-side \
  --infra-root /home/dellg16ssg/multi-robot-coordination/collaborative-perception-BEVP/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure/infrastructure-side \
  --old-classes Pedestrian Cyclist Car \
  --new-classes Pedestrian Car \
  --backup-suffix _3class_backup \
  --dry-run
```

### To restore original 3-class PKLs later:

```bash
# Vehicle side
cd /home/dellg16ssg/multi-robot-coordination/collaborative-perception-BEVP/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure/vehicle-side
mv kitti_infos_train_3class_backup.pkl kitti_infos_train.pkl
mv kitti_infos_val_3class_backup.pkl kitti_infos_val.pkl
mv kitti_infos_trainval_3class_backup.pkl kitti_infos_trainval.pkl
mv kitti_dbinfos_train_3class_backup.pkl kitti_dbinfos_train.pkl

# Infrastructure side
cd /home/dellg16ssg/multi-robot-coordination/collaborative-perception-BEVP/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure/infrastructure-side
mv kitti_infos_train_3class_backup.pkl kitti_infos_train.pkl
mv kitti_infos_val_3class_backup.pkl kitti_infos_val.pkl
mv kitti_infos_trainval_3class_backup.pkl kitti_infos_trainval.pkl
mv kitti_dbinfos_train_3class_backup.pkl kitti_dbinfos_train.pkl
```

## Step 2: Train Vehicle-Side Detector (2-class)

```bash
cd ~/mmdetection3d

CFG_VEH=configs/_custom/pp_dairv2x_vehicle_2class.py

python tools/train.py "$CFG_VEH" \
  --work-dir work_dirs/pp_dairv2x_vehicle_2class \
  --cfg-options \
    train_cfg.max_epochs=80 \
    train_cfg.val_interval=2 \
    default_hooks.checkpoint.interval=2
```

### Quick test (1 epoch):

```bash
python tools/train.py "$CFG_VEH" \
  --work-dir work_dirs/pp_dairv2x_vehicle_2class_test \
  --cfg-options \
    train_cfg.max_epochs=1 \
    train_cfg.val_interval=1 \
    default_hooks.checkpoint.interval=1
```

### If you get CUDA OOM, reduce batch size:

```bash
python tools/train.py "$CFG_VEH" \
  --work-dir work_dirs/pp_dairv2x_vehicle_2class \
  --cfg-options \
    train_cfg.max_epochs=80 \
    train_cfg.val_interval=2 \
    default_hooks.checkpoint.interval=2 \
    train_dataloader.batch_size=2
```

## Step 3: Train Infrastructure-Side Detector (2-class)

```bash
cd ~/mmdetection3d

CFG_INF=configs/_custom/pp_dairv2x_infra_2class.py

python tools/train.py "$CFG_INF" \
  --work-dir work_dirs/pp_dairv2x_infra_2class \
  --cfg-options \
    train_cfg.max_epochs=80 \
    train_cfg.val_interval=2 \
    default_hooks.checkpoint.interval=2
```

### Quick test (1 epoch):

```bash
python tools/train.py "$CFG_INF" \
  --work-dir work_dirs/pp_dairv2x_infra_2class_test \
  --cfg-options \
    train_cfg.max_epochs=1 \
    train_cfg.val_interval=1 \
    default_hooks.checkpoint.interval=1
```

### If you get CUDA OOM, reduce batch size:

```bash
python tools/train.py "$CFG_INF" \
  --work-dir work_dirs/pp_dairv2x_infra_2class \
  --cfg-options \
    train_cfg.max_epochs=80 \
    train_cfg.val_interval=2 \
    default_hooks.checkpoint.interval=2 \
    train_dataloader.batch_size=2
```

## Step 4: Evaluate Vehicle Detector on Val Split

```bash
CFG_VEH=configs/_custom/pp_dairv2x_vehicle_2class.py
CKPT_VEH=work_dirs/pp_dairv2x_vehicle_2class/epoch_80.pth

python tools/test.py "$CFG_VEH" "$CKPT_VEH" --task lidar_det
```

## Step 5: Evaluate Infrastructure Detector on Val Split

```bash
CFG_INF=configs/_custom/pp_dairv2x_infra_2class.py
CKPT_INF=work_dirs/pp_dairv2x_infra_2class/epoch_80.pth

python tools/test.py "$CFG_INF" "$CKPT_INF" --task lidar_det
```

## Optional: Visualize Single Sample Predictions

### Vehicle side:

```bash
CFG_VEH=configs/_custom/pp_dairv2x_vehicle_2class.py
CKPT_VEH=work_dirs/pp_dairv2x_vehicle_2class/epoch_80.pth

python tools/vis_single_sample_pred_gt.py "$CFG_VEH" "$CKPT_VEH" \
  --sample-id 000289 \
  --score-thr 0.3 \
  --show \
  --out-dir work_dirs/pp_dairv2x_vehicle_2class/vis_single
```

### Infrastructure side:

```bash
CFG_INF=configs/_custom/pp_dairv2x_infra_2class.py
CKPT_INF=work_dirs/pp_dairv2x_infra_2class/epoch_80.pth

python tools/vis_single_sample_pred_gt.py "$CFG_INF" "$CKPT_INF" \
  --sample-id 007489 \
  --score-thr 0.3 \
  --show \
  --out-dir work_dirs/pp_dairv2x_infra_2class/vis_single
```

## Key Differences from Synthetic Dataset

| Aspect | Synthetic Dataset | DAIR-V2X Dataset |
|--------|------------------|------------------|
| **Classes** | 3 (Pedestrian, Cyclist, Car) | 2 (Pedestrian, Car) |
| **Vehicle data root** | `data/kitti/` | `/path/to/DAIR-V2X-C/.../vehicle-side/` |
| **Infra data root** | `data/kitti_infra/` | `/path/to/DAIR-V2X-C/.../infrastructure-side/` |
| **Point cloud range** | `[0, -39.68, -3, 92.16, 39.68, 1]` | Same |
| **Voxel size** | `[0.16, 0.16, 4]` | Same |
| **Anchor sizes** | 3 (Ped, Cyclist, Car) | 2 (Ped, Car) |
| **Training epochs** | 1 (test) or 80 (full) | Same |
| **DB sampler** | 3 classes | 2 classes |

## Config Files

- **Vehicle config**: `configs/_custom/pp_dairv2x_vehicle_2class.py`
- **Infra config**: `configs/_custom/pp_dairv2x_infra_2class.py`

Both configs are based on your synthetic dataset configs with:
- Updated `data_root` to DAIR-V2X paths
- 2 classes only (Pedestrian, Car)
- Same PointPillars architecture and hyperparameters
- Same training schedule

## Checkpoints

After training, checkpoints will be saved to:
- Vehicle: `work_dirs/pp_dairv2x_vehicle_2class/epoch_*.pth`
- Infra: `work_dirs/pp_dairv2x_infra_2class/epoch_*.pth`

## Next Steps: Late Fusion

After training both detectors, you can run late fusion evaluation similar to your synthetic workflow. The key difference is that DAIR-V2X vehicle/infra samples have different IDs, so you'll need to use the cooperative `data_info.json` to pair them correctly (like the sanity check script does).

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size with `train_dataloader.batch_size=2` or `=1`

### Issue: "File not found: kitti_infos_train.pkl"
**Solution**: Make sure you ran the 2-class PKL creation script (Step 1)

### Issue: "No objects found for class Cyclist"
**Solution**: This is expected - we filtered out Cyclist class. The warning is harmless.

### Issue: Want to switch back to 3-class
**Solution**: Restore backup PKLs (see Step 1) and use the original 3-class configs

## Quick Reference Commands

```bash
# Environment setup
cd ~/mmdetection3d
conda activate mmdet3d_pp
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Create 2-class PKLs (with backup)
python tools/create_dairv2x_2class_dataset.py \
  --vehicle-root /home/dellg16ssg/.../vehicle-side \
  --infra-root /home/dellg16ssg/.../infrastructure-side \
  --old-classes Pedestrian Cyclist Car \
  --new-classes Pedestrian Car

# Train vehicle (80 epochs)
python tools/train.py configs/_custom/pp_dairv2x_vehicle_2class.py \
  --work-dir work_dirs/pp_dairv2x_vehicle_2class \
  --cfg-options train_cfg.max_epochs=80

# Train infra (80 epochs)
python tools/train.py configs/_custom/pp_dairv2x_infra_2class.py \
  --work-dir work_dirs/pp_dairv2x_infra_2class \
  --cfg-options train_cfg.max_epochs=80

# Evaluate vehicle
python tools/test.py configs/_custom/pp_dairv2x_vehicle_2class.py \
  work_dirs/pp_dairv2x_vehicle_2class/epoch_80.pth --task lidar_det

# Evaluate infra
python tools/test.py configs/_custom/pp_dairv2x_infra_2class.py \
  work_dirs/pp_dairv2x_infra_2class/epoch_80.pth --task lidar_det
```
