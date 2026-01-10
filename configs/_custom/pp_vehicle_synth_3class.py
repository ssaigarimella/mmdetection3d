_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

# point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
# point_cloud_range = [-80, -80, -5, 80, 80, 5]
point_cloud_range = [0, -39.68, -3, 92.16, 39.68, 1]

# dataset settings
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
backend_args = None

# PointPillars adopted a different sampling strategies among classes
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

# PointPillars uses different augmentation hyper parameters
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
#     dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=True),  # disabled: Cyclist not present in dbinfos
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    dataset=dict(dataset=dict(pipeline=train_pipeline, metainfo=metainfo)))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001
epoch_num = 80
optim_wrapper = dict(
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        convert_to_iter_based=True)
]
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=2)
val_cfg = dict()
test_cfg = dict()

# -------------------------
# HARD OVERRIDE DATASET ROOTS (robust for nested mmengine merges)
# Put this at the very bottom of the config.
# -------------------------

ROOT = 'data/kitti/'       # vehicle config

def _find_first_dataset_cfg(node):
    """Return the first dict that looks like the actual dataset config."""
    if isinstance(node, dict):
        # Most reliable: explicit dataset type
        if node.get('type', None) in ('KittiDataset',):
            return node
        # Fallback: config dict that already has data_root/ann_file
        if 'data_root' in node and 'ann_file' in node:
            return node
        for v in node.values():
            hit = _find_first_dataset_cfg(v)
            if hit is not None:
                return hit
    elif isinstance(node, (list, tuple)):
        for v in node:
            hit = _find_first_dataset_cfg(v)
            if hit is not None:
                return hit
    return None

def _patch_one(dl, ann_file, pts_prefix='training/velodyne_reduced'):
    ds = _find_first_dataset_cfg(dl)
    if ds is None:
        raise RuntimeError(f'Could not locate dataset cfg inside dataloader: {dl.keys()}')

    ds['data_root'] = ROOT
    ds['ann_file'] = ann_file
    ds.setdefault('data_prefix', {})
    ds['data_prefix']['pts'] = pts_prefix
    ds['metainfo'] = metainfo

# Patch all three
_patch_one(train_dataloader, 'kitti_infos_train.pkl')
_patch_one(val_dataloader,   'kitti_infos_val.pkl')
_patch_one(test_dataloader,  'kitti_infos_val.pkl')

# Also keep db_sampler consistent (if you ever re-enable it)
db_sampler['data_root'] = ROOT
db_sampler['info_path'] = ROOT + 'kitti_dbinfos_train.pkl'
