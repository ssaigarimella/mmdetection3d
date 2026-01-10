_base_ = ['./pp_vehicle_synth_3class.py']

test_dataloader = dict(
    dataset=dict(
        # base has data_root = 'data/kitti/' so keep this relative
        ann_file='kitti_infos_val.pkl',
    )
)

test_evaluator = dict(
    _delete_=True,
    type='DumpResults',
    out_file_path='work_dirs/pp_vehicle_synth_3class/val_preds_vehicle.pkl'
)
