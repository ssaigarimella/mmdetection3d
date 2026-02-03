#!/usr/bin/env python3
"""
Create DAIR-V2X 2-class (Car + Pedestrian) dataset PKLs from existing 3-class PKLs.

IMPORTANT: This version PRESERVES original KITTI label indices to work with MMDet3D:
- Pedestrian: index 0 (unchanged)
- Cyclist: index 1 (REMOVED, not remapped)
- Car: index 2 (unchanged)

This is required because KittiDataset has hardcoded METAINFO with these indices.
"""

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np


def filter_instances_keep_indices(instances: List[Dict], classes_to_keep: List[str],
                                    all_classes: List[str]) -> List[Dict]:
    """
    Filter instances to keep only specified classes WITHOUT remapping label indices.

    Args:
        instances: List of instance dicts
        classes_to_keep: Class names to keep (e.g., ['Pedestrian', 'Car'])
        all_classes: All class names with original indices (e.g., ['Pedestrian', 'Cyclist', 'Car'])

    Returns:
        Filtered instances list with original label indices preserved
    """
    # Build set of valid indices
    valid_indices = {all_classes.index(name) for name in classes_to_keep if name in all_classes}

    filtered = []
    for inst in instances:
        # Check both bbox_label and bbox_label_3d
        label_3d = inst.get('bbox_label_3d', -1)

        # Keep only if label is in our valid set
        if label_3d in valid_indices:
            filtered.append(inst)

    return filtered


def process_pkl(input_pkl: Path, output_pkl: Path, classes_to_keep: List[str],
                all_classes: List[str]) -> None:
    """
    Process a single PKL file to create 2-class version (new MMDet3D format).
    PRESERVES original label indices.

    Args:
        input_pkl: Input PKL path
        output_pkl: Output PKL path
        classes_to_keep: Classes to keep (e.g., ['Pedestrian', 'Car'])
        all_classes: Original class names in order (e.g., ['Pedestrian', 'Cyclist', 'Car'])
    """
    print(f"[INFO] Processing: {input_pkl.name}")

    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)

    # Handle new MMDet3D format with data_list
    if 'data_list' in data:
        info_list = data['data_list']
        print(f"       Original samples: {len(info_list)}")

        # Filter each sample's instances
        filtered_count = 0
        total_orig_instances = 0
        total_filtered_instances = 0

        for info in info_list:
            if 'instances' in info:
                orig_instances = info['instances']
                total_orig_instances += len(orig_instances)

                filtered_instances = filter_instances_keep_indices(
                    orig_instances, classes_to_keep, all_classes)
                info['instances'] = filtered_instances

                total_filtered_instances += len(filtered_instances)
                if len(filtered_instances) > 0:
                    filtered_count += 1

        print(f"       Samples with annotations: {filtered_count}")
        print(f"       Total instances: {total_orig_instances} -> {total_filtered_instances}")

        # Update metainfo - classes list has filtered classes but maintains original indices
        if 'metainfo' in data:
            # IMPORTANT: metainfo classes should match what you pass to model config
            # But KITTI dataset will use its hardcoded METAINFO internally
            data['metainfo']['classes'] = classes_to_keep
            data['metainfo']['categories'] = {name: all_classes.index(name)
                                             for name in classes_to_keep
                                             if name in all_classes}

        filtered_data = data

    # Handle old MMDet3D format with infos
    elif 'infos' in data:
        print(f"       Warning: Old format detected, this script is designed for new format")
        filtered_data = data

    else:
        raise ValueError(f"Unknown PKL structure: {list(data.keys())}")

    # Write output
    with open(output_pkl, 'wb') as f:
        pickle.dump(filtered_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"       Wrote: {output_pkl}")


def main():
    parser = argparse.ArgumentParser(
        description="Create 2-class DAIR-V2X dataset from 3-class (PRESERVES label indices)")
    parser.add_argument('--vehicle-root', type=str, required=True,
                        help='Path to vehicle-side directory')
    parser.add_argument('--infra-root', type=str, required=True,
                        help='Path to infrastructure-side directory')
    parser.add_argument('--backup-suffix', type=str, default='_3class_backup',
                        help='Suffix for backup files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done without actually doing it')

    args = parser.parse_args()

    # IMPORTANT: Original KITTI class order
    all_classes = ['Pedestrian', 'Cyclist', 'Car']
    classes_to_keep = ['Pedestrian', 'Car']

    vehicle_root = Path(args.vehicle_root)
    infra_root = Path(args.infra_root)

    if not vehicle_root.exists():
        print(f"[ERROR] Vehicle root not found: {vehicle_root}")
        return
    if not infra_root.exists():
        print(f"[ERROR] Infra root not found: {infra_root}")
        return

    print(f"[INFO] Creating 2-class dataset (PRESERVING label indices)")
    print(f"[INFO] All classes (original KITTI order): {all_classes}")
    print(f"[INFO] Classes to keep: {classes_to_keep}")
    print(f"[INFO] Backup suffix: {args.backup_suffix}")
    print(f"[INFO] Label indices will be preserved: Pedestrian=0, Car=2")

    # PKL files to process
    pkl_files = [
        'kitti_infos_train.pkl',
        'kitti_infos_val.pkl',
        'kitti_infos_trainval.pkl',
        'kitti_infos_test.pkl',
    ]

    for side_name, side_root in [('vehicle', vehicle_root), ('infrastructure', infra_root)]:
        print(f"\n{'='*60}")
        print(f"Processing {side_name} side: {side_root}")
        print(f"{'='*60}")

        for pkl_name in pkl_files:
            input_pkl = side_root / pkl_name

            if not input_pkl.exists():
                print(f"[SKIP] {pkl_name} not found")
                continue

            # Backup original
            backup_pkl = side_root / f"{input_pkl.stem}{args.backup_suffix}{input_pkl.suffix}"

            if args.dry_run:
                print(f"[DRY-RUN] Would backup: {input_pkl} -> {backup_pkl}")
                print(f"[DRY-RUN] Would create: {input_pkl} (2-class version)")
            else:
                if not backup_pkl.exists():
                    print(f"[BACKUP] {input_pkl.name} -> {backup_pkl.name}")
                    shutil.copy2(input_pkl, backup_pkl)
                else:
                    print(f"[BACKUP] {backup_pkl.name} already exists, using it as source")

                # Create temporary output
                temp_output = side_root / f"{input_pkl.stem}_2class_temp{input_pkl.suffix}"

                # IMPORTANT: Always process from backup
                source_pkl = backup_pkl if backup_pkl.exists() else input_pkl

                # Process
                process_pkl(source_pkl, temp_output, classes_to_keep, all_classes)

                # Move temp to original location
                shutil.move(str(temp_output), str(input_pkl))
                print(f"[DONE] Updated {input_pkl.name} to 2-class version")

    print(f"\n{'='*60}")
    print("[DONE] All PKLs processed!")
    print(f"[INFO] Original 3-class PKLs backed up with suffix: {args.backup_suffix}")
    print(f"[INFO] Label indices preserved: Pedestrian=0, Car=2 (Cyclist=1 removed)")
    print(f"[INFO] To restore originals: mv *{args.backup_suffix}.pkl <original_name>.pkl")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
