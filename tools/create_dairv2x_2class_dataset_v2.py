#!/usr/bin/env python3
"""
Create DAIR-V2X 2-class (Car + Pedestrian) dataset PKLs from existing 3-class PKLs.

This script handles the NEW MMDet3D format with 'instances' lists.

This script:
1. Backs up original 3-class PKLs
2. Creates new 2-class PKLs by filtering out Cyclist class and remapping labels
3. Preserves original dataset structure
"""

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np


def remap_label(label: int, old_classes: List[str], new_classes: List[str]) -> int:
    """
    Remap a single label index from old class list to new class list.

    Args:
        label: Original label index
        old_classes: Original class names (e.g., ['Pedestrian', 'Cyclist', 'Car'])
        new_classes: New class names (e.g., ['Pedestrian', 'Car'])

    Returns:
        Remapped label index, or -1 if label should be removed
    """
    # Handle special values
    if label < 0:
        return -1  # Keep DontCare as -1

    if label >= len(old_classes):
        print(f"       Warning: Label {label} out of range for {old_classes}")
        return -1

    old_class_name = old_classes[label]
    if old_class_name in new_classes:
        return new_classes.index(old_class_name)
    else:
        return -1  # Filter out this class


def filter_instances(instances: List[Dict], old_classes: List[str], new_classes: List[str]) -> List[Dict]:
    """
    Filter instances list to keep only specified classes and remap labels.

    Args:
        instances: List of instance dicts
        old_classes: Original class names
        new_classes: Classes to keep

    Returns:
        Filtered and remapped instances list
    """
    filtered = []

    for inst in instances:
        # Remap bbox_label
        if 'bbox_label' in inst:
            new_label = remap_label(inst['bbox_label'], old_classes, new_classes)
            if new_label < 0:
                continue  # Skip this instance
            inst['bbox_label'] = new_label

        # Remap bbox_label_3d
        if 'bbox_label_3d' in inst:
            new_label_3d = remap_label(inst['bbox_label_3d'], old_classes, new_classes)
            if new_label_3d < 0:
                continue  # Skip this instance
            inst['bbox_label_3d'] = new_label_3d

        filtered.append(inst)

    return filtered


def process_pkl(input_pkl: Path, output_pkl: Path, old_classes: List[str], new_classes: List[str]) -> None:
    """
    Process a single PKL file to create 2-class version (new MMDet3D format).

    Args:
        input_pkl: Input PKL path
        output_pkl: Output PKL path
        old_classes: Original class names
        new_classes: Classes to keep
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

                filtered_instances = filter_instances(orig_instances, old_classes, new_classes)
                info['instances'] = filtered_instances

                total_filtered_instances += len(filtered_instances)
                if len(filtered_instances) > 0:
                    filtered_count += 1

        print(f"       Samples with annotations: {filtered_count}")
        print(f"       Total instances: {total_orig_instances} -> {total_filtered_instances}")

        # Update metainfo
        if 'metainfo' in data:
            data['metainfo']['classes'] = new_classes
            data['metainfo']['categories'] = {name: idx for idx, name in enumerate(new_classes)}

        filtered_data = data

    # Handle old MMDet3D format with infos (shouldn't happen for DAIR-V2X but keep for compatibility)
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
    parser = argparse.ArgumentParser(description="Create 2-class DAIR-V2X dataset from 3-class (new format)")
    parser.add_argument('--vehicle-root', type=str, required=True,
                        help='Path to vehicle-side directory')
    parser.add_argument('--infra-root', type=str, required=True,
                        help='Path to infrastructure-side directory')
    parser.add_argument('--backup-suffix', type=str, default='_3class_backup',
                        help='Suffix for backup files')
    parser.add_argument('--old-classes', nargs='+', default=['Pedestrian', 'Cyclist', 'Car'],
                        help='Original class names in order')
    parser.add_argument('--new-classes', nargs='+', default=['Pedestrian', 'Car'],
                        help='Classes to keep')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done without actually doing it')

    args = parser.parse_args()

    vehicle_root = Path(args.vehicle_root)
    infra_root = Path(args.infra_root)

    if not vehicle_root.exists():
        print(f"[ERROR] Vehicle root not found: {vehicle_root}")
        return
    if not infra_root.exists():
        print(f"[ERROR] Infra root not found: {infra_root}")
        return

    print(f"[INFO] Creating 2-class dataset")
    print(f"[INFO] Old classes: {args.old_classes}")
    print(f"[INFO] New classes: {args.new_classes}")
    print(f"[INFO] Backup suffix: {args.backup_suffix}")

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

                # IMPORTANT: Always process from backup, not from current file
                # (in case current file is already processed)
                source_pkl = backup_pkl if backup_pkl.exists() else input_pkl

                # Process
                process_pkl(source_pkl, temp_output, args.old_classes, args.new_classes)

                # Move temp to original location
                shutil.move(str(temp_output), str(input_pkl))
                print(f"[DONE] Updated {input_pkl.name} to 2-class version")

    print(f"\n{'='*60}")
    print("[DONE] All PKLs processed!")
    print(f"[INFO] Original 3-class PKLs backed up with suffix: {args.backup_suffix}")
    print(f"[INFO] To restore originals: mv *{args.backup_suffix}.pkl <original_name>.pkl")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
