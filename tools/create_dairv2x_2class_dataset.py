#!/usr/bin/env python3
"""
Create DAIR-V2X 2-class (Car + Pedestrian) dataset PKLs from existing 3-class PKLs.

This script:
1. Backs up original 3-class PKLs
2. Creates new 2-class PKLs by filtering out Cyclist class
3. Preserves original dataset structure
"""

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np


def remap_labels(labels: np.ndarray, old_classes: List[str], new_classes: List[str]) -> np.ndarray:
    """
    Remap label indices from old class list to new class list.

    Args:
        labels: Original label indices
        old_classes: Original class names (e.g., ['Pedestrian', 'Cyclist', 'Car'])
        new_classes: New class names (e.g., ['Pedestrian', 'Car'])

    Returns:
        Remapped label indices, or -1 for classes not in new_classes
    """
    mapping = {}
    for old_idx, old_name in enumerate(old_classes):
        if old_name in new_classes:
            new_idx = new_classes.index(old_name)
            mapping[old_idx] = new_idx
        else:
            mapping[old_idx] = -1  # Mark for removal

    # Remap with bounds checking for unexpected labels (like -1 for DontCare)
    remapped = []
    for l in labels:
        l_int = int(l)
        if l_int in mapping:
            remapped.append(mapping[l_int])
        else:
            # Handle unexpected labels (e.g., -1 for DontCare) by keeping as -1
            remapped.append(-1)

    return np.array(remapped, dtype=labels.dtype)


def filter_info_by_classes(info_dict: Dict, old_classes: List[str], new_classes: List[str]) -> Dict:
    """
    Filter a single info dict to keep only specified classes.

    Args:
        info_dict: Single sample info dict
        old_classes: Original class names
        new_classes: Classes to keep

    Returns:
        Filtered info dict
    """
    if 'annos' not in info_dict or info_dict['annos'] is None:
        return info_dict

    annos = info_dict['annos']

    # Get original labels
    if 'gt_names' in annos:
        gt_names = annos['gt_names']

        # Find indices to keep
        keep_mask = np.array([name in new_classes for name in gt_names], dtype=bool)

        # Filter all annotation fields
        for key in annos.keys():
            if isinstance(annos[key], np.ndarray):
                if len(annos[key]) == len(gt_names):
                    annos[key] = annos[key][keep_mask]

        # Remap label indices if they exist
        # After filtering, labels might still have old indices (e.g., Car=2 in 3-class)
        # We need to remap them to new indices (e.g., Car=1 in 2-class)
        if 'gt_labels' in annos and len(annos['gt_labels']) > 0:
            annos['gt_labels'] = remap_labels(annos['gt_labels'], old_classes, new_classes)
        if 'gt_labels_3d' in annos and len(annos['gt_labels_3d']) > 0:
            annos['gt_labels_3d'] = remap_labels(annos['gt_labels_3d'], old_classes, new_classes)
        if 'labels' in annos and len(annos['labels']) > 0:
            annos['labels'] = remap_labels(annos['labels'], old_classes, new_classes)

        # Filter out any annotations with -1 labels (DontCare or filtered classes)
        # This is a safety check in case remapping produced -1 values
        valid_mask = None
        for label_key in ['gt_labels', 'gt_labels_3d', 'labels']:
            if label_key in annos and len(annos[label_key]) > 0:
                if valid_mask is None:
                    valid_mask = annos[label_key] >= 0
                else:
                    valid_mask = valid_mask & (annos[label_key] >= 0)

        if valid_mask is not None and not valid_mask.all():
            print(f"       Warning: Filtering out {(~valid_mask).sum()} annotations with invalid labels")
            # Apply valid_mask to all annotation fields
            for key in annos.keys():
                if isinstance(annos[key], np.ndarray):
                    if len(annos[key]) == len(valid_mask):
                        annos[key] = annos[key][valid_mask]

    return info_dict


def filter_dbinfos(dbinfos: Dict, new_classes: List[str]) -> Dict:
    """
    Filter dbinfos dict to keep only specified classes.

    Args:
        dbinfos: Database info dict with class names as keys
        new_classes: Classes to keep

    Returns:
        Filtered dbinfos dict
    """
    filtered = {}
    for cls in new_classes:
        if cls in dbinfos:
            filtered[cls] = dbinfos[cls]
    return filtered


def process_pkl(input_pkl: Path, output_pkl: Path, old_classes: List[str], new_classes: List[str], is_dbinfo: bool = False) -> None:
    """
    Process a single PKL file to create 2-class version.

    Args:
        input_pkl: Input PKL path
        output_pkl: Output PKL path
        old_classes: Original class names
        new_classes: Classes to keep
        is_dbinfo: Whether this is a dbinfo PKL (different structure)
    """
    print(f"[INFO] Processing: {input_pkl.name}")

    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)

    if is_dbinfo:
        # dbinfo structure: dict with class names as keys
        filtered_data = filter_dbinfos(data, new_classes)
        print(f"       Original classes: {list(data.keys())}")
        print(f"       New classes: {list(filtered_data.keys())}")
        for cls in new_classes:
            orig_count = len(data.get(cls, []))
            new_count = len(filtered_data.get(cls, []))
            print(f"       {cls}: {orig_count} objects")
    else:
        # info structure: dict with 'data_list' or 'infos' key
        if 'data_list' in data:
            info_list = data['data_list']
        elif 'infos' in data:
            info_list = data['infos']
        elif isinstance(data, list):
            info_list = data
        else:
            raise ValueError(f"Unknown PKL structure: {list(data.keys())}")

        print(f"       Original samples: {len(info_list)}")

        # Filter each sample
        filtered_list = []
        for info in info_list:
            filtered_info = filter_info_by_classes(info, old_classes, new_classes)
            # Only keep samples that still have annotations after filtering
            if 'annos' in filtered_info and filtered_info['annos'] is not None:
                if 'gt_names' in filtered_info['annos'] and len(filtered_info['annos']['gt_names']) > 0:
                    filtered_list.append(filtered_info)
            else:
                # Keep samples without annotations (for test set)
                filtered_list.append(filtered_info)

        print(f"       Filtered samples: {len(filtered_list)}")

        # Build categories dict for KITTI metric
        categories = {name: idx for idx, name in enumerate(new_classes)}

        if 'data_list' in data:
            filtered_data = {'data_list': filtered_list, 'metainfo': {'classes': new_classes, 'categories': categories}}
        elif 'infos' in data:
            filtered_data = {'infos': filtered_list, 'metadata': {'classes': new_classes, 'categories': categories}}
        else:
            filtered_data = filtered_list

    # Write output
    with open(output_pkl, 'wb') as f:
        pickle.dump(filtered_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"       Wrote: {output_pkl}")


def main():
    parser = argparse.ArgumentParser(description="Create 2-class DAIR-V2X dataset from 3-class")
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
        'kitti_dbinfos_train.pkl',
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
                    print(f"[BACKUP] {backup_pkl.name} already exists, skipping backup")

                # Create temporary output
                temp_output = side_root / f"{input_pkl.stem}_2class_temp{input_pkl.suffix}"

                # Process
                is_dbinfo = 'dbinfos' in pkl_name
                process_pkl(input_pkl, temp_output, args.old_classes, args.new_classes, is_dbinfo)

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
