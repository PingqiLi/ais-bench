#!/usr/bin/env python3
"""
Prepare CEval dataset for parallel evaluation.

This script splits the CEval dataset into N parts for parallel evaluation.
Each part contains every Nth sample (stride-based splitting).

Usage:
    python tools/prepare_parallel_ceval.py --num-splits 4
"""

import argparse
import csv
import os
import shutil
from pathlib import Path


def split_ceval_dataset(num_splits: int = 4):
    """Split CEval dataset into multiple parts for parallel evaluation."""

    # Paths
    script_dir = Path(__file__).parent
    ais_bench_root = script_dir.parent
    original_dataset_path = ais_bench_root / 'datasets' / 'ceval' / 'formal_ceval'

    if not original_dataset_path.exists():
        print(f"Error: Original dataset not found at {original_dataset_path}")
        return False

    print(f"Splitting CEval dataset into {num_splits} parts...")
    print(f"Original dataset: {original_dataset_path}")

    total_samples = 0
    split_samples = [0] * num_splits

    # Process each split (dev, val, test)
    for split_name in ['dev', 'val', 'test']:
        split_dir = original_dataset_path / split_name
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found, skipping")
            continue

        # Create output directories for each parallel instance
        output_dirs = []
        for i in range(num_splits):
            output_dir = ais_bench_root / 'datasets' / 'ceval' / f'parallel_ceval_{i}' / split_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dirs.append(output_dir)

        # Process each CSV file
        for csv_file in split_dir.glob('*.csv'):
            with open(csv_file, 'r', encoding='utf-8') as f_in:
                reader = csv.reader(f_in)
                header = next(reader)
                rows = list(reader)
                total_samples += len(rows)

                # Split rows across instances
                split_rows = [[] for _ in range(num_splits)]
                for idx, row in enumerate(rows):
                    split_idx = idx % num_splits
                    split_rows[split_idx].append(row)
                    split_samples[split_idx] += 1

                # Write split CSV files
                for i in range(num_splits):
                    output_file = output_dirs[i] / csv_file.name
                    with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
                        writer = csv.writer(f_out)
                        writer.writerow(header)
                        writer.writerows(split_rows[i])

    print(f"\n✓ Dataset splitting complete!")
    print(f"Total samples: {total_samples}")
    for i in range(num_splits):
        print(f"  Part {i}: {split_samples[i]} samples ({split_samples[i]/total_samples*100:.1f}%)")
        print(f"    → datasets/ceval/parallel_ceval_{i}/")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CEval dataset for parallel evaluation"
    )
    parser.add_argument(
        '--num-splits',
        type=int,
        default=4,
        help='Number of splits (default: 4)'
    )

    args = parser.parse_args()

    if args.num_splits < 2:
        print("Error: num-splits must be at least 2")
        return 1

    success = split_ceval_dataset(args.num_splits)
    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
