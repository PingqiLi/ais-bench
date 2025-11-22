#!/usr/bin/env python3
"""
Split a custom JSONL dataset for parallel evaluation.

This script splits a custom dataset into N parts for parallel evaluation
across multiple vLLM instances.

Usage:
    python3 tools/prepare_parallel_custom.py \
        --input datasets/custom_sampled_eval.jsonl \
        --num-splits 4
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], path: str):
    """Save data to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def split_dataset(input_path: str, num_splits: int = 4) -> bool:
    """Split custom dataset into N parts for parallel evaluation."""

    print(f"\n{'='*80}")
    print(f"Splitting Custom Dataset for Parallel Evaluation")
    print(f"{'='*80}\n")

    # Load input dataset
    print(f"Loading dataset: {input_path}")
    data = load_jsonl(input_path)
    total_samples = len(data)

    if total_samples == 0:
        print("❌ No data found in input file")
        return False

    print(f"Total samples: {total_samples}")
    print(f"Splitting into {num_splits} parts using stride-based method\n")

    # Prepare output paths
    input_file = Path(input_path)
    output_dir = input_file.parent
    base_name = input_file.stem  # e.g., "custom_sampled_eval"

    # Split data
    split_data = [[] for _ in range(num_splits)]
    for idx, item in enumerate(data):
        split_idx = idx % num_splits
        split_data[split_idx].append(item)

    # Save split files
    for i in range(num_splits):
        output_path = output_dir / f"{base_name}_parallel_{i}.jsonl"
        save_jsonl(split_data[i], str(output_path))

        percentage = len(split_data[i]) / total_samples * 100
        print(f"Part {i}: {len(split_data[i]):4d} samples ({percentage:5.1f}%) → {output_path}")

    print(f"\n{'='*80}")
    print(f"✓ Split complete!")
    print(f"{'='*80}\n")

    print("Next steps:")
    print("1. Create dataset configs for each split (or use --datasets with path)")
    print("2. Launch vLLM instances on different ports")
    print("3. Run ais_bench in parallel:")
    print()
    for i in range(num_splits):
        port = 8000 + i
        output_path = output_dir / f"{base_name}_parallel_{i}.jsonl"
        print(f"   # Instance {i}:")
        print(f"   ais_bench --models vllm_api_port_{port} \\")
        print(f"             --custom-dataset-path {output_path} \\")
        print(f"             --work-dir outputs/instance_{i} \\")
        print(f"             --mode all")
        print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Split custom dataset for parallel evaluation"
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--num-splits',
        type=int,
        default=4,
        help='Number of splits (default: 4)'
    )

    args = parser.parse_args()

    if args.num_splits < 2:
        print("❌ Error: num-splits must be at least 2")
        return 1

    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1

    success = split_dataset(args.input, args.num_splits)
    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
