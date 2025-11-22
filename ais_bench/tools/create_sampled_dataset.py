#!/usr/bin/env python3
"""
Create a custom sampled dataset from multiple existing datasets.

This script samples data from aime2024, math500, ceval, mmlu, and livecodebench
to create a unified evaluation dataset for quick testing and validation.

Usage:
    python3 tools/create_sampled_dataset.py \
        --aime-count 30 \
        --math-count 40 \
        --ceval-count 50 \
        --mmlu-count 50 \
        --livecodebench-count 30 \
        --output datasets/my_custom_eval.jsonl \
        --seed 42
"""

import argparse
import json
import random
import os
from pathlib import Path
from typing import List, Dict, Any


class DatasetSampler:
    """Sample data from multiple datasets and merge into a single dataset."""

    def __init__(self, seed: int = None):
        """Initialize sampler with optional random seed."""
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        # Get ais_bench root directory
        script_dir = Path(__file__).parent
        self.ais_bench_root = script_dir.parent

    def load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        full_path = self.ais_bench_root / path

        if not full_path.exists():
            print(f"Warning: {full_path} not found, skipping")
            return []

        with open(full_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        return data

    def sample_aime2024(self, count: int) -> List[Dict[str, Any]]:
        """Sample from AIME2024 dataset."""
        print(f"Sampling {count} from AIME2024...")
        data = self.load_jsonl('datasets/aime/aime.jsonl')

        if not data:
            return []

        samples = random.sample(data, min(count, len(data)))

        # Standardize fields
        for item in samples:
            item['source_dataset'] = 'aime2024'
            # AIME uses 'question' and 'answer' - already standard

        print(f"  ✓ Sampled {len(samples)} / {len(data)} items")
        return samples

    def sample_math500(self, count: int) -> List[Dict[str, Any]]:
        """Sample from MATH500 dataset."""
        print(f"Sampling {count} from MATH500...")
        # Try multiple possible paths for MATH dataset
        possible_paths = [
            'datasets/math/test_prm800k_500.jsonl',
            'datasets/math/test.jsonl',
            'datasets/math/prm800k_500.jsonl',
        ]

        data = []
        for path in possible_paths:
            data = self.load_jsonl(path)
            if data:
                break

        if not data:
            print(f"  ⚠ MATH500 dataset not found, skipping")
            return []

        samples = random.sample(data, min(count, len(data)))

        # Standardize fields
        for item in samples:
            item['source_dataset'] = 'math500'
            # MATH uses 'problem' and 'solution', map to standard names
            if 'problem' in item and 'question' not in item:
                item['question'] = item['problem']
            if 'solution' in item and 'answer' not in item:
                item['answer'] = item['solution']

        print(f"  ✓ Sampled {len(samples)} / {len(data)} items")
        return samples

    def sample_ceval(self, count: int) -> List[Dict[str, Any]]:
        """Sample from CEval dataset."""
        print(f"Sampling {count} from CEval...")

        # CEval has multiple subjects, sample from val split
        ceval_path = self.ais_bench_root / 'datasets' / 'ceval' / 'formal_ceval' / 'val'

        if not ceval_path.exists():
            print(f"  ⚠ CEval dataset not found at {ceval_path}, skipping")
            return []

        # Load all CSV files and combine
        all_data = []
        import csv

        for csv_file in ceval_path.glob('*.csv'):
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row['subject'] = csv_file.stem.replace('_val', '')
                    all_data.append(row)

        if not all_data:
            print(f"  ⚠ No CEval data loaded, skipping")
            return []

        samples = random.sample(all_data, min(count, len(all_data)))

        # Standardize fields
        for item in samples:
            item['source_dataset'] = 'ceval'
            # CEval already has question, A, B, C, D, answer

        print(f"  ✓ Sampled {len(samples)} / {len(all_data)} items")
        return samples

    def sample_mmlu(self, count: int) -> List[Dict[str, Any]]:
        """Sample from MMLU dataset."""
        print(f"Sampling {count} from MMLU...")

        # MMLU has multiple subjects
        mmlu_path = self.ais_bench_root / 'datasets' / 'mmlu'

        if not mmlu_path.exists():
            print(f"  ⚠ MMLU dataset not found at {mmlu_path}, skipping")
            return []

        # Try to load from test or val
        all_data = []
        import csv

        for split in ['test', 'val']:
            split_path = mmlu_path / split
            if not split_path.exists():
                continue

            for csv_file in split_path.glob('*.csv'):
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 6:  # question, A, B, C, D, answer
                            item = {
                                'question': row[0],
                                'A': row[1],
                                'B': row[2],
                                'C': row[3],
                                'D': row[4],
                                'answer': row[5],
                                'subject': csv_file.stem.replace('_test', '').replace('_val', '')
                            }
                            all_data.append(item)

        if not all_data:
            print(f"  ⚠ No MMLU data loaded, skipping")
            return []

        samples = random.sample(all_data, min(count, len(all_data)))

        # Add source tag
        for item in samples:
            item['source_dataset'] = 'mmlu'

        print(f"  ✓ Sampled {len(samples)} / {len(all_data)} items")
        return samples

    def sample_livecodebench(self, count: int) -> List[Dict[str, Any]]:
        """Sample from LiveCodeBench dataset."""
        print(f"Sampling {count} from LiveCodeBench...")

        possible_paths = [
            'datasets/livecodebench/code_generation_lite.jsonl',
            'datasets/livecodebench/test.jsonl',
        ]

        data = []
        for path in possible_paths:
            data = self.load_jsonl(path)
            if data:
                break

        if not data:
            print(f"  ⚠ LiveCodeBench dataset not found, skipping")
            return []

        samples = random.sample(data, min(count, len(data)))

        # Standardize fields
        for item in samples:
            item['source_dataset'] = 'livecodebench'
            # LiveCodeBench might use 'prompt' instead of 'question'
            if 'prompt' in item and 'question' not in item:
                item['question'] = item['prompt']

        print(f"  ✓ Sampled {len(samples)} / {len(data)} items")
        return samples

    def create_dataset(self,
                      aime_count: int = 20,
                      math_count: int = 20,
                      ceval_count: int = 20,
                      mmlu_count: int = 20,
                      livecodebench_count: int = 20,
                      shuffle: bool = True) -> List[Dict[str, Any]]:
        """Create combined dataset from multiple sources."""

        print("\n" + "="*80)
        print("Creating Custom Sampled Dataset")
        print("="*80 + "\n")

        all_samples = []

        # Sample from each dataset
        if aime_count > 0:
            all_samples.extend(self.sample_aime2024(aime_count))

        if math_count > 0:
            all_samples.extend(self.sample_math500(math_count))

        if ceval_count > 0:
            all_samples.extend(self.sample_ceval(ceval_count))

        if mmlu_count > 0:
            all_samples.extend(self.sample_mmlu(mmlu_count))

        if livecodebench_count > 0:
            all_samples.extend(self.sample_livecodebench(livecodebench_count))

        # Shuffle if requested
        if shuffle:
            random.shuffle(all_samples)
            print(f"\n✓ Shuffled combined dataset")

        return all_samples

    def save_dataset(self, data: List[Dict[str, Any]], output_path: str):
        """Save dataset to JSONL file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"\n{'='*80}")
        print(f"✓ Saved {len(data)} samples to: {output_file}")
        print(f"{'='*80}\n")

        # Print statistics
        source_counts = {}
        for item in data:
            source = item.get('source_dataset', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1

        print("Dataset composition:")
        for source, count in sorted(source_counts.items()):
            percentage = count / len(data) * 100
            print(f"  {source:20s}: {count:4d} samples ({percentage:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Create custom sampled dataset from multiple sources"
    )

    parser.add_argument(
        '--aime-count',
        type=int,
        default=20,
        help='Number of samples from AIME2024 (default: 20)'
    )
    parser.add_argument(
        '--math-count',
        type=int,
        default=20,
        help='Number of samples from MATH500 (default: 20)'
    )
    parser.add_argument(
        '--ceval-count',
        type=int,
        default=20,
        help='Number of samples from CEval (default: 20)'
    )
    parser.add_argument(
        '--mmlu-count',
        type=int,
        default=20,
        help='Number of samples from MMLU (default: 20)'
    )
    parser.add_argument(
        '--livecodebench-count',
        type=int,
        default=20,
        help='Number of samples from LiveCodeBench (default: 20)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='datasets/custom_sampled_eval.jsonl',
        help='Output file path (default: datasets/custom_sampled_eval.jsonl)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Do not shuffle the final dataset'
    )

    args = parser.parse_args()

    # Create sampler
    sampler = DatasetSampler(seed=args.seed)

    # Create dataset
    dataset = sampler.create_dataset(
        aime_count=args.aime_count,
        math_count=args.math_count,
        ceval_count=args.ceval_count,
        mmlu_count=args.mmlu_count,
        livecodebench_count=args.livecodebench_count,
        shuffle=not args.no_shuffle
    )

    # Save dataset
    if dataset:
        sampler.save_dataset(dataset, args.output)
        return 0
    else:
        print("\n❌ No data was sampled. Please check dataset paths.")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
