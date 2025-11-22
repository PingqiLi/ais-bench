# Parallel Evaluation Support for AISBench

This branch (`quant_eval`) contains pre-configured files for parallel evaluation using multiple vLLM instances.

## Overview

For W4A4 quantization models that only support TP=1, we can achieve 4x speedup by running 4 independent vLLM instances in parallel on different NPUs, each processing a different portion of the dataset.

## Components

### 1. Dataset Configs (Pre-Split)

**Location**: `benchmark/configs/datasets/ceval/`

```
ceval_parallel_0_gen_0_shot_cot_chat_prompt.py  # Instance 0: samples 0, 4, 8, 12, ...
ceval_parallel_1_gen_0_shot_cot_chat_prompt.py  # Instance 1: samples 1, 5, 9, 13, ...
ceval_parallel_2_gen_0_shot_cot_chat_prompt.py  # Instance 2: samples 2, 6, 10, 14, ...
ceval_parallel_3_gen_0_shot_cot_chat_prompt.py  # Instance 3: samples 3, 7, 11, 15, ...
```

Each config points to a pre-split dataset directory:
- `ais_bench/datasets/ceval/parallel_ceval_0/`
- `ais_bench/datasets/ceval/parallel_ceval_1/`
- `ais_bench/datasets/ceval/parallel_ceval_2/`
- `ais_bench/datasets/ceval/parallel_ceval_3/`

### 2. Model Configs (Different Ports)

**Location**: `benchmark/configs/models/vllm_api/`

```
vllm_api_port_8000.py  # Instance 0 → connects to localhost:8000
vllm_api_port_8001.py  # Instance 1 → connects to localhost:8001
vllm_api_port_8002.py  # Instance 2 → connects to localhost:8002
vllm_api_port_8003.py  # Instance 3 → connects to localhost:8003
```

### 3. Dataset Preparation Tool

**Location**: `tools/prepare_parallel_ceval.py`

Splits the CEval dataset CSV files into N parts using stride-based splitting.

## Setup Instructions

### Step 1: Prepare Split Datasets

Run the preparation script to split the CEval dataset:

```bash
cd /path/to/ais_bench
python3 tools/prepare_parallel_ceval.py --num-splits 4
```

This creates:
```
datasets/ceval/
├── formal_ceval/          # Original dataset
├── parallel_ceval_0/      # Split for instance 0
│   ├── dev/
│   ├── val/
│   └── test/
├── parallel_ceval_1/      # Split for instance 1
├── parallel_ceval_2/      # Split for instance 2
└── parallel_ceval_3/      # Split for instance 3
```

### Step 2: Launch vLLM Instances

Launch 4 vLLM instances on different NPUs with different ports:

```bash
# Instance 0 (NPU 0, Port 8000)
ASCEND_RT_VISIBLE_DEVICES=0 vllm serve /path/to/model \
  --port 8000 --tensor-parallel-size 1 --max-model-len 32768 \
  --quantization ascend &

# Instance 1 (NPU 1, Port 8001)
ASCEND_RT_VISIBLE_DEVICES=1 vllm serve /path/to/model \
  --port 8001 --tensor-parallel-size 1 --max-model-len 32768 \
  --quantization ascend &

# Instance 2 (NPU 2, Port 8002)
ASCEND_RT_VISIBLE_DEVICES=2 vllm serve /path/to/model \
  --port 8002 --tensor-parallel-size 1 --max-model-len 32768 \
  --quantization ascend &

# Instance 3 (NPU 3, Port 8003)
ASCEND_RT_VISIBLE_DEVICES=3 vllm serve /path/to/model \
  --port 8003 --tensor-parallel-size 1 --max-model-len 32768 \
  --quantization ascend &
```

### Step 3: Run Parallel Evaluation

Run ais_bench in parallel (4 separate terminals or use screen/tmux):

```bash
# Terminal 1: Instance 0
ais_bench \
  --models vllm_api_port_8000 \
  --datasets ceval_parallel_0_gen_0_shot_cot_chat_prompt \
  --mode all \
  --work-dir outputs/instance_0 \
  --merge-ds

# Terminal 2: Instance 1
ais_bench \
  --models vllm_api_port_8001 \
  --datasets ceval_parallel_1_gen_0_shot_cot_chat_prompt \
  --mode all \
  --work-dir outputs/instance_1 \
  --merge-ds

# Terminal 3: Instance 2
ais_bench \
  --models vllm_api_port_8002 \
  --datasets ceval_parallel_2_gen_0_shot_cot_chat_prompt \
  --mode all \
  --work-dir outputs/instance_2 \
  --merge-ds

# Terminal 4: Instance 3
ais_bench \
  --models vllm_api_port_8003 \
  --datasets ceval_parallel_3_gen_0_shot_cot_chat_prompt \
  --mode all \
  --work-dir outputs/instance_3 \
  --merge-ds
```

### Step 4: Aggregate Results

After all instances complete, aggregate the results from:
- `outputs/instance_0/`
- `outputs/instance_1/`
- `outputs/instance_2/`
- `outputs/instance_3/`

## Advantages

✅ **No Runtime Config Patching** - All configs are pre-created and version controlled
✅ **No Race Conditions** - Each instance uses its own config files
✅ **Clean Separation** - Clear isolation between parallel instances
✅ **Reproducible** - Configs are checked into git
✅ **Maintainable** - Easy to modify ports, parameters, etc.

## File Structure

```
ais_bench/
├── benchmark/
│   ├── configs/
│   │   ├── datasets/
│   │   │   └── ceval/
│   │   │       ├── ceval_gen_0_shot_cot_chat_prompt.py  (original)
│   │   │       ├── ceval_parallel_0_gen_0_shot_cot_chat_prompt.py
│   │   │       ├── ceval_parallel_1_gen_0_shot_cot_chat_prompt.py
│   │   │       ├── ceval_parallel_2_gen_0_shot_cot_chat_prompt.py
│   │   │       └── ceval_parallel_3_gen_0_shot_cot_chat_prompt.py
│   │   └── models/
│   │       └── vllm_api/
│   │           ├── vllm_api_general_chat.py  (original)
│   │           ├── vllm_api_port_8000.py
│   │           ├── vllm_api_port_8001.py
│   │           ├── vllm_api_port_8002.py
│   │           └── vllm_api_port_8003.py
├── datasets/
│   └── ceval/
│       ├── formal_ceval/       (original, full dataset)
│       ├── parallel_ceval_0/   (1/4 of dataset)
│       ├── parallel_ceval_1/   (1/4 of dataset)
│       ├── parallel_ceval_2/   (1/4 of dataset)
│       └── parallel_ceval_3/   (1/4 of dataset)
└── tools/
    └── prepare_parallel_ceval.py
```

## Notes

- This approach works for any dataset, not just CEval
- Can easily extend to more/fewer instances by creating more configs
- Pre-splitting datasets avoids runtime overhead
- Each instance's results can be analyzed independently before aggregation

---

## Custom Sampled Datasets

### Creating a Custom Evaluation Dataset

For quick validation and testing, you can create a custom dataset by sampling from multiple existing datasets.

#### Available Tool: `tools/create_sampled_dataset.py`

**Features:**
- Sample from AIME2024, MATH500, CEval, MMLU, LiveCodeBench
- Customize sample count for each dataset
- Random sampling with optional seed for reproducibility
- Automatically standardizes field names
- Adds `source_dataset` tag to each sample

**Usage Example:**
```bash
# Create a custom dataset with 30 samples from each source
python3 tools/create_sampled_dataset.py \
  --aime-count 30 \
  --math-count 40 \
  --ceval-count 50 \
  --mmlu-count 50 \
  --livecodebench-count 30 \
  --output datasets/my_custom_eval.jsonl \
  --seed 42
```

**Output:**
```
datasets/my_custom_eval.jsonl  # Combined dataset (200 samples total)
```

#### Running Evaluation

**Single instance:**
```bash
ais_bench \
  --models vllm_api_general_chat \
  --datasets custom_sampled_eval_gen_0_shot_cot_chat_prompt \
  --mode all \
  --work-dir outputs/custom_eval
```

**Parallel evaluation:**
```bash
# 1. Split the custom dataset
python3 tools/prepare_parallel_custom.py \
  --input datasets/my_custom_eval.jsonl \
  --num-splits 4

# 2. Run in parallel (4 terminals)
ais_bench --models vllm_api_port_8000 --custom-dataset-path datasets/my_custom_eval_parallel_0.jsonl --work-dir outputs/instance_0 --mode all
ais_bench --models vllm_api_port_8001 --custom-dataset-path datasets/my_custom_eval_parallel_1.jsonl --work-dir outputs/instance_1 --mode all
ais_bench --models vllm_api_port_8002 --custom-dataset-path datasets/my_custom_eval_parallel_2.jsonl --work-dir outputs/instance_2 --mode all
ais_bench --models vllm_api_port_8003 --custom-dataset-path datasets/my_custom_eval_parallel_3.jsonl --work-dir outputs/instance_3 --mode all
```

### Dataset Config

**Location**: `benchmark/configs/datasets/custom/custom_sampled_eval_gen_0_shot_cot_chat_prompt.py`

This config can load any JSONL file created by the sampling tool.

### Use Cases

✅ **Quick validation** - Test W4A4 vs BF16 with small dataset (< 200 samples)
✅ **Framework debugging** - Test parallel evaluation with minimal data
✅ **Parameter tuning** - Quickly iterate on generation parameters
✅ **Benchmark comparison** - Create consistent test set across runs

