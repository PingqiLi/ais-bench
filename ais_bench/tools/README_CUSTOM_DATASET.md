# 自定义采样数据集使用指南

## 快速开始

### 步骤1: 生成数据集

```bash
cd /path/to/ais_bench

python3 tools/create_sampled_dataset.py \
  --aime-count 30 \
  --math-count 40 \
  --ceval-count 50 \
  --mmlu-count 50 \
  --gpqa-count 30 \
  --livecodebench-count 20 \
  --output datasets/custom_eval \
  --seed 42
```

**生成3个文件**：
```
datasets/custom_eval_mcq.jsonl       # CEval + MMLU + GPQA (选择题)
datasets/custom_eval_math_qa.jsonl   # AIME + MATH (数学题)
datasets/custom_eval_code_qa.jsonl   # LiveCodeBench (代码题)
```

### 步骤2: 创建 meta.json

⚠️ **QA数据集必须创建meta.json，否则准确率会接近0%**

```bash
# MATH-QA
cp tools/math_qa_meta_template.json ais_bench/datasets/custom_eval_math_qa.jsonl.meta.json

# Code-QA
cp tools/code_qa_meta_template.json ais_bench/datasets/custom_eval_code_qa.jsonl.meta.json
```

### 步骤3: 运行评测

⚠️ **路径说明**：可以使用相对路径（相对于项目根目录）或绝对路径

```bash
# MCQ (不需要meta.json)
ais_bench \
  --models vllm_api_general_chat \
  --custom-dataset-path ais_bench/datasets/custom_eval_mcq.jsonl \
  --mode all \
  --work-dir outputs/custom_eval_mcq

# MATH-QA (必须指定meta.json)
ais_bench \
  --models vllm_api_general_chat \
  --custom-dataset-path ais_bench/datasets/custom_eval_math_qa.jsonl \
  --custom-dataset-meta-path ais_bench/datasets/custom_eval_math_qa.jsonl.meta.json \
  --mode all \
  --work-dir outputs/custom_eval_math_qa

# Code-QA (必须指定meta.json)
ais_bench \
  --models vllm_api_general_chat \
  --custom-dataset-path ais_bench/datasets/custom_eval_code_qa.jsonl \
  --custom-dataset-meta-path ais_bench/datasets/custom_eval_code_qa.jsonl.meta.json \
  --mode all \
  --work-dir outputs/custom_eval_code_qa
```

## 支持的数据集

| 数据集 | 类型 | 说明 |
|--------|------|------|
| CEval | MCQ | 中文知识评测 (A/B/C/D选项) |
| MMLU | MCQ | 大规模多任务语言理解 (A/B/C/D选项) |
| GPQA | MCQ | 研究生级科学问题 (A/B/C/D选项) |
| AIME2024 | MATH-QA | 美国数学竞赛 (需要MATHEvaluator) |
| MATH500 | MATH-QA | 数学问题求解 (需要MATHEvaluator) |
| LiveCodeBench | Code-QA | 代码生成 (需要LCBCodeGenerationEvaluator) |

### 下载 LiveCodeBench

```bash
cd ais_bench/datasets
git lfs install
git clone https://huggingface.co/datasets/livecodebench/code_generation_lite
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--aime-count` | AIME2024采样数量 | 20 |
| `--math-count` | MATH500采样数量 | 20 |
| `--ceval-count` | CEval采样数量 | 20 |
| `--mmlu-count` | MMLU采样数量 | 20 |
| `--gpqa-count` | GPQA采样数量 | 0 |
| `--livecodebench-count` | LiveCodeBench采样数量 | 20 |
| `--output` | 输出文件路径前缀 | `datasets/custom_sampled_eval` |
| `--seed` | 随机种子 | None |
| `--no-shuffle` | 不打乱数据顺序 | False |

## 常见问题

### 为什么要分成3个文件？

不同数据集需要不同的evaluator：
- **MCQ**: 自动使用 `OptionSimAccEvaluator`
- **MATH-QA**: 需要 `MATHEvaluator` 提取 `\boxed{}` 答案
- **Code-QA**: 需要 `LCBCodeGenerationEvaluator` 执行代码

如果混在一起，无法为不同题目指定不同evaluator。

### 为什么准确率是0%？

**原因**: 忘记指定 `--custom-dataset-meta-path`，导致使用默认的 `AccEvaluator` (精确字符串匹配)。

**解决**: 必须为MATH-QA和Code-QA指定meta.json文件。
