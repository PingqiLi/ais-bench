# 自定义采样数据集使用指南

## 概述

`create_sampled_dataset.py` 工具可以从多个现有数据集中随机采样，生成用于快速评测的自定义数据集。

## 支持的数据集

### MCQ (Multiple Choice Questions - 选择题)
- **CEval**: 中文知识评测基准 (A, B, C, D 选项)
- **MMLU**: 大规模多任务语言理解 (A, B, C, D 选项)
- **GPQA**: 研究生级科学问题 (A, B, C, D 选项)

### QA (Question-Answer - 问答题)
- **AIME2024**: 美国数学竞赛题 (数值答案)
- **MATH500**: 数学问题求解 (数学解答)
- **LiveCodeBench**: 代码生成任务 (Python代码)

### 数据集准备说明

大部分数据集位于 `ais_bench/datasets/` 目录下。**LiveCodeBench** 需要单独下载：

```bash
# 进入数据集目录
cd ais_bench/datasets

# 下载LiveCodeBench (需要git-lfs)
git lfs install
git clone https://huggingface.co/datasets/livecodebench/code_generation_lite

# 验证下载成功
tree code_generation_lite/
# 应该看到: test.jsonl, test1.jsonl, test2.jsonl, etc.
```

## 重要限制

⚠️ **不同数据集需要不同的evaluator，必须分文件！**

### 原因1: MCQ和QA不能混合
AISBench读取第一行判断数据集类型（有A/B/C/D就是MCQ，否则是QA）

### 原因2: QA数据集的evaluator不同
- **AIME/MATH** → 需要 `MATHEvaluator` 提取 `\boxed{}` 答案
- **LiveCodeBench** → 需要代码执行evaluator
- 简单的 `AccEvaluator` 只能精确匹配，会导致数学题全错！

### 解决方案
工具自动生成**3个独立文件**：
- `*_mcq.jsonl` - CEval, MMLU, GPQA（选择题）
- `*_math_qa.jsonl` - AIME, MATH（数学题，需要MATHEvaluator）
- `*_code_qa.jsonl` - LiveCodeBench（代码题，需要代码执行）

## 使用方法

### 步骤1: 生成采样数据集

```bash
cd /path/to/ais_bench

# 生成自定义采样数据集
python3 tools/create_sampled_dataset.py \
  --aime-count 30 \
  --math-count 40 \
  --ceval-count 50 \
  --mmlu-count 50 \
  --gpqa-count 30 \
  --livecodebench-count 0 \
  --output datasets/custom_eval \
  --seed 42
```

**输出（3个文件）**：
```
datasets/custom_eval_mcq.jsonl       # 130个选择题
datasets/custom_eval_math_qa.jsonl   # 70个数学题
datasets/custom_eval_code_qa.jsonl   # 0个代码题（设为0可跳过）
```

### 步骤2: 为MATH-QA创建.meta.json（关键！）

MATH-QA数据集需要使用MATHEvaluator，必须创建meta文件：

```bash
# 从ais_bench根目录执行
cd /path/to/ais_bench

# 复制模板
cp tools/math_qa_meta_template.json datasets/custom_eval_math_qa.jsonl.meta.json
```

**meta.json内容**：
```json
{
  "evaluator": "ais_bench.benchmark.datasets.math.MATHEvaluator",
  "evaluator_kwargs": {"version": "v2"},
  "template": "Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer: {answer}"
}
```

⚠️ **注意**：meta.json文件名必须是 `<dataset-file-name>.meta.json` 格式！

### 步骤3: 运行评测

```bash
# 1. 评测MCQ数据集（选择题）
ais_bench \
  --models vllm_api_general_chat \
  --custom-dataset-path datasets/custom_eval_mcq.jsonl \
  --mode all \
  --work-dir outputs/custom_eval_mcq

# 2. 评测MATH-QA数据集（数学题 - 使用meta.json）
ais_bench \
  --models vllm_api_general_chat \
  --custom-dataset-path datasets/custom_eval_math_qa.jsonl \
  --custom-dataset-meta-path datasets/custom_eval_math_qa.jsonl.meta.json \
  --mode all \
  --work-dir outputs/custom_eval_math_qa
```

### 步骤4: 查看结果

```
outputs/custom_eval_mcq/results/       # MCQ评测结果
outputs/custom_eval_math_qa/results/   # MATH-QA评测结果（应该有正常准确率）
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
| `--seed` | 随机种子（可复现） | None |
| `--no-shuffle` | 不打乱数据顺序 | False |

## 数据集格式

### MCQ格式示例
```json
{
  "question": "以下哪个是Python的关键字？",
  "A": "print",
  "B": "if",
  "C": "input",
  "D": "len",
  "answer": "B",
  "source_dataset": "ceval",
  "subject": "computer_programming"
}
```

### QA格式示例
```json
{
  "question": "计算 2^10 的值",
  "answer": "1024",
  "source_dataset": "aime2024"
}
```

## 高级用法

### 仅生成MCQ数据集
```bash
python3 tools/create_sampled_dataset.py \
  --ceval-count 100 \
  --mmlu-count 100 \
  --gpqa-count 50 \
  --aime-count 0 \
  --math-count 0 \
  --livecodebench-count 0 \
  --output datasets/mcq_only
```

### 仅生成QA数据集
```bash
python3 tools/create_sampled_dataset.py \
  --aime-count 50 \
  --math-count 100 \
  --livecodebench-count 50 \
  --ceval-count 0 \
  --mmlu-count 0 \
  --gpqa-count 0 \
  --output datasets/qa_only
```

### 使用.meta.json进行额外配置

创建 `datasets/custom_eval_mcq.jsonl.meta.json`:
```json
{
  "request_count": 50,
  "sampling_mode": "random"
}
```

运行评测时指定meta文件：
```bash
ais_bench \
  --models vllm_api_general_chat \
  --custom-dataset-path datasets/custom_eval_mcq.jsonl \
  --custom-dataset-meta-path datasets/custom_eval_mcq.jsonl.meta.json \
  --mode all \
  --work-dir outputs/custom_eval_mcq
```

## 典型使用场景

### 快速验证W4A4 vs BF16量化效果

```bash
# 1. 生成小数据集 (每个数据集20个样本)
python3 tools/create_sampled_dataset.py \
  --aime-count 20 \
  --math-count 20 \
  --ceval-count 20 \
  --mmlu-count 20 \
  --output datasets/quick_val \
  --seed 42

# 2. 评测W4A4模型
ais_bench --models vllm_api_general_chat \
  --custom-dataset-path datasets/quick_val_mcq.jsonl \
  --mode all --work-dir outputs/w4a4_mcq

ais_bench --models vllm_api_general_chat \
  --custom-dataset-path datasets/quick_val_qa.jsonl \
  --mode all --work-dir outputs/w4a4_qa

# 3. 切换到BF16模型重新评测
# (修改vLLM启动参数，去掉 --quantization ascend)
# 然后重新运行上述ais_bench命令到不同的work-dir
```

### 调试评测框架

```bash
# 生成极小数据集用于快速测试
python3 tools/create_sampled_dataset.py \
  --aime-count 5 \
  --ceval-count 5 \
  --output datasets/debug_test \
  --seed 42

# 快速运行评测
ais_bench --models vllm_api_general_chat \
  --custom-dataset-path datasets/debug_test_qa.jsonl \
  --mode all --work-dir outputs/debug
```

## 常见问题

**Q: 为什么需要运行两次ais_bench？**

A: 因为AISBench通过读取数据集第一行来判断整个文件是MCQ还是QA类型，不支持混合类型。所以需要将MCQ和QA分成两个文件，分别评测。

**Q: 如何合并MCQ和QA的评测结果？**

A: 两次评测的结果保存在不同的work-dir中。如果需要统一分析，可以手动读取两个results目录中的JSON文件进行合并分析。

**Q: 可以只评测其中一种类型吗？**

A: 可以。设置不需要的数据集count为0即可。例如只需要MCQ数据集，就将`--aime-count 0 --math-count 0 --livecodebench-count 0`。

**Q: 如何确保采样的可复现性？**

A: 使用 `--seed` 参数指定随机种子，相同的seed会产生相同的采样结果。

## 技术细节

- 采样方式：使用Python `random.sample()` 进行无放回随机采样
- 字段标准化：自动将不同数据集的字段名统一为 `question` 和 `answer`
- 来源标记：每条数据自动添加 `source_dataset` 字段标记来源
- 数据混洗：默认打乱数据顺序（可用 `--no-shuffle` 关闭）
- 自动分类：根据是否有A、B、C、D字段自动分为MCQ和QA文件
