# CMMLU
## 数据集简介
CMMLU是一套专门针对中文语言与文化背景设计的大模型综合能力评估体系，旨在系统检验语言模型在高级知识储备与推理能力上的表现。该评测涵盖67个学科主题，构建了从基础教育到专业进阶的完整知识体系，既包含物理、数学等需要计算能力的理科项目，也涉及人文社科等学科领域。由于语境和表述的特殊性，许多任务难以通过其他语言直接转译实现。此外，CMMLU中大量题目的答案具有鲜明的中国本土特征，其正确性在其他地区或语言体系中可能并不成立。

## 数据集原始获取链接
[https://huggingface.co/datasets/haonan-li/cmmlu](https://huggingface.co/datasets/haonan-li/cmmlu)

## 数据集内容格式(处理后)
### 文件结构
```
cmmlu
├── dev
│   ├── agronomy.csv
│   ├── anatomy.csv
│   ├── ancient_chinese.csv
│   ├── arts.csv
│   ├── astronomy.csv
│   ├── business_ethics.csv
│   ├── chinese_civil_service_exam.csv
│   ├── chinese_driving_rule.csv
│   ├── chinese_food_culture.csv
│   ├── chinese_foreign_policy.csv
│   ├── chinese_history.csv
│   ├── chinese_literature.csv
│   ├── chinese_teacher_qualification.csv
│   ├── clinical_knowledge.csv
│   ├── college_actuarial_science.csv
│   ├── college_education.csv
│   ├── college_engineering_hydrology.csv
│   ├── college_law.csv
│   ├── college_mathematics.csv
│   ├── college_medical_statistics.csv
│   ├── college_medicine.csv
│   ├── computer_science.csv
│   ├── computer_security.csv
│   ├── conceptual_physics.csv
│   ├── construction_project_management.csv
│   ├── economics.csv
│   ├── education.csv
│   ├── electrical_engineering.csv
│   ├── elementary_chinese.csv
│   ├── elementary_commonsense.csv
│   ├── elementary_information_and_technology.csv
│   ├── elementary_mathematics.csv
│   ├── ethnology.csv
│   ├── food_science.csv
│   ├── genetics.csv
│   ├── global_facts.csv
│   ├── high_school_biology.csv
│   ├── high_school_chemistry.csv
│   ├── high_school_geography.csv
│   ├── high_school_mathematics.csv
│   ├── high_school_physics.csv
│   ├── high_school_politics.csv
│   ├── human_sexuality.csv
│   ├── international_law.csv
│   ├── journalism.csv
│   ├── jurisprudence.csv
│   ├── legal_and_moral_basis.csv
│   ├── logical.csv
│   ├── machine_learning.csv
│   ├── management.csv
│   ├── marketing.csv
│   ├── marxist_theory.csv
│   ├── modern_chinese.csv
│   ├── nutrition.csv
│   ├── philosophy.csv
│   ├── professional_accounting.csv
│   ├── professional_law.csv
│   ├── professional_medicine.csv
│   ├── professional_psychology.csv
│   ├── public_relations.csv
│   ├── security_study.csv
│   ├── sociology.csv
│   ├── sports_science.csv
│   ├── traditional_chinese_medicine.csv
│   ├── virology.csv
│   ├── world_history.csv
│   └── world_religions.csv
└── test
    ├── agronomy.csv
    ├── anatomy.csv
    ├── ancient_chinese.csv
    ├── arts.csv
    ├── astronomy.csv
    ├── business_ethics.csv
    ├── chinese_civil_service_exam.csv
    ├── chinese_driving_rule.csv
    ├── chinese_food_culture.csv
    ├── chinese_foreign_policy.csv
    ├── chinese_history.csv
    ├── chinese_literature.csv
    ├── chinese_teacher_qualification.csv
    ├── clinical_knowledge.csv
    ├── college_actuarial_science.csv
    ├── college_education.csv
    ├── college_engineering_hydrology.csv
    ├── college_law.csv
    ├── college_mathematics.csv
    ├── college_medical_statistics.csv
    ├── college_medicine.csv
    ├── computer_science.csv
    ├── computer_security.csv
    ├── conceptual_physics.csv
    ├── construction_project_management.csv
    ├── economics.csv
    ├── education.csv
    ├── electrical_engineering.csv
    ├── elementary_chinese.csv
    ├── elementary_commonsense.csv
    ├── elementary_information_and_technology.csv
    ├── elementary_mathematics.csv
    ├── ethnology.csv
    ├── food_science.csv
    ├── genetics.csv
    ├── global_facts.csv
    ├── high_school_biology.csv
    ├── high_school_chemistry.csv
    ├── high_school_geography.csv
    ├── high_school_mathematics.csv
    ├── high_school_physics.csv
    ├── high_school_politics.csv
    ├── human_sexuality.csv
    ├── international_law.csv
    ├── journalism.csv
    ├── jurisprudence.csv
    ├── legal_and_moral_basis.csv
    ├── logical.csv
    ├── machine_learning.csv
    ├── management.csv
    ├── marketing.csv
    ├── marxist_theory.csv
    ├── modern_chinese.csv
    ├── nutrition.csv
    ├── philosophy.csv
    ├── professional_accounting.csv
    ├── professional_law.csv
    ├── professional_medicine.csv
    ├── professional_psychology.csv
    ├── public_relations.csv
    ├── security_study.csv
    ├── sociology.csv
    ├── sports_science.csv
    ├── traditional_chinese_medicine.csv
    ├── virology.csv
    ├── world_history.csv
    └── world_religions.csv
```
### 数据集内容样例格式
|task_id|prompt|test|
| ----- | ---- | ---- |
|HumanEval/0|from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "entry_point": "has_close_elements", "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n|\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n|

### 处理后的数据集获取链接
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/cmmlu.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/cmmlu.zip)

## 可用数据集任务
### cmmlu_gen_0_shot_cot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|cmmlu_gen_0_shot_cot_chat_prompt|CMMLU数据集生成式任务, prompt带逻辑链|accuracy|0-shot|对话格式|[cmmlu_gen_0_shot_cot_chat_prompt.py](cmmlu_gen_0_shot_cot_chat_prompt.py)|

#### 命令行调用
```shell
# 每个数据集文件作为单一task来评测
## 对话格式后端
ais_bench --models vllm_api_general_chat --datasets cmmlu_gen_0_shot_cot_chat_prompt

# 合并多个数据集文件统一评测
## 对话格式后端
ais_bench --models vllm_api_general_chat --datasets cmmlu_gen_0_shot_cot_chat_prompt --merge-ds
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.cmmlu.cmmlu_gen_0_shot_cot_chat_prompt import cmmlu_datasets
datasets = [
    *cmmlu_datasets,
]
```

### cmmlu_gen_5_shot_cot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|cmmlu_gen_5_shot_cot_chat_prompt|CMMLU数据集生成式任务, prompt带逻辑链|accuracy|5-shot|对话格式|[cmmlu_gen_5_shot_cot_chat_prompt.py](cmmlu_gen_5_shot_cot_chat_prompt.py)|

#### 命令行调用
```shell
# 每个数据集文件作为单一task来评测
## 对话格式后端
ais_bench --models vllm_api_general_chat --datasets cmmlu_gen_5_shot_cot_chat_prompt

# 合并多个数据集文件统一评测
## 对话格式后端
ais_bench --models vllm_api_general_chat --datasets cmmlu_gen_5_shot_cot_chat_prompt --merge-ds
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.cmmlu.cmmlu_gen_5_shot_cot_chat_prompt import cmmlu_datasets
datasets = [
    *cmmlu_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见配置文件的注释


