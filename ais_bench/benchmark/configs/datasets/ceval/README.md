# C-Eval
## 数据集简介
C-Eval 是一个针对基础模型的综合性中文评估套件。它包含 13948 道多项选择题，涵盖 52 个不同学科以及四个难度等级。

## 数据集原始获取链接
[https://github.com/SJTU-LIT/ceval#data](https://github.com/SJTU-LIT/ceval#data)

## 数据集内容格式(处理后)
### 文件结构
```
ceval/formal_ceval
├── dev
│   ├── accountant_dev.csv
│   ├── advanced_mathematics_dev.csv
│   ├── art_studies_dev.csv
│   ├── basic_medicine_dev.csv
│   ├── business_administration_dev.csv
│   ├── chinese_language_and_literature_dev.csv
│   ├── civil_servant_dev.csv
│   ├── clinical_medicine_dev.csv
│   ├── college_chemistry_dev.csv
│   ├── college_economics_dev.csv
│   ├── college_physics_dev.csv
│   ├── college_programming_dev.csv
│   ├── computer_architecture_dev.csv
│   ├── computer_network_dev.csv
│   ├── discrete_mathematics_dev.csv
│   ├── education_science_dev.csv
│   ├── electrical_engineer_dev.csv
│   ├── environmental_impact_assessment_engineer_dev.csv
│   ├── fire_engineer_dev.csv
│   ├── high_school_biology_dev.csv
│   ├── high_school_chemistry_dev.csv
│   ├── high_school_chinese_dev.csv
│   ├── high_school_geography_dev.csv
│   ├── high_school_history_dev.csv
│   ├── high_school_mathematics_dev.csv
│   ├── high_school_physics_dev.csv
│   ├── high_school_politics_dev.csv
│   ├── ideological_and_moral_cultivation_dev.csv
│   ├── law_dev.csv
│   ├── legal_professional_dev.csv
│   ├── logic_dev.csv
│   ├── mao_zedong_thought_dev.csv
│   ├── marxism_dev.csv
│   ├── metrology_engineer_dev.csv
│   ├── middle_school_biology_dev.csv
│   ├── middle_school_chemistry_dev.csv
│   ├── middle_school_geography_dev.csv
│   ├── middle_school_history_dev.csv
│   ├── middle_school_mathematics_dev.csv
│   ├── middle_school_physics_dev.csv
│   ├── middle_school_politics_dev.csv
│   ├── modern_chinese_history_dev.csv
│   ├── operating_system_dev.csv
│   ├── physician_dev.csv
│   ├── plant_protection_dev.csv
│   ├── probability_and_statistics_dev.csv
│   ├── professional_tour_guide_dev.csv
│   ├── sports_science_dev.csv
│   ├── tax_accountant_dev.csv
│   ├── teacher_qualification_dev.csv
│   ├── urban_and_rural_planner_dev.csv
│   └── veterinary_medicine_dev.csv
├── test
│   ├── accountant_test.csv
│   ├── advanced_mathematics_test.csv
│   ├── art_studies_test.csv
│   ├── basic_medicine_test.csv
│   ├── business_administration_test.csv
│   ├── chinese_language_and_literature_test.csv
│   ├── civil_servant_test.csv
│   ├── clinical_medicine_test.csv
│   ├── college_chemistry_test.csv
│   ├── college_economics_test.csv
│   ├── college_physics_test.csv
│   ├── college_programming_test.csv
│   ├── computer_architecture_test.csv
│   ├── computer_network_test.csv
│   ├── discrete_mathematics_test.csv
│   ├── education_science_test.csv
│   ├── electrical_engineer_test.csv
│   ├── environmental_impact_assessment_engineer_test.csv
│   ├── fire_engineer_test.csv
│   ├── high_school_biology_test.csv
│   ├── high_school_chemistry_test.csv
│   ├── high_school_chinese_test.csv
│   ├── high_school_geography_test.csv
│   ├── high_school_history_test.csv
│   ├── high_school_mathematics_test.csv
│   ├── high_school_physics_test.csv
│   ├── high_school_politics_test.csv
│   ├── ideological_and_moral_cultivation_test.csv
│   ├── law_test.csv
│   ├── legal_professional_test.csv
│   ├── logic_test.csv
│   ├── mao_zedong_thought_test.csv
│   ├── marxism_test.csv
│   ├── metrology_engineer_test.csv
│   ├── middle_school_biology_test.csv
│   ├── middle_school_chemistry_test.csv
│   ├── middle_school_geography_test.csv
│   ├── middle_school_history_test.csv
│   ├── middle_school_mathematics_test.csv
│   ├── middle_school_physics_test.csv
│   ├── middle_school_politics_test.csv
│   ├── modern_chinese_history_test.csv
│   ├── operating_system_test.csv
│   ├── physician_test.csv
│   ├── plant_protection_test.csv
│   ├── probability_and_statistics_test.csv
│   ├── professional_tour_guide_test.csv
│   ├── sports_science_test.csv
│   ├── tax_accountant_test.csv
│   ├── teacher_qualification_test.csv
│   ├── urban_and_rural_planner_test.csv
│   └── veterinary_medicine_test.csv
└── val
    ├── accountant_val.csv
    ├── advanced_mathematics_val.csv
    ├── art_studies_val.csv
    ├── basic_medicine_val.csv
    ├── business_administration_val.csv
    ├── chinese_language_and_literature_val.csv
    ├── civil_servant_val.csv
    ├── clinical_medicine_val.csv
    ├── college_chemistry_val.csv
    ├── college_economics_val.csv
    ├── college_physics_val.csv
    ├── college_programming_val.csv
    ├── computer_architecture_val.csv
    ├── computer_network_val.csv
    ├── discrete_mathematics_val.csv
    ├── education_science_val.csv
    ├── electrical_engineer_val.csv
    ├── environmental_impact_assessment_engineer_val.csv
    ├── fire_engineer_val.csv
    ├── high_school_biology_val.csv
    ├── high_school_chemistry_val.csv
    ├── high_school_chinese_val.csv
    ├── high_school_geography_val.csv
    ├── high_school_history_val.csv
    ├── high_school_mathematics_val.csv
    ├── high_school_physics_val.csv
    ├── high_school_politics_val.csv
    ├── ideological_and_moral_cultivation_val.csv
    ├── law_val.csv
    ├── legal_professional_val.csv
    ├── logic_val.csv
    ├── mao_zedong_thought_val.csv
    ├── marxism_val.csv
    ├── metrology_engineer_val.csv
    ├── middle_school_biology_val.csv
    ├── middle_school_chemistry_val.csv
    ├── middle_school_geography_val.csv
    ├── middle_school_history_val.csv
    ├── middle_school_mathematics_val.csv
    ├── middle_school_physics_val.csv
    ├── middle_school_politics_val.csv
    ├── modern_chinese_history_val.csv
    ├── operating_system_val.csv
    ├── physician_val.csv
    ├── plant_protection_val.csv
    ├── probability_and_statistics_val.csv
    ├── professional_tour_guide_val.csv
    ├── sports_science_val.csv
    ├── tax_accountant_val.csv
    ├── teacher_qualification_val.csv
    ├── urban_and_rural_planner_val.csv
    └── veterinary_medicine_val.csv
```
### 数据集内容样例格式
| id | question                  | A                                       | B                                                           | C                       | D              |
|:---:|:-------------------------:|:---------------------------------------:|:-----------------------------------------------------------:|:-----------------------:|:--------------:|
| 0  | 下列关于资本结构理论的说法中，不正确的是____。 | 代理理论、权衡理论、有企业所得税条件下的MM理论，都认为企业价值与资本结构有关 | 按照优序融资理论的观点，考虑信息不对称和逆向选择的影响，管理者偏好首选留存收益筹资，然后是发行新股筹资，最后是债务筹资 | 权衡理论是对有企业所得税条件下的MM理论的扩展 | 代理理论是对权衡理论的扩展  |


### 处理后的数据集获取链接
[https://www.modelscope.cn/datasets/opencompass/ceval-exam/resolve/master/ceval-exam.zip](https://www.modelscope.cn/datasets/opencompass/ceval-exam/resolve/master/ceval-exam.zip)

## 可用数据集任务

### ceval_gen_0_shot_str
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|ceval_gen_0_shot_str|C-Eval数据集生成式任务|accuracy|0-shot|字符串格式|[ceval_gen_0_shot_str.py](ceval_gen_0_shot_str.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets ceval_gen_0_shot_str
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.ceval.ceval_gen_0_shot_str import ceval_datasets
datasets = [
    *ceval_datasets,
]
```

### ceval_gen_5_shot_str
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|ceval_gen_5_shot_str|C-Eval数据集生成式任务|accuracy|5-shot|字符串格式|[ceval_gen_5_shot_str.py](ceval_gen_5_shot_str.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets ceval_gen_5_shot_str
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.ceval.ceval_gen_5_shot_str import ceval_datasets
datasets = [
    *ceval_datasets,
]
```

### ceval_gen_0_shot_cot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|ceval_gen_0_shot_cot_chat_prompt||C-Eval数据集生成式任务，prompt带逻辑链|accuracy|0-shot|对话格式|[ceval_gen_0_shot_cot_chat_prompt.py](ceval_gen_0_shot_cot_chat_prompt.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets ceval_gen_0_shot_cot_chat_prompt
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.ceval.ceval_gen_0_shot_cot_chat_prompt import ceval_datasets
datasets = [
    *ceval_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见Python源码配置文件的注释

