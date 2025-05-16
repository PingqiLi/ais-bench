# HumanEvalx
## 数据集简介
HumanEval-X 是由清华大学 KEG 实验室 THUDM 提供的一套多语言代码生成模型的评价标准。它包含 820 个高质量手写样本，覆盖 Python、C++、Java、JavaScript 和 Go 语言。

**注意**：数据集运行前请先安装依赖[extra.txt](../../../../../requirements/extra.txt)
```shell
# 需要处在最外层benchmark文件夹下，运行下列指令：
pip3 install -r requirements/extra.txt
```

## 数据集原始获取链接
[https://huggingface.co/datasets/THUDM/humaneval-x](https://huggingface.co/datasets/THUDM/humaneval-x)

## 数据集内容格式(处理后)
### 文件结构
```
humanevalx
└── humanevalx_cpp.jsonl
└── humanevalx_go.jsonl
└── humanevalx_java.jsonl
└── humanevalx_js.jsonl
└── humanevalx_python.jsonl
```

### 数据集内容样例格式
|task_id|prompt|canonical_solution|test|
| ----- | ---- | ---- | ---- |
|CPP/0|/*\nCheck if in given vector of numbers, are any two numbers closer to each other than\ngiven threshold.\n>>> has_close_elements({1.0, 2.0, 3.0}, 0.5)\nfalse\n>>> has_close_elements({1.0, 2.8, 3.0, 4.0, 5.0, 2.0}, 0.3)\ntrue\n*/\n#include<stdio.h>\n#include<vector>\n#include<math.h>\nusing namespace std;\nbool has_close_elements(vector<float> numbers, float threshold){\n|    int i,j;\n    \n    for (i=0;i<numbers.size();i++)\n    for (j=i+1;j<numbers.size();j++)\n    if (abs(numbers[i]-numbers[j])<threshold)\n    return true;\n\n    return false;\n}\n\n|#undef NDEBUG\n#include<assert.h>\nint main(){\n    vector<float> a={1.0, 2.0, 3.9, 4.0, 5.0, 2.2};\n    assert (has_close_elements(a, 0.3)==true);\n    assert (has_close_elements(a, 0.05) == false);\n\n    assert (has_close_elements({1.0, 2.0, 5.9, 4.0, 5.0}, 0.95) == true);\n    assert (has_close_elements({1.0, 2.0, 5.9, 4.0, 5.0}, 0.8) ==false);\n    assert (has_close_elements({1.0, 2.0, 3.0, 4.0, 5.0}, 2.0) == true);\n    assert (has_close_elements({1.1, 2.2, 3.1, 4.1, 5.1}, 1.0) == true);\n    assert (has_close_elements({1.1, 2.2, 3.1, 4.1, 5.1}, 0.5) == false);\n    \n}\n", "declaration": "#include<stdio.h>\n#include<vector>\n#include<math.h>\nusing namespace std;\n#include<algorithm>\n#include<stdlib.h>\nbool has_close_elements(vector<float> numbers, float threshold){\n", "example_test": "#undef NDEBUG\n#include<assert.h>\nint main(){\n    assert (has_close_elements({1.0, 2.0, 3.0}, 0.5) == false && \"failure 1\");\n    assert (has_close_elements({1.0, 2.8, 3.0, 4.0, 5.0, 2.0}, 0.3) && \"failure 2\") ;\n}\n|


### 处理后的数据集获取链接
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humanevalx.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humanevalx.zip)

## 可用数据集任务
### humanevalx_gen_0_shot
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|humanevalx_gen_0_shot|humanevalx数据集生成式任务|pass@1|0-shot|字符串格式|[humanevalx_gen_0_shot.py](humanevalx_gen_0_shot.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets humanevalx_gen_0_shot
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.humanevalx.humanevalx_gen_0_shot import humanevalx_datasets
datasets = [
    *humanevalx_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见Python源码配置文件的注释