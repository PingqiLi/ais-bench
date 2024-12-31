# 大模型精度评测组件evaluate使用说明

## 工具介绍

### 工具概述
AISBench工具配套大模型精度评测组件，用于全流程大模型评测以及生成式大模型结果多维分析。

### 工具依赖

- OS：Linux
- Python版本：Python 3.7及以上版本

## 安装和卸载

### 安装evaluate

1. 从[evaluate发行版](https://gitee.com/aisbench/evaluate/releases/)获取whl包，`ais_bench_evaluate-<version>-py3-none-any.whl`, 通过`pip install`命令安装：
    ```bash
    pip install ais_bench_evaluate-<version>-py3-none-any.whl --force-reinstall
    ```
2. 执行`pip show ais_bench_evaluate`确认安装完成，输出结果如下：
   ```bash
   Name: ais-bench-evaluate
   Version:
   Summary: ais_bench evaluate
   Home-page:
   Author:
   Author-email:
   License:
   Location: /xxxx/site-packages
   Requires:
   Required-by:
   ```
### 卸载evaluate
卸载evaluate工具可以使用如下命令：
```bash
pip uninstall ais_bench_evaluate
```

## 工厂类使用方法
### 导入依赖包
```python
from ais_bench.evaluate.interface import dataset_factory, measurement_factory
```

### API使用介绍
#### 评测数据集类工厂

 - 接口原型

   ```python
   dataset_instance = dataset_factory.get(dataset_name, dataset_path, shot)
   ```

 - 参数说明

   | 参数名       | 入参类型 | 说明                                                         | 是否必选 |
   | ------------ | -------- | ------------------------------------------------------------ | -------- |
   | dataset_name | str      | 数据集名称，支持的数据集详见“附录 > **支持的数据集**”。      | 是       |
   | dataset_path | str      | 数据集路径，不设置时会自动下载到evaluate安装路径下的dataset目录下。 | 否       |
   | shot         | int      | 构造输入中的prompt数量，0 <= shot <= 5。                     | 否       |

#### 评测指标类工厂

 - 接口原型

   ```python
   measurement_instance = measurement_factory.get(measurement_name, **kwargs)
   ```
 - 参数说明

   | 参数名           | 入参类型 | 说明                                                    | 是否必选 |
   | ---------------- | -------- | ------------------------------------------------------- | -------- |
   | measurement_name | str      | 评测指标名称，支持的评测指标详见“附录 > **评测指标**”。 | 是       |
   | kwargs           | -        | 额外参数，详见“附录 > **评测指标**”。                   | 否       |

## 全流程评测使用方法（Evaluator类）
### 导入依赖包
```python
from ais_bench.evaluate.interface import Evaluator
```
### API使用介绍
#### Evaluator类
- 类原型

  ```python
  class Evaluator(generate_func, dataset_instance, measurement_instance, rank=0):
  ```
- 初始化参数

  | 参数名               | 入参类型        | 说明                                                         | 是否必选 |
  | -------------------- | --------------- | ------------------------------------------------------------ | -------- |
  | generate_func        | function object | 封装了大模型推理能力的函数对象，输入自然语言，输出自然语言。 | 是       |
  | dataset_instance     | BaseDataset     | 评测数据集类实例。                                           | 是       |
  | measurement_instance | BaseMeasurement | 评测指标类实例。                                             | 是       |
  | rank                 | int             | 多进程推理情况下进程的标号，只有在0号进程中对数据进行评测。  | 否       |

#### evaluate（评测接口）
- 功能说明

  进行评测，打屏评测结果，输出评测结果。

- 函数原型

  ```python
  evaluate()
  ```
- 返回值

  float, dict


## 生成式大模型结果多维分析使用方法（Filter类）
### 导入依赖包
```python
from ais_bench.evaluate.interface import Filter
```

### API使用方法
#### Filter类
类原型

```python
class Filter():
```

#### add_measurement（添加评测指标接口）
- 功能说明

  添加需要用于评测的指标，支持的指标及额外参数详见“附录 > **评测指标**”。

- 函数原型

  ```python
  add_measurement(measurement: str, **kwargs)
  ```
- 返回值

  None

#### do_measuring（执行评测接口）
- 功能说明

  用所有添加的评测的指标对传入的result_dict进行评测。

- 函数原型

  ```python
  do_measuring(result_dict: dict):
  ```
- result_dict格式

  | key名         | value类型   | 说明               | 是否必选 |
  | ------------- | ----------- | ------------------ | -------- |
  | target_output | `List[str]` | 标杆模型输出列表。 | 是       |
  | model_output  | `List[str]` | 实际模型输出列表。 | 是       |
  | indexs        | `List[int]` | 索引。             | 是       |

  三个List中的元素需要一一对应，List长度需要一致。



- 返回值

  None

#### do_filtering（执行筛选接口）
- 功能说明

  对评测完的结果按照thresholds进行筛选，并且在out_dir下生成 `${pid}_${6位随机字符}_filtering_result.csv` 文件。可选参数thresholds为字典，key为评测指标名称，value为threshold。未配置或错误配置时使用默认threshold，详见附录 > **评测指标**"。

- 函数原型

  ```python
  do_filtering(out_dir: str, thresholds=None: dict):
  ```
- 返回值

  pandas.dataframe


## 附录
### 支持数据集

|评测数据集名称|说明|
|----|----|
|ceval|一个用于基础模型评估的多层次多学科的中文评估套件。|
|mmlu|大规模多任务语言理解数据集。|
|gsm8k|小学数学题数据集。|


### 评测指标

|评测指标名|说明|结果范围|额外参数|默认threshold(仅用于筛选功能)|
|----|----|----|----|----|
|accuracy|正确率|0，1|无|>=1|
|edit-distance|编辑距离|[0, +∞)|无|<=5|
|bleu|机器翻译质量评估指标|[0, 1]|ngram：int，可取值：1，2，3，4，默认为1。|>=0.4|
|rouge|自动文本摘要评估指标|[0, 1]|rouge_type：str，可取值：rouge-1, rouge-2, rouge-l，默认为rouge-1。|>=0.4|
|distinct|实际模型输出的多样性得分|[0, 1]|ngram：int，可取值：1，2，3，4，默认为2。|>=0.6|
|abnormal-string-rate|实际模型输出的异常字符率|[0, 1]|无|<=0.3|
|relative-distinct|相对多样性得分|[0, +∞)|ngram：int，可取值：1，2，3，4，默认为2。|>=0.8|
|relative-abnormal-string-rate|相对异常字符率|[0, +∞)|无|<=1.2|

###  参考样例

- [全流程精度评测](sample/sample_evaluator.py)
- [生成式大模型结果多维分析](sample/sample_filter.py)
