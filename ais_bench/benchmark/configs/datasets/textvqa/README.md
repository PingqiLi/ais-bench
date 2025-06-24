# TextVQA
## 数据集简介
TextVQA为图片文本多模态理解数据集，文本为每张图片相关的问题，数据集中的图片来自OpenImages。


## 数据集获取链接
[https://huggingface.co/datasets/maoxx241/textvqa_subset](https://huggingface.co/datasets/maoxx241/textvqa_subset)

## 数据集内容格式(处理后)
### 文件结构
```
textvqa
├── train_images
│   ├── 0004c9478eeda995.jpg
│   └── 00054dab88635bdb.jpg
├── textvqa_val.jsonl
└── textvqa_val_annotations.json
```
### 数据集内容样例格式

|image|question|question_id|answer|
| ---- | ---- | ----- | ----- |
|"data/textvqa/train_images/003a8ae2ef43b901.jpg"|"what is the brand of this camera?"|34602|"dakota"|


## 可用数据集任务

### textvqa_gen
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|textvqa_gen|TextVQA数据集生成式任务|VQA|0-shot|列表格式（包含文本和图片两种数据）|[textvqa_gen.py](textvqa_gen.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets textvqa_gen
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.textvqa.textvqa_gen import textvqa_datasets
datasets = [
    *textvqa_datasets,
]
```

**注:** 该数据集任务下，会直接将图片路径传入服务化，需确保服务化支持该格式输入并且有权限访问该路径图片。


### textvqa_gen_base64
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|textvqa_gen_base64|TextVQA数据集生成式任务|VQA|0-shot|列表格式（包含文本和图片两种数据）|[textvqa_gen_base64.py](textvqa_gen_base64.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets textvqa_gen_base64
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.textvqa.textvqa_gen_base64 import textvqa_datasets
datasets = [
    *textvqa_datasets,
]
```

**注:** 该数据集任务下，会将图片数据转化为base64格式再传入服务化，需确保服务化支持该输入格式数据。
