# VideoBench
## 数据集简介
VideoBench是一个视频相关大模型的评估基准，AISBench支持VideoBench作为文本视频多模态理解任务的测评，文本为关于视频内容的选择题。


## 数据集获取链接
[https://huggingface.co/datasets/maoxx241/videobench_subset](https://huggingface.co/datasets/maoxx241/videobench_subset)

## 数据集内容格式(处理后)
### 文件结构
```
videobench
├── answer
│   └── ANSWER.json
├── ActivityNet_QA_new.json
├── Driving-decision-making_QA_new.json
├── Driving-exam_QA_new.json
├── MOT_QA_new.json
├── MSRVTT_QA_new.json
├── MSVD_QA_new.json
├── NBA_QA_new.json
├── SQA3D_QA_new.json
├── TGIF_QA_new.json
└── Ucfcrime_QA_new.json
```
### 数据集内容样例格式

|vid_path|video_id|question|choices|
| ---- | ---- | ----- | ----- |
|"/data/VedioBench/video/NBA/10.mp4"|10|"What type of game is this?"|{"A": "NBA regular season", "B": "NBA Finals", "C": "NBA All-Star game", "D": "NBA playoffs"}|


## 可用数据集任务

### videobench_gen
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|videobench_gen|VideoBench数据集生成式任务|accuracy|0-shot|列表格式（包含文本和视频两种数据）|[videobench_gen.py](videobench_gen.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets videobench_gen
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.videobench.videobench_gen import videobench_datasets
datasets = [
    *videobench_datasets,
]
```
**注:** 该数据集任务下，会直接将视频路径传入服务化，需确保服务化支持该格式输入并且有权限访问该路径视频。

### videobench_gen_base64
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|videobench_gen_base64|VideoBench数据集生成式任务|accuracy|0-shot|列表格式（包含文本和视频两种数据）|[videobench_gen_base64.py](videobench_gen_base64.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets videobench_gen_base64
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.videobench.videobench_gen_base64 import videobench_datasets
datasets = [
    *videobench_datasets,
]
```

**注:** 该数据集任务下，会先将视频进行抽帧再转化为base64格式传入服务化，需确保服务化支持该输入格式数据。其中num_frames表示视频抽帧数，默认为5。
