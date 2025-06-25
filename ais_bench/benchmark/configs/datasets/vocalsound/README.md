# VocalSound
## 数据集简介
VocalSound是一个用于人类时声音识别的数据集，说话者包含了不同年龄、性别和国家，共有超过21000条wav格式的语音文件，覆盖了laughter（笑声）、sigh（叹息）、cough（咳嗽）、throat clearing（清嗓子）、sneeze（打喷嚏）、sniff（抽鼻子）等六种不同类型的声音。模型需要判断不同的语音文件属于哪一类的声音。


## 数据集获取链接
[https://huggingface.co/datasets/maoxx241/audio_vocalsound_16k_subset](https://huggingface.co/datasets/maoxx241/audio_vocalsound_16k_subset)

## 数据集内容格式(处理后)
### 文件结构
如下所示，每个音频文件名中都包含对应的类别信息。
```
vocalsound
├── f0003_0_cough.wav
├── f0004_0_laughter.wav
└── f0007_0_sneeze.wav
```


## 可用数据集任务

### vocalsound_gen
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|vocalsound_gen|VocalSound数据集生成式任务|accuracy|0-shot|列表格式（包含文本和音频两种数据）|[vocalsound_gen.py](vocalsound_gen.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets vocalsound_gen
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.vocalsound.vocalsound_gen import vocalsound_datasets
datasets = [
    *vocalsound_datasets,
]
```

**注:** 该数据集任务下会直接将音频路径传入服务化，需确保服务化支持该格式输入并且有权限访问该路径音频。


### vocalsound_gen_base64
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|vocalsound_gen_base64|VocalSound数据集生成式任务|accuracy|0-shot|列表格式（包含文本和音频两种数据）|[vocalsound_gen_base64.py](vocalsound_gen_base64.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets vocalsound_gen_base64
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.vocalsound.vocalsound_gen_base64 import vocalsound_datasets
datasets = [
    *vocalsound_datasets,
]
```

**注:** 该数据集任务下，会将音频数据转化为base64格式再传入服务化，需确保服务化支持该输入格式数据。

