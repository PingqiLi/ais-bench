# LongBench
## 数据集简介
LongBench是个用于双语、多任务、全面评估大型语言模型长上下文理解能力的基准测试。LongBench涵盖了不同语言（中文和英文），以便更全面地评估大模型在长上下文中的多语言能力。此外，LongBench包含六大类别和二十一项不同的任务，涵盖单文档问答、多文档问答、摘要、小样本学习、合成任务和代码补全等关键的长文本应用场景。
LongBench包含14个英文任务、5个中文任务和2个代码任务，大部人任务的平均长度在5k到15k之间，共包含4750条测试数据。
> 🔗 数据集主页链接[https://huggingface.co/datasets/zai-org/LongBench](https://huggingface.co/datasets/zai-org/LongBench)
## 数据集部署
建议从HuggingFace下载数据集：[https://huggingface.co/datasets/zai-org/LongBench](https://huggingface.co/datasets/zai-org/LongBench)
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径）
- 部署完成后，在`{工具根路径}/ais_bench/datasets`目录下执行`tree LongBench/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    LongBench/
    ├── data
    │   ├── 2wikimqa_e.jsonl
    │   ├── 2wikimqa.jsonl
    │   ├── dureader.jsonl
    │   ├── gov_report_e.jsonl
    │   ├── gov_report.jsonl
    │   ├── hotpotqa_e.jsonl
    │   ├── hotpotqa.jsonl
    │   ├── lcc_e.jsonl
    │   ├── lcc.jsonl
    │   ├── lsht.jsonl
    │   ├── multifieldqa_en_e.jsonl
    │   ├── multifieldqa_en.jsonl
    │   ├── multifieldqa_zh.jsonl
    │   ├── multi_news_e.jsonl
    │   ├── multi_news.jsonl
    │   ├── musique.jsonl
    │   ├── narrativeqa.jsonl
    │   ├── passage_count_e.jsonl
    │   ├── passage_count.jsonl
    │   ├── passage_retrieval_en_e.jsonl
    │   ├── passage_retrieval_en.jsonl
    │   ├── passage_retrieval_zh.jsonl
    │   ├── qasper_e.jsonl
    │   ├── qasper.jsonl
    │   ├── qmsum.jsonl
    │   ├── repobench-p_e.jsonl
    │   ├── repobench-p.jsonl
    │   ├── samsum_e.jsonl
    │   ├── samsum.jsonl
    │   ├── trec_e.jsonl
    │   ├── trec.jsonl
    │   ├── triviaqa_e.jsonl
    │   ├── triviaqa.jsonl
    │   └── vcsum.jsonl
    └── LongBench.py
    ```
## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|longbench|longbench|准确率(accuracy)|0-shot|对话格式|[longbench.py](longbench.py)|
|longbench_2wikimqa_gen|longbench_2wikimqa_gen|准确率(accuracy)|0-shot|对话格式|[longbench_2wikimqa_gen.py](longbench2wikimqa/longbench_2wikimqa_gen.py)|
|longbench_dureader_gen|longbench_dureader_gen|准确率(accuracy)|0-shot|对话格式|[longbench_dureader_gen.py](longbenchdureader/longbench_dureader_gen.py)|
|longbench_gov_report_gen|longbench_gov_report_gen|准确率(accuracy)|0-shot|对话格式|[longbench_gov_report_gen.py](longbenchgov_report/longbench_gov_report_gen.py)|
|longbench_hotpotqa_gen|longbench_hotpotqa_gen|准确率(accuracy)|0-shot|对话格式|[longbench_hotpotqa_gen.py](longbenchhotpotqa/longbench_hotpotqa_gen.py)|
|longbench_lcc_gen|longbench_lcc_gen|准确率(accuracy)|0-shot|对话格式|[longbench_lcc_gen.py](longbenchlcc/longbench_lcc_gen.py)|
|longbench_lsht_gen|longbench_lsht_gen|准确率(accuracy)|0-shot|对话格式|[longbench_lsht_gen.py](longbenchlsht/longbench_lsht_gen.py)|
|longbench_multi_news_gen|longbench_multi_news_gen|准确率(accuracy)|0-shot|对话格式|[longbench_multi_news_gen.py](longbenchmulti_news/longbench_multi_news_gen.py)|
|longbench_multifieldqa_en_gen|longbench_multifieldqa_en_gen|准确率(accuracy)|0-shot|对话格式|[longbench_multifieldqa_en_gen.py](longbenchmultifieldqa_en/longbench_multifieldqa_en_gen.py)|
|longbench_multifieldqa_zh_gen|longbench_multifieldqa_zh_gen|准确率(accuracy)|0-shot|对话格式|[longbench_multifieldqa_zh_gen.py](longbenchmultifieldqa_zh/longbench_multifieldqa_zh_gen.py)|
|longbench_musique_gen|longbench_musique_gen|准确率(accuracy)|0-shot|对话格式|[longbench_musique_gen.py](longbenchmusique/longbench_musique_gen.py)|
|longbench_narrativeqa_gen|longbench_narrativeqa_gen|准确率(accuracy)|0-shot|对话格式|[longbench_narrativeqa_gen.py](longbenchnarrativeqa/longbench_narrativeqa_gen.py)|
|longbench_passage_count_gen|longbench_passage_count_gen|准确率(accuracy)|0-shot|对话格式|[longbench_passage_count_gen.py](longbenchpassage_count/longbench_passage_count_gen.py)|
|longbench_passage_retrieval_en_gen|longbench_passage_retrieval_en_gen|准确率(accuracy)|0-shot|对话格式|[longbench_passage_retrieval_en_gen.py](longbenchpassage_retrieval_en/longbench_passage_retrieval_en_gen.py)|
|longbench_passage_retrieval_zh_gen|longbench_passage_retrieval_zh_gen|准确率(accuracy)|0-shot|对话格式|[longbench_passage_retrieval_zh_gen.py](longbenchpassage_retrieval_zh/longbench_passage_retrieval_zh_gen.py)|
|longbench_qasper_gen|longbench_qasper_gen|准确率(accuracy)|0-shot|对话格式|[longbench_qasper_gen.py](longbenchqasper/longbench_qasper_gen.py)|
|longbench_qmsum_gen|longbench_qmsum_gen|准确率(accuracy)|0-shot|对话格式|[longbench_qmsum_gen.py](longbenchqmsum/longbenchqmsum_gen.py)|
|longbench_repobench_gen|longbench_repobench_gen|准确率(accuracy)|0-shot|对话格式|[longbench_repobench_gen.py](longbenchrepobench/longbench_repobench_gen.py)|
|longbench_samsum_gen|longbench_samsum_gen|准确率(accuracy)|0-shot|对话格式|[longbench_samsum_gen.py](longbenchsamsum/longbench_samsum_gen.py)|
|longbench_trec_gen|longbench_trec_gen|准确率(accuracy)|0-shot|对话格式|[longbench_trec_gen.py](longbenchtrec/longbench_trec_gen.py)|
|longbench_triviaqa_gen|longbench_triviaqa_gen|准确率(accuracy)|0-shot|对话格式|[longbench_triviaqa_gen.py](longbenchtriviaqa/longbench_triviaqa_gen.py)|
|longbench_vcsum_gen|longbench_vcsum_gen|准确率(accuracy)|0-shot|对话格式|[longbench_vcsum_gen.py](longbenchvcsum/longbench_vcsum_gen.py)|