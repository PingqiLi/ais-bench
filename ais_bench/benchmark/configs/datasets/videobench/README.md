# VideoBench
## 使用说明
1 不同的视频输入格式满足不同服务化输入数据格式要求，video_url表示传入视频路径，video_url_base64表示将视频转化为base64再传入，默认为video_url。
2 传入video_url_base64时，必须传入num_frames参数，表示对视频的抽帧数量，默认值为5。
3 测试过程中选择EVAL_QA_5000数据集，[数据链接](https://huggingface.co/datasets/maoxx241/videobench_subset)
4 数据集中传入的视频路径需保证服务端可以访问。