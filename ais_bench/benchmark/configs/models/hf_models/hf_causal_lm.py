from ais_bench.benchmark.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM, # 通过AutoModelForCausalLM.from_pretrained 加载模型
        abbr='hf-causal-lm',
        path='THUDM/chatglm-6b', # path to model dir, current value is just a example
        tokenizer_path='THUDM/chatglm-6b', # path to tokenizer dir, current value is just a example
        model_kwargs=dict( # 模型参数参考 huggingface.co/docs/transformers/v4.50.0/en/model_doc/auto#transformers.AutoModel.from_pretrained
            device_map='auto',
        ),
        tokenizer_kwargs=dict( # tokenizer参数参考 huggingface.co/docs/transformers/v4.50.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase
            padding_side='left',
        ),
        generation_kwargs = dict( # 后处理参数参考huggingface.co/docs/transformers/main_classes/test_generation
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            do_sample = True,
            seed = None,
            repetition_penalty = 1.03,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_padding=True,
    )
]