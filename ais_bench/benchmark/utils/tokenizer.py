import os
import glob
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, ques_string: str, add_special_tokens=True):
        """Encode the input text to tokens."""
        pass

    @abstractmethod
    def decode(self, token_ids: list, skip_special_tokens=False):
        """Decode the tokens back to text."""
        pass

    @abstractmethod
    def batch_encode_plus(self, messages, add_special_tokens=True) -> list:
        pass



class HuggingfaceTokenizer(Tokenizer):
    def __init__(self, model_name_or_path: str, trust_remote_code: bool = False):
        import transformers
        from transformers import AutoTokenizer
        try:
            self.tokenizer_model = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=trust_remote_code, use_fast=True, local_files_only=True
            )
        except RecursionError as e:
            raise RecursionError(
                f"Failed to load the Tokenizer using AutoTokenizer. The current transformers "
                f"version is {transformers.__version__}. Please visit the HuggingFace official "
                "website to update to the latest weights and Tokenizer."
            ) from e

    def encode(self, ques_string: str, add_special_tokens=True) -> list:
        return self.tokenizer_model.encode(ques_string, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list, skip_special_tokens=False) -> str:
        return self.tokenizer_model.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_encode_plus(self, messages, add_special_tokens=True) -> list:
        return self.tokenizer_model.batch_encode_plus(messages, add_special_tokens=add_special_tokens)


class MindformersTokenizer(Tokenizer):
    def __init__(self, model_name_or_path: str, trust_remote_code: bool = False):
        import mindformers
        from mindformers import get_model
        try:
            self.tokenizer_model, _ = get_model(
                model_name_or_path, trust_remote_code=trust_remote_code, use_fast=True, local_files_only=True)
        except RecursionError as e:
            raise RecursionError(
                f"Failed to load the Tokenizer using AutoTokenizer. The current mindformers "
                f"version is {mindformers.__version__}. Please visit the Mindformers official "
                "website to update to the latest weights and Tokenizer."
            ) from e

    def encode(self, ques_string: str, add_special_tokens=True) -> list:
        return self.tokenizer_model.encode(ques_string, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list, skip_special_tokens=False) -> str:
        return self.tokenizer_model.decode(token_ids, skip_special_tokens=skip_special_tokens)


class BenchmarkTokenizer:
    def __init__(self, model_path: str, tokenizer_type: str = None, trust_remote_code: bool = False, **kwargs):
        if tokenizer_type is None:
            yaml_files = glob.glob(os.path.join(model_path, "*.yaml"))
            if len(yaml_files) > 0:
                tokenizer_type = 'mindformers'
            else:
                tokenizer_type = 'transformers'

        if tokenizer_type == 'transformers':
            self.tokenizer = HuggingfaceTokenizer(model_path, trust_remote_code=trust_remote_code, **kwargs)
        elif tokenizer_type == 'mindformers':
            self.tokenizer = MindformersTokenizer(model_path, trust_remote_code=trust_remote_code, **kwargs)
        else:
            raise ValueError("Tokenizer Type Not Supported")

    def encode(self, text: str, add_special_tokens=True, **kwargs):
        return self.tokenizer.encode(text, add_special_tokens, **kwargs)

    def decode(self, tokens, skip_special_tokens=False, **kwargs):
        return self.tokenizer.decode(tokens, skip_special_tokens, **kwargs)

    def batch_encode_plus(self, messages, add_special_tokens=True) -> list:
        return self.tokenizer.batch_encode_plus(messages, add_special_tokens=add_special_tokens)
