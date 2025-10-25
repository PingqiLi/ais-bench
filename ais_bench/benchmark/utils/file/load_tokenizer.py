import os

from transformers import AutoTokenizer

def load_tokenizer(tokenizer_path: str):
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer path {tokenizer_path} does not exist")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer from {tokenizer_path}: {e}")
    return tokenizer
