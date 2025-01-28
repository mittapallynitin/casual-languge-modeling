from typing import Any

from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


def create_model() -> tuple[GPT2LMHeadModel, PreTrainedTokenizerFast]:
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="./tokenizer/custom_tokenizer.json",  # Path to the saved tokenizer
        unk_token="<unk>",                        # Unknown token
        pad_token="<pad>",                        # Padding token
        bos_token="<s>",                          # Beginning-of-sequence token
        eos_token="</s>"                          # End-of-sequence token
    )
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,  # Match tokenizer vocab size
        n_positions=512,                        # Maximum sequence length (same as max_len)
        n_embd=384,                             # Embedding dimension
        n_layer=12,                             # Number of transformer layers
        n_head=12,                              # Number of attention heads
        bos_token_id=tokenizer.bos_token_id,  # Beginning-of-sequence token
        eos_token_id=tokenizer.eos_token_id   # End-of-sequence token
    )
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(config.vocab_size)
    return model, tokenizer

