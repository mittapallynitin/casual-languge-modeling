from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.normalizers import NFKC, Sequence, Strip, StripAccents
from tokenizers.processors import TemplateProcessing

import preprocessor as pp

required_field = ["func_documentation_string", "func_code_string"]

def train_tokenizer(data, vocab_size=16384, save_path="./tokenizer"):
    """
    Train a BPE tokenizer with normalization, pre-tokenization, and post-processing.
    """
    # Initialize a tokenizer with a BPE model
    tokenizer = Tokenizer(models.BPE())

    # Step 1: Add a normalizer
    tokenizer.normalizer = Sequence([
        StripAccents(),  # Unicode normalization
        Strip(),  # Remove leading/trailing spaces
    ])

    # Step 2: Add a pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),  # Split by whitespace
        pre_tokenizers.Punctuation(),  # Split punctuation
    ])

    # Step 3: Define a trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    # Train the tokenizer
    tokenizer.train_from_iterator(data, trainer)

    # Step 4: Add a post-processor
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",  # For single sequences
        pair="<s> $A </s> $B:1 </s>:1",  # For paired sequences
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )

    # Save the tokenizer
    tokenizer.save(f"{save_path}/custom_tokenizer.json")
    print(f"Tokenizer saved at {save_path}/custom_tokenizer.json")

    return tokenizer

if __name__ == "__main__":
    # Prepare the data for training the tokenizer
    data = load_dataset("code_search_net", "python", trust_remote_code=True)
    data = [f"{item['description']} {item['code']}" for item in pp.preprocess_batch(data["train"])]

    # Train and save the tokenizer
    custom_tokenizer = train_tokenizer(data)

    # Test the tokenizer with a sample input
    sample_input = "Create a function to add two numbers"
    encoded = custom_tokenizer.encode(sample_input)
    print("Encoded Tokens:", encoded.tokens)