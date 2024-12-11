from pathlib import Path

from datasets import Dataset, load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Replace, Sequence, StripAccents
from tokenizers.pre_tokenizers import ByteLevel, Split
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


def get_normalizer() -> Sequence:
    normalizer = Sequence([
            NFD(),
            StripAccents(),
            Replace("\t", "    "),  # Replace tabs with spaces
            Replace("#.*", "<COMMENT>"),  # Replace comments
            Replace(r"(['\"])(?:(?=(\\?))\2.)*?\1", "<STRING>")
    ])
    return normalizer

def get_post_processor():
    return TemplateProcessing(
        single="<START> $A <END>",  # Single sequence
        pair="<START> $A <SEP> $B <END>",  # Pair sequence
        special_tokens=[
            ("<START>", 0),
            ("<END>", 1),
            ("<SEP>", 2),
            ("<INDENT>", 3),
            ("<DEDENT>", 4),
            ("_", 5)
        ]
    )

def get_pre_tokenizer():
    return ByteLevel() 

def get_tokenizer(dataset: Dataset, name: str) -> Tokenizer:
  """
  The function trains a ByteLevelBPETokenizer on the dataset. 

  Creates a tokenizer and saves it to disk.

  Args:
      dataset: The dataset to train the tokenizer on.
      name: The name of the tokenizer.

  Returns:
      tokenizer: The trained tokenizer.
  """

  tokenizer_path = Path("./tokenizer/{name}_tokenizer.json".format(name = name))
  print(tokenizer_path)

  if tokenizer_path.exists():
    # If tokenizer exists return the tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
  else:
    # If tokenizer does not exist train the tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = get_normalizer()
    tokenizer.pre_tokenizer = get_pre_tokenizer()
    tokenizer.post_processor = get_post_processor()


    trainer = BpeTrainer(
        vocab_size=16_384,  # Adjust based on your dataset size
        special_tokens=["<START>", "<END>", "<SEP>", "<INDENT>", "<DEDENT>", "<unk>", "<pad>", "<mask>", "_"]
    )
    tokenizer.train_from_iterator(dataset['func_code_string'], trainer)

    # Save files to disk
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(f"./tokenizer/{name}_tokenizer.json".format(name = name))
    return tokenizer

def clean_decoded(text):
   return text.replace("Ġ", " ").replace("_", "").replace("Ċ", "\n")

if __name__ == "__main__":
  name = "python"
  dataset = load_dataset("code_search_net", "python", trust_remote_code=True)["train"]
  tokenizer = get_tokenizer(dataset, name)

  # test tokenizer on example
  text = "def hello_world():\n    print('Hello, World!')"
  encoded = tokenizer.encode(text)
  print(encoded.tokens)
  decoded = tokenizer.decode(encoded.ids)
  print(clean_decoded(decoded))