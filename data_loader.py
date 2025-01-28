import datasets as hf_datasets
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast as ptf

import preprocessor as pp


class CodeDataset(Dataset):
    def __init__(self, data: hf_datasets.Dataset):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return pp.preprocess_record(self.data[index])

class CodeInputCollator:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        batch = [example for example in batch if example is not None]
        inputs = [example["description"] for example in batch]
        outputs = [example["code"] for example in batch]

        tokens = self.tokenizer(
            inputs, outputs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_len
        )

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

def get_data_loaders(tokenizer: ptf, max_len: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = hf_datasets.load_dataset("code_search_net", "python", trust_remote_code=True)
    train_dataset = CodeDataset(dataset["train"])
    val_dataset = CodeDataset(dataset["validation"])
    test_dataset = CodeDataset(dataset["test"])

    codeInputCollator = CodeInputCollator(tokenizer, max_len=max_len)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=codeInputCollator
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=codeInputCollator
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=codeInputCollator
    )

    return train_dataloader, val_dataloader, test_dataloader