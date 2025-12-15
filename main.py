import os
from functools import partial
from torchtext.data import get_tokenizer  # dummy tokenizer
from torch.utils.data import DataLoader

from dataset.code_datasets import CodeRefactoringDataset
from utils.data import load_data
from utils.parser import CodeParser


path = os.path.dirname(os.path.realpath(__file__))
DATASET_FILE_NAME = "dataset.parquet"

filepath = os.path.join(path, DATASET_FILE_NAME)

data = load_data(filepath)


# can be bytes(string, encoding) or b'string'
code_bytes = bytes("""hello world""", "utf-8")

import torch


# dummy tokenizer, must use code-bert tokenizer
class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2

    def __call__(self, text, return_tensors="pt", padding=True, truncation=True):
        # Fake tokenization: one token per word
        token_ids = (
            [self.cls_token_id]
            + [i + 3 for i, _ in enumerate(text.split())]
            + [self.sep_token_id]
        )
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, **kwargs):
        return "[Dummy decoded text]"


tokenizer = DummyTokenizer()

dataset_partial = partial(
    CodeRefactoringDataset,
    tokenizer=tokenizer,
    parser=CodeParser(),
)

splits = ["train", "valid", "test"]
data_dict = {}
for split in splits:
    data_split = data[data.split_name == split]
    data_dict[split] = dataset_partial(data=data_split)


from utils.collate import collate_fn

partial_collate = partial(collate_fn, pad_id=tokenizer.pad_token_id)

train_dataloader = DataLoader(
    data_dict["train"],
    shuffle=True,
    batch_size=2,
    collate_fn=partial_collate,
)


it = iter(train_dataloader)
next(it)


## <mask> bert Mask
## 50264
