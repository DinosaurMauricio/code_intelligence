import os
import torch

from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.collate import collate_fn
from dataset.code_datasets import CodeRefactoringDataset
from utils.data import load_data
from utils.parser import CodeParser
from utils.config import set_seed


path = os.path.dirname(os.path.realpath(__file__))
DATASET_FILE_NAME = "dataset.parquet"
BATCH_SIZE = 5
SEED = 42
LR = 0.0001

filepath = os.path.join(path, DATASET_FILE_NAME)

data = load_data(filepath)


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


import torch.nn as nn


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        print(f"Dummy encoder initialized with hidden size {hidden_size}")

        # Add a single dummy parameter so .parameters() works
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        # Return zeros, correct shape
        last_hidden_state = torch.zeros((batch_size, seq_len, self.hidden_size))

        return type("Output", (), {"last_hidden_state": last_hidden_state})

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


tokenizer = DummyTokenizer()
model = DummyEncoder()

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


partial_collate = partial(collate_fn, pad_id=tokenizer.pad_token_id)

train_dataloader = DataLoader(
    data_dict["train"],
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=partial_collate,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

set_seed(SEED)
print(f"Seed: {SEED}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler(device=device)  # type: ignore


for _ in range(BATCH_SIZE):
    for inputs in tqdm(train_dataloader):
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            outputs = model(**inputs)
            loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
