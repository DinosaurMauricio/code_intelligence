import os
import torch

from functools import partial
from tqdm import tqdm
from omegaconf import OmegaConf

from utils.collate import collate_fn
from utils.data import load_data, DatasetBuilder, DataLoaderBuilder
from utils.parser import CodeParser
from utils.config import set_seed

# from dummy_classes import DummyEncoder, DummyTokenizer

PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
config = OmegaConf.load(PROJECT_PATH + "/config.yaml")

DATA_FILE_PATH = os.path.join(PROJECT_PATH, config.files.data)
data = load_data(DATA_FILE_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if config.train.seed:
    set_seed(config.train.seed)
    print(f"Seed: {config.train.seed}")

tokenizer = # DummyTokenizer()
model = # DummyEncoder()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)
scaler = torch.amp.GradScaler(device=device)  # type: ignore

datasets = DatasetBuilder(tokenizer, CodeParser()).build(data)

for key, dataset in datasets.items():
    print(f"[{key}] split: {len(dataset)} samples")

partial_collate_fn = partial(collate_fn, pad_id=tokenizer.pad_token_id)
dataloaders = DataLoaderBuilder(partial_collate_fn, config).build(datasets)

# for _ in range(BATCH_SIZE):
#    for inputs in tqdm(train_dataloader):
#        optimizer.zero_grad()
#
#        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):

            
#            outputs = model(**inputs)
#            loss = outputs.loss
#            scaler.scale(loss).backward()
#            scaler.step(optimizer)
#            scaler.update()
#
