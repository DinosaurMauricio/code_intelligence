import os
import torch

from functools import partial
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model

from utils.collate import collate_fn
from utils.data import load_data, DatasetBuilder, DataLoaderBuilder
from utils.parser import CodeParser
from utils.config import set_seed
from utils.processor import CodeMaskingProcessor
from utils.trainer import MaskedLanguageModelTrainer
from utils.constants import MASK_PLACEHOLDER

from dummy_classes import DummyEncoder, DummyTokenizer

PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
config = OmegaConf.load(PROJECT_PATH + "/config.yaml")
DATA_FILE_PATH = os.path.join(PROJECT_PATH, config.files.data)

data = load_data(DATA_FILE_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if config.train.seed:
    set_seed(config.train.seed)
    print(f"Seed: {config.train.seed}")

tokenizer = DummyTokenizer()
# add to tokenizer vocabulary <mask_holder>
tokenizer.add_tokens([MASK_PLACEHOLDER])
model = DummyEncoder()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)
scaler = torch.amp.GradScaler(device=device)  # type: ignore

datasets = DatasetBuilder(CodeParser(), CodeMaskingProcessor(tokenizer)).build(data)

for key, dataset in datasets.items():
    print(f"[{key}] split: {len(dataset)} samples")


partial_collate_fn = partial(collate_fn, tokenizer=tokenizer)
dataloaders = DataLoaderBuilder(partial_collate_fn, config).build(datasets)

lora_config = LoraConfig(
    r=16,  # rank
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable params: {params}")


trainer = MaskedLanguageModelTrainer(model, tokenizer, optimizer, scaler, device)

trainer.train(dataloaders, config.train.epochs)
