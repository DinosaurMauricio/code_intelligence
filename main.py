import os
import torch

from functools import partial
from tqdm import tqdm
from omegaconf import OmegaConf

from utils.collate import collate_fn
from utils.data import load_data, DatasetBuilder, DataLoaderBuilder
from utils.parser import CodeParser
from utils.config import set_seed

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
model = DummyEncoder()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)
scaler = torch.amp.GradScaler(device=device)  # type: ignore

datasets = DatasetBuilder(tokenizer, CodeParser()).build(data)

for key, dataset in datasets.items():
    print(f"[{key}] split: {len(dataset)} samples")

partial_collate_fn = partial(collate_fn, pad_id=tokenizer.pad_token_id)
dataloaders = DataLoaderBuilder(partial_collate_fn, config).build(datasets)

# for param in model.roberta.parameters():
#    param.requires_grad = False
#
# for param in model.lm_head.parameters():
#    param.requires_grad = True

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(params)


losses = []
accuracies = []
accuracies_eval = []
losses_eval = []
EPOCHS = 10
for epoch in range(EPOCHS, EPOCHS + 5):
    total_loss = 0.0
    total_acc = 0.0
    total_preds = 0
    model.train()
    for inputs, labels in tqdm(dataloaders["train"]):
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16):

            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
        total_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mask_indices = torch.nonzero(inputs["input_ids"] == tokenizer.mask_token_id)
        batch_indices = mask_indices[:, 0]
        seq_indices = mask_indices[:, 1]
        result = outputs.logits[batch_indices, seq_indices, :].argmax(1)
        total_preds += len(result)

        total_acc += (result == labels[batch_indices, seq_indices]).sum().item()

    avg_loss = total_loss / len(dataloaders["train"])
    avg_acc = total_acc / total_preds
    losses.append(avg_loss)
    accuracies.append(avg_acc)

    print(f"Average loss train: {avg_loss}")
    print(f"Average accuracy train: {avg_acc}")

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_preds = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders["val"]):
            with torch.autocast(device_type=device, dtype=torch.float16):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss

            # losses_eval.append(loss.item())
            total_loss += loss.item()

            mask_indices = torch.nonzero(inputs["input_ids"] == tokenizer.mask_token_id)

            batch_indices = mask_indices[:, 0]
            seq_indices = mask_indices[:, 1]
            result = outputs.logits[batch_indices, seq_indices, :].argmax(1)
            total_preds += len(result)

            total_acc += (result == labels[batch_indices, seq_indices]).sum().item()

    avg_loss = total_loss / len(dataloaders["val"])
    avg_acc = total_acc / total_preds
    accuracies_eval.append(avg_acc)
    losses_eval.append(avg_loss)

    print(f"Average loss eval: {avg_loss}")
    print(f"Average accuracy eval : {avg_acc}")

    torch.save(model.state_dict(), f"model_{epoch}.pth")
