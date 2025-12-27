import torch


def collate_fn(batch, tokenizer):
    labels = torch.tensor([sample["label"] for sample in batch])
    masked_inputs = torch.tensor([sample["input"]["token_ids"] for sample in batch])
    attention = torch.tensor([sample["input"]["attention"] for sample in batch])

    inputs = {"input_ids": masked_inputs, "attention_mask": attention}

    return inputs, labels
