import torch


def collate_fn(batch, tokenizer):
    labels = [sample["label"] for sample in batch]
    masked_inputs = torch.tensor([sample["masked_input"] for sample in batch])

    labels_tokenized = tokenizer(
        labels,
        return_tensors="pt",
        # TODO: check how truncating affects as some codes are to big, they have +1000 words
        # maybe I can first truncate and then mask. Or I can just filter the dataset
        max_length=256,
        truncation=True,
        padding="max_length",
    )

    labels_tokenized["input_ids"][masked_inputs != tokenizer.mask_token_id] = -100

    inputs = {
        "input_ids": masked_inputs,
        "attention_mask": labels_tokenized["attention_mask"],
    }

    return inputs, labels_tokenized["input_ids"]
