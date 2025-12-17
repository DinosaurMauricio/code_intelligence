import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch, pad_id):
    masked_inputs = []
    masked_inputs_attention = []
    labels = []

    for sample in batch:
        masked_inputs.append(torch.tensor(sample["masked_input"]["input_ids"]))
        masked_inputs_attention.append(
            torch.tensor(sample["masked_input"]["attention_mask"])
        )
        labels.append(torch.tensor(sample["label"]["input_ids"]))

    padded_input = pad_sequence(masked_inputs, padding_value=pad_id, batch_first=True)
    padded_input_attention = pad_sequence(
        masked_inputs_attention, padding_value=0, batch_first=True
    )
    padded_labels = pad_sequence(labels, padding_value=pad_id, batch_first=True)

    inputs = {
        "input_ids": padded_input,
        "attention_mask": padded_input_attention,
    }

    return inputs, padded_labels
