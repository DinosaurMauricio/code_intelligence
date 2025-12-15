import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch, pad_id):

    masked_inputs = []
    masked_inputs_attention = []
    labels = []
    labels_attention = []

    for sample in batch:
        masked_inputs.append(sample["masked_input"]["input_ids"].squeeze(0))
        masked_inputs_attention.append(
            sample["masked_input"]["attention_mask"].squeeze(0)
        )
        labels.append(sample["label"]["input_ids"].squeeze(0))
        labels_attention.append(sample["label"]["input_ids"].squeeze(0))

    padded_input = pad_sequence(masked_inputs, padding_value=pad_id, batch_first=True)
    padded_input_attention = pad_sequence(
        masked_inputs_attention, padding_value=0, batch_first=True
    )
    padded_label = pad_sequence(labels, padding_value=pad_id, batch_first=True)
    padded_label_attention = pad_sequence(
        labels_attention, padding_value=0, batch_first=True
    )

    samples = {
        "input_ids": padded_input,
        "input_attentions": padded_input_attention,
        "labels": padded_label,
        "labels_attentions": padded_label_attention,
    }

    return samples
