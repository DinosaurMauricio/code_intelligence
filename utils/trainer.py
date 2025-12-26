import torch

from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class MLMMetrics:
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []


class MaskedLanguageModelTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        scaler,
        device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = optimizer
        self.scaler = scaler
        self.metrics = MLMMetrics()

    def train(self, dataloaders, epochs):

        for epoch in range(epochs):

            train_loss, train_acc = self._train_epoch(dataloaders["train"])
            val_loss, val_acc = self._validate_epoch(dataloaders["val"])

            self._record_metrics(train_loss, train_acc, val_loss, val_acc)
            print(f"Train Avg. Loss: {train_loss} Train Avg. Acc.: {train_acc}")
            print(f"Val. Avg. Loss: {val_loss} Val. Avg. Acc.: {val_acc}")

            torch.save(self.model.state_dict(), f"model_{epoch}.pth")

        return self.metrics

    def _train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_predictions = 0

        for inputs, labels in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            num_correct, num_preds = self._calculate_masked_accuracy(
                outputs.logits,
                inputs["input_ids"],
                labels,
            )

            total_loss += loss.item()
            total_correct += num_correct
            total_predictions += num_preds

        avg_loss = total_loss / len(dataloader)
        avg_accuracy = (
            total_correct / total_predictions if total_predictions > 0 else 0.0
        )

        return avg_loss, avg_accuracy

    def _validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_predictions = 0

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validation"):

                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    labels = labels.to(self.device)
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss

                num_correct, num_preds = self._calculate_masked_accuracy(
                    outputs.logits,
                    inputs["input_ids"],
                    labels,
                )

                total_loss += loss.item()
                total_correct += num_correct
                total_predictions += num_preds

            avg_loss = total_loss / len(dataloader)
            avg_accuracy = (
                total_correct / total_predictions if total_predictions > 0 else 0.0
            )

        return avg_loss, avg_accuracy

    def _calculate_masked_accuracy(self, logits, input_ids, labels):
        mask_positions = torch.nonzero(input_ids == self.tokenizer.mask_token_id)

        if len(mask_positions) == 0:
            return 0, 0

        batch_indices = mask_positions[:, 0]
        seq_indices = mask_positions[:, 1]

        predictions = logits[batch_indices, seq_indices, :].argmax(dim=1)
        targets = labels[batch_indices, seq_indices]

        num_correct = (predictions == targets).sum().item()
        total_predictions = len(predictions)

        return num_correct, total_predictions

    def _record_metrics(self, train_loss, train_acc, val_loss, val_acc):
        self.metrics.train_losses.append(train_loss)
        self.metrics.train_accuracies.append(train_acc)
        self.metrics.val_losses.append(val_loss)
        self.metrics.val_accuracies.append(val_acc)
