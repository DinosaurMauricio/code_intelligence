import torch

from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class MLMMetrics:
    train_losses = []
    train_accuracies = []
    train_topk_accuracies = []
    train_mrr_scores = []
    val_losses = []
    val_accuracies = []
    val_topk_accuracies = []
    val_mrr_scores = []


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
            print(f"Epoch: {epoch}")

            train_loss, train_acc, train_topk, train_mrr = self._train_epoch(
                dataloaders["train"]
            )
            val_loss, val_acc, val_topk, val_mrr = self._validate_epoch(
                dataloaders["val"]
            )

            self._record_metrics(
                train_loss,
                train_acc,
                train_topk,
                train_mrr,
                val_loss,
                val_acc,
                val_topk,
                val_mrr,
            )
            print(
                f"Train Avg. Loss: {train_loss} Train Avg. Acc.: {train_acc} Train Avg. Top-k Acc.: {train_topk} Train Avg. MRR: {train_mrr}"
            )
            print(
                f"Val. Avg. Loss: {val_loss} Val. Avg. Acc.: {val_acc} Val. Avg. Top-k Acc.: {val_topk} Val. Avg. MRR: {val_mrr}"
            )

            torch.save(self.model.state_dict(), f"model_{epoch}.pth")

        return self.metrics

    def _train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_predictions = 0
        total_topk_correct = 0
        total_topk_predictions = 0
        total_mrr = 0.0
        total_mrr_len = 0

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
            topk_correct, topk_preds = self._topk_accuracy(
                outputs.logits,
                inputs["input_ids"],
                labels,
            )
            mrr, mrr_len = self._mrr(
                outputs.logits,
                inputs["input_ids"],
                labels,
            )

            total_loss += loss.item()
            total_correct += num_correct
            total_predictions += num_preds
            total_topk_correct += topk_correct
            total_topk_predictions += topk_preds
            total_mrr += mrr
            total_mrr_len += mrr_len

        avg_loss = total_loss / len(dataloader)
        avg_accuracy = (
            total_correct / total_predictions if total_predictions > 0 else 0.0
        )
        avg_topk_accuracy = total_topk_correct / total_topk_predictions
        avg_mrr = total_mrr / total_mrr_len if total_mrr_len > 0 else 0.0

        return avg_loss, avg_accuracy, avg_topk_accuracy, avg_mrr

    def _validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_predictions = 0
        total_topk_correct = 0
        total_topk_predictions = 0
        total_mrr = 0.0
        total_mrr_len = 0

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
                topk_correct, topk_preds = self._topk_accuracy(
                    outputs.logits,
                    inputs["input_ids"],
                    labels,
                )
                mrr, mrr_len = self._mrr(
                    outputs.logits,
                    inputs["input_ids"],
                    labels,
                )

                total_loss += loss.item()
                total_correct += num_correct
                total_predictions += num_preds
                total_topk_correct += topk_correct
                total_topk_predictions += topk_preds
                total_mrr += mrr
                total_mrr_len += mrr_len

            avg_loss = total_loss / len(dataloader)
            avg_accuracy = (
                total_correct / total_predictions if total_predictions > 0 else 0.0
            )
            avg_topk_accuracy = total_topk_correct / total_topk_predictions
            avg_mrr = total_mrr / total_mrr_len if total_mrr_len > 0 else 0.0

        return avg_loss, avg_accuracy, avg_topk_accuracy, avg_mrr

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

    def _topk_accuracy(self, logits, input_ids, labels, k=5):
        mask_positions = torch.nonzero(input_ids == self.tokenizer.mask_token_id)

        if len(mask_positions) == 0:
            return 0, 0

        batch_indices = mask_positions[:, 0]
        seq_indices = mask_positions[:, 1]

        _, top_k_preds_indices = torch.topk(logits[batch_indices, seq_indices, :], k)

        targets = labels[batch_indices, seq_indices]

        matches = targets.unsqueeze(dim=1) == top_k_preds_indices
        num_correct = torch.any(matches, dim=1).float().sum().item()

        return num_correct, len(top_k_preds_indices)

    def _mrr(self, logits, input_ids, labels, k=5):
        mask_positions = torch.nonzero(input_ids == self.tokenizer.mask_token_id)

        if len(mask_positions) == 0:
            return 0, 0

        batch_indices = mask_positions[:, 0]
        seq_indices = mask_positions[:, 1]

        _, top_k_preds_indices = torch.topk(logits[batch_indices, seq_indices, :], k)

        targets = labels[batch_indices, seq_indices]

        matches = targets.unsqueeze(dim=1) == top_k_preds_indices

        positions = torch.arange(1, k + 1, device=logits.device)
        rank_tensor = torch.where(matches, positions, k + 1)
        min_result = torch.min(rank_tensor, dim=-1).values
        rr = torch.where(min_result <= k, 1.0 / min_result.float(), 0.0)

        return rr.sum().item(), rr.numel()

    def _record_metrics(
        self,
        train_loss,
        train_acc,
        train_topk,
        train_mrr,
        val_loss,
        val_acc,
        val_topk,
        val_mrr,
    ):
        self.metrics.train_losses.append(train_loss)
        self.metrics.train_accuracies.append(train_acc)
        self.metrics.val_losses.append(val_loss)
        self.metrics.val_accuracies.append(val_acc)
        self.metrics.train_topk_accuracies.append(train_topk)
        self.metrics.val_topk_accuracies.append(val_topk)
        self.metrics.train_mrr_scores.append(train_mrr)
        self.metrics.val_mrr_scores.append(val_mrr)
