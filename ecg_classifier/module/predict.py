import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from typing import Tuple, List, Any, Callable, Optional

from ecg_classifier.module.callbacks import EarlyStopping
from ecg_classifier.module.metrics import find_best_threshold, get_metrics
from ecg_classifier.utils import device


def train(
    net: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_epoch: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: Any,
    early_stopping: EarlyStopping,
    compute_metric_best_thr: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Tuple[nn.Module, np.ndarray, List[float], List[float]]:
    loss_train_history = []
    loss_val_history = []

    threshold_preds = []

    writer = SummaryWriter(log_dir="logs")

    for epoch in range(n_epoch):
        print("Epoch {}/{}:".format(epoch + 1, n_epoch), flush=True)

        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']}")

        train_loss = val_loss = 0.0
        val_labels, val_prob = [], []

        net.train()

        for _, train_batch in enumerate(train_loader):
            samples, labels = train_batch["ecg_signals"].to(device), train_batch[
                "labels"
            ].to(device)
            optimizer.zero_grad()

            preds = net(samples)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        loss_train_history.append(train_loss)

        net.eval()

        with torch.no_grad():
            for val_batch in val_loader:
                samples, labels = val_batch["ecg_signals"].to(device), val_batch[
                    "labels"
                ].to(device)
                preds = net(samples)
                val_loss += criterion(preds, labels).item()

                preds = torch.sigmoid(preds)

                val_prob.append(preds.cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_loss /= len(val_loader)
        loss_val_history.append(val_loss)

        val_labels = np.concatenate(val_labels)
        val_prob = np.concatenate(val_prob)

        scheduler.step(val_loss)

        threshold_preds = find_best_threshold(
            val_labels, val_prob, compute_metric_best_thr
        )

        print("\nValidation metrics:")
        val_sens, val_spec = get_metrics(val_labels, val_prob, threshold_preds)

        print(f"\ntrain Loss: {train_loss:.4f}" f"\nval Loss: {val_loss:.4f}")

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch + 1)

        if early_stopping(val_loss, val_sens, val_spec, net, threshold_preds):
            break

    writer.close()

    torch.save(early_stopping.best_model.state_dict(), "save_best_models/best_model.pt")
    torch.save(early_stopping.best_threshold, "save_best_models/best_threshold.pt")

    return (
        early_stopping.best_model,
        early_stopping.best_threshold,
        loss_train_history,
        loss_val_history,
    )


def test(
    net: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    threshold_preds: np.ndarray,
) -> float:
    net.eval()

    test_loss = 0.0
    test_labels, test_prob = [], []

    with torch.no_grad():
        for batch_idx, test_batch in enumerate(test_loader):
            samples, labels = test_batch["ecg_signals"].to(device), test_batch[
                "labels"
            ].to(device)
            preds = net(samples)

            test_loss += criterion(preds, labels).item()

            preds = torch.sigmoid(preds)

            test_prob.append(preds.cpu().numpy())
            test_labels.append(labels.cpu().numpy())

    test_loss /= len(test_loader)

    print("\nTest metrics:")
    get_metrics(np.concatenate(test_labels), np.concatenate(test_prob), threshold_preds)

    print(f"\ntest Loss: {test_loss:.4f}")

    return test_loss
