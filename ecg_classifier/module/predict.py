import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from typing import Tuple, List, Any

from ecg_classifier.module.callbacks import EarlyStopping
from ecg_classifier.module.metrics import (
    find_best_threshold,
    get_metrics,
    get_metrics_mtl,
    find_best_threshold_mtl,
)
from ecg_classifier.utils import log_output


def train(
    net: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_epoch: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: Any,
    early_stopping: EarlyStopping,
    alpha: float = 0.5,
    device: str = "cpu",
    use_metadata: bool = False,
) -> Tuple[nn.Module, np.ndarray, List[float], List[float]]:
    loss_train_history = []
    loss_val_history = []

    threshold_preds = []

    writer = SummaryWriter(log_dir="logs")

    for epoch in range(n_epoch):
        log_output("Epoch {}/{}:".format(epoch + 1, n_epoch))

        for param_group in optimizer.param_groups:
            log_output(f"Current learning rate: {param_group['lr']}")

        train_loss = val_loss = 0.0
        val_labels, val_prob = [], []

        net.train()

        for batch_idx, train_batch in enumerate(train_loader):
            samples_ecg, labels = train_batch["ecg_signals"].to(device), train_batch[
                "labels"
            ].to(device)

            optimizer.zero_grad()

            if use_metadata:
                samples_meta = train_batch["metadata"].to(device)
                preds = net(samples_ecg, samples_meta)
            else:
                preds = net(samples_ecg)

            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        loss_train_history.append(train_loss)

        net.eval()

        with torch.no_grad():
            for val_batch in val_loader:
                samples_ecg, labels = val_batch["ecg_signals"].to(device), val_batch[
                    "labels"
                ].to(device)

                if use_metadata:
                    samples_meta = val_batch["metadata"].to(device)
                    preds = net(samples_ecg, samples_meta)
                else:
                    preds = net(samples_ecg)

                val_loss += criterion(preds, labels).item()

                preds = torch.sigmoid(preds)

                val_prob.append(preds.cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_loss /= len(val_loader)
        loss_val_history.append(val_loss)

        val_labels = np.concatenate(val_labels)
        val_prob = np.concatenate(val_prob)

        scheduler.step(val_loss)

        threshold_preds = find_best_threshold(val_labels, val_prob, alpha)

        log_output("\nValidation metrics:")
        get_metrics(val_labels, val_prob, threshold_preds)

        log_output(f"\ntrain Loss: {train_loss:.4f}" f"\nval Loss: {val_loss:.4f}")

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch + 1)

        if early_stopping(val_loss, net, threshold_preds):
            break

    writer.close()

    early_stopping.save_best_model()

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
    device: str = "cpu",
    use_metadata: bool = False,
) -> float:
    net.eval()

    test_loss = 0.0
    test_labels, test_prob = [], []

    with torch.no_grad():
        for batch_idx, test_batch in enumerate(test_loader):
            samples_ecg, labels = test_batch["ecg_signals"].to(device), test_batch[
                "labels"
            ].to(device)

            if use_metadata:
                samples_meta = test_batch["metadata"].to(device)
                preds = net(samples_ecg, samples_meta)
            else:
                preds = net(samples_ecg)

            test_loss += criterion(preds, labels).item()

            preds = torch.sigmoid(preds)

            test_prob.append(preds.cpu().numpy())
            test_labels.append(labels.cpu().numpy())

    test_loss /= len(test_loader)

    log_output("\nTest metrics:")
    get_metrics(np.concatenate(test_labels), np.concatenate(test_prob), threshold_preds)

    log_output(f"\nTest Loss: {test_loss:.4f}")

    return test_loss


def trainMTL(
    net: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_epoch: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: Any,
    early_stopping: EarlyStopping,
    alpha: float = 0.5,
    device: str = "cpu",
) -> Tuple[nn.Module, np.ndarray, List[float], List[float]]:
    loss_train_history = []
    loss_val_history = []

    threshold_ecg = []
    threshold_meta = 0.5

    writer = SummaryWriter(log_dir="logs")

    for epoch in range(n_epoch):
        log_output("Epoch {}/{}:".format(epoch + 1, n_epoch))

        for param_group in optimizer.param_groups:
            log_output(f"Current learning rate: {param_group['lr']}")

        train_loss = val_loss = 0.0
        val_labels_ecg, val_prob_ecg = [], []
        val_labels_meta, val_prob_meta = [], []

        net.train()

        for _, train_batch in enumerate(train_loader):
            samples_ecg, labels_ecg, labels_meta = (
                train_batch["ecg_signals"].to(device),
                train_batch["labels"].to(device),
                train_batch["metadata"].to(device),
            )
            labels = torch.cat([labels_ecg, labels_meta], dim=1)

            optimizer.zero_grad()
            preds = net(samples_ecg)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        loss_train_history.append(train_loss)

        net.eval()

        with torch.no_grad():
            for val_batch in val_loader:
                samples_ecg, labels_ecg, labels_meta = (
                    val_batch["ecg_signals"].to(device),
                    val_batch["labels"].to(device),
                    val_batch["metadata"].to(device),
                )
                labels = torch.cat([labels_ecg, labels_meta], dim=1)

                preds = net(samples_ecg)
                val_loss += criterion(preds, labels).item()
                preds = torch.sigmoid(preds)

                val_prob_ecg.append(preds.cpu().numpy()[:, 0].reshape(-1, 1))
                val_labels_ecg.append(labels_ecg.cpu().numpy())
                val_prob_meta.append(preds.cpu().numpy()[:, 1].reshape(-1, 1))
                val_labels_meta.append(labels_meta.cpu().numpy())

        val_loss /= len(val_loader)

        loss_val_history.append(val_loss)

        val_labels_ecg = np.concatenate(val_labels_ecg)
        val_prob_ecg = np.concatenate(val_prob_ecg)
        val_labels_meta = np.concatenate(val_labels_meta)
        val_prob_meta = np.concatenate(val_prob_meta)

        scheduler.step(val_loss)

        threshold_ecg = find_best_threshold(val_labels_ecg, val_prob_ecg, alpha)
        threshold_meta = find_best_threshold_mtl(val_labels_meta, val_prob_meta)

        log_output("\nValidation metrics:")
        get_metrics_mtl(
            val_labels_ecg,
            val_prob_ecg,
            val_labels_meta,
            val_prob_meta,
            threshold_ecg,
            threshold_meta,
        )

        log_output(f"\ntrain Loss: {train_loss:.4f} \nval Loss: {val_loss:.4f}")

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch + 1)

        if early_stopping(val_loss, net, threshold_ecg, threshold_meta):
            break

    writer.close()

    early_stopping.save_best_model()

    return (
        early_stopping.best_model,
        early_stopping.best_threshold_ecg,
        early_stopping.best_threshold_meta,
        loss_train_history,
        loss_val_history,
    )


def testMTL(
    net: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    threshold_ecg: np.ndarray,
    threshold_meta: np.ndarray,
    device: str = "cpu",
) -> float:
    net.eval()

    test_loss = 0.0
    test_labels_ecg, test_prob_ecg = [], []
    test_labels_meta, test_prob_meta = [], []

    with torch.no_grad():
        for _, test_batch in enumerate(test_loader):
            samples_ecg, labels_ecg, labels_meta = (
                test_batch["ecg_signals"].to(device),
                test_batch["labels"].to(device),
                test_batch["metadata"].to(device),
            )
            labels = torch.cat([labels_ecg, labels_meta], dim=1)

            preds = net(samples_ecg)
            test_loss += criterion(preds, labels).item()
            preds = torch.sigmoid(preds)

            test_prob_ecg.append(preds.cpu().numpy()[:, 0].reshape(-1, 1))
            test_labels_ecg.append(labels_ecg.cpu().numpy())
            test_prob_meta.append(preds.cpu().numpy()[:, 1].reshape(-1, 1))
            test_labels_meta.append(labels_meta.cpu().numpy())

    test_loss /= len(test_loader)

    test_labels_ecg = np.concatenate(test_labels_ecg)
    test_prob_ecg = np.concatenate(test_prob_ecg)
    test_labels_meta = np.concatenate(test_labels_meta)
    test_prob_meta = np.concatenate(test_prob_meta)

    log_output("\nTest metrics:")
    get_metrics_mtl(
        test_labels_ecg,
        test_prob_ecg,
        test_labels_meta,
        test_prob_meta,
        threshold_ecg,
        threshold_meta,
    )

    log_output(f"\ntest Loss: {test_loss:.4f}")

    return test_loss
