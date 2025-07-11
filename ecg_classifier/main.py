import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from ecg_classifier.module import train, test
from ecg_classifier.utils import device


@hydra.main(
    config_path="configs/", config_name="config.yaml", version_base=hydra.__version__
)
def main(cfg: DictConfig):
    ecg_dataset = instantiate(cfg.dataset)

    train_dataset, val_dataset, test_dataset = ecg_dataset.get_dataset()
    pos_weight = ecg_dataset.get_pos_weight().to(device)
    ecg_dataset.close_dataset()

    batch_size = cfg.model.train.batch_size
    n_epoch = cfg.model.train.n_epoch

    criterion = instantiate(cfg.criterion, pos_weight=pos_weight)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    net = instantiate(cfg.model.resnet).to(device)
    optimizer = instantiate(cfg.optimizer, params=net.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    early_stopping = instantiate(cfg.early_stopping)

    net, threshold_preds, train_history, val_history = train(
        net,
        train_loader,
        val_loader,
        n_epoch,
        optimizer,
        criterion,
        scheduler,
        early_stopping,
    )
    test_loss = test(net, test_loader, criterion, threshold_preds)


if __name__ == "__main__":
    main()
