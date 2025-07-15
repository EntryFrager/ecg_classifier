import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from ecg_classifier.module import train, test
from ecg_classifier.utils import setup_device, SeedEverything, setup_logger


@hydra.main(
    config_path="configs/", config_name="config.yaml", version_base=hydra.__version__
)
def main(cfg: DictConfig):
    setup_logger()
    SeedEverything()
    device = setup_device()

    ecg_dataset = instantiate(cfg.dataset)
    train_dataset, val_dataset, test_dataset = ecg_dataset.get_dataset()
    pos_weight = ecg_dataset.get_pos_weight().to(device)
    ecg_dataset.close_dataset()

    batch_size = cfg.model.train.batch_size
    n_epoch = cfg.model.train.n_epoch

    criterion = instantiate(cfg.criterion, pos_weight=pos_weight)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

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
        device=device,
    )
    test_loss = test(net, test_loader, criterion, threshold_preds, device)


if __name__ == "__main__":
    main()
