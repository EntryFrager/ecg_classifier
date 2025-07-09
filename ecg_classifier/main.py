import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from ecg_classifier import ECGDataset, train, test, device, get_stat


@hydra.main(config_path="configs/", config_name="config.yaml", version_base=hydra.__version__)
def main(cfg: DictConfig):
    path = cfg.dataset.path
    sampling_rate = cfg.dataset.sampling_rate
    target_labels = cfg.dataset.target_labels
    use_metadata = cfg.dataset.use_metadata
    use_pqrst = cfg.dataset.use_pqrst

    ecg_dataset = ECGDataset(path, 
                            target_labels, 
                            sampling_rate, 
                            use_pqrst=use_pqrst, 
                            use_metadata=use_metadata)

    ptbxl = ecg_dataset.ptbxl_dataset
    get_stat(ptbxl, target_labels)

    train_dataset, val_dataset, test_dataset = ecg_dataset.get_dataset()
    pos_weight = ecg_dataset.get_pos_weight().to(device)
    ecg_dataset.close_dataset()

    batch_size  = cfg.train.batch_size
    n_epoch     = cfg.train.n_epoch
    num_classes = cfg.dataset.num_classes

    patience   = cfg.early_stopping.patience
    threshold_preds = [0.5] * num_classes

    criterion = instantiate(cfg.criterion, pos_weight=pos_weight)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size)

    net = instantiate(cfg.model).to(device)
    optimizer = instantiate(cfg.optimizer, params=net.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    net, threshold_preds, train_history, val_history = train(net, train_loader, val_loader, 
                                                             n_epoch, optimizer, criterion, 
                                                             scheduler, threshold_preds, patience)
    test_loss = test(net, test_loader, criterion, threshold_preds)


if __name__ == "__main__":
    main()