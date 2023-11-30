from torch.utils.data import DataLoader

from dataset.nyu import NYUDataset
from dataset.kitti import KITTIDataset
from config_settings import config_settings


def build_dataloader(settings: config_settings):
    """
    return train_loader, val_loader, test_loader
    """
    if settings.dataset_choose == "nyu":
        train_dataset = NYUDataset(settings, "train")
        train_loader = DataLoader(
            train_dataset, batch_size=settings.batch_size, num_workers=settings.loader_workers, shuffle=True, pin_memory=True, drop_last=True
        )

        val_dataset = NYUDataset(settings, "val")
        val_loader = DataLoader(val_dataset, batch_size=1, num_workers=settings.loader_workers, shuffle=False, pin_memory=True, drop_last=True)

        return train_loader, val_loader, None

    elif settings.dataset_choose == "kitti":
        train_dataset = KITTIDataset(settings, "train")
        train_loader = DataLoader(
            train_dataset, batch_size=settings.batch_size, num_workers=settings.loader_workers, shuffle=True, pin_memory=True, drop_last=True
        )

        val_dataset = KITTIDataset(settings, "val")
        val_loader = DataLoader(val_dataset, batch_size=1, num_workers=settings.loader_workers, shuffle=False, pin_memory=True, drop_last=True)

        test_dataset = KITTIDataset(settings, "test")
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=settings.loader_workers, shuffle=False, pin_memory=True, drop_last=True)

        return train_loader, val_loader, test_loader
    else:
        raise NotImplementedError
