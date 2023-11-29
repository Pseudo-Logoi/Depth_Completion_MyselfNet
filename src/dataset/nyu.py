import torch
from torchvision import transforms
from torch.utils.data import Dataset

import os
import pandas as pd
import h5py as h5

from PIL import Image
import matplotlib.pyplot as plt

from config_settings import config_settings


class NYUDataset(Dataset):
    def __init__(self, settings: config_settings, split="train"):
        assert split in ["train", "test"], "split should be train or test"

        """
        sparse_density: void_1500: 0.005, void_500: 0.0016, void_150: 0.0005
        """
        super().__init__()

        if split == "train":
            self.csv_file = pd.read_csv(os.path.join(settings.nyu_dataset_root_path, settings.nyu_train_csv))
        elif split == "test":
            self.csv_file = pd.read_csv(os.path.join(settings.nyu_dataset_root_path, settings.nyu_test_csv))

        self.nyu_dataset_root_path = settings.nyu_dataset_root_path
        self.sparse_density = settings.sparse_density
        self.length = len(self.csv_file)

        self.common_transform = (
            transforms.Compose(
                [
                    # transforms.ToTensor(),
                    # transforms.Resize((480, 640), antialias=True),  # 缩放到 480*640
                    transforms.RandomRotation(10),  # 随机旋转 +-10 degree
                    transforms.CenterCrop((384, 512)),  # 中心裁剪
                    transforms.RandomHorizontalFlip(),  # 随机水平翻转
                    transforms.RandomVerticalFlip(),  # 随机垂直翻转
                ]
            )
            if split == "train"
            else transforms.Compose(
                [
                    # transforms.Resize((480, 640), antialias=True),  # 缩放到 480*640
                    transforms.CenterCrop((384, 512)),  # 中心裁剪
                ]
            )
        )

        self.rgb_transform = transforms.Compose(
            [
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # 随机改变亮度、对比度和饱和度
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.depth_transform = transforms.Compose([])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        hd5_file_name = os.path.join(self.nyu_dataset_root_path, self.csv_file["Name"][idx])
        hd5_file = h5.File(hd5_file_name, "r")
        rgb_h5 = hd5_file["rgb"][:].transpose(1, 2, 0)
        depth_h5 = hd5_file["depth"][:]

        rgb_image = Image.fromarray(rgb_h5, mode="RGB")
        depth_image = Image.fromarray(depth_h5.astype("float32"), mode="F")

        # transform
        rgb_tensor = transforms.ToTensor()(rgb_image)
        depth_tensor = transforms.ToTensor()(depth_image)

        common_tensor = torch.cat((rgb_tensor, depth_tensor), 0)
        common_tensor = self.common_transform(common_tensor)

        rgb_tensor_raw = common_tensor[:3, :, :]
        rgb_tensor = self.rgb_transform(common_tensor[:3, :, :])
        depth_tensor = self.depth_transform(common_tensor[3:, :, :])

        return {"gt": depth_tensor, "rgb": rgb_tensor, "sqarse_dep": self.create_sparse_depth(depth_tensor), "raw_rgb": rgb_tensor_raw}

    def create_sparse_depth(self, depth_image):
        random_mask = torch.bernoulli(torch.ones_like(depth_image, dtype=torch.float32) * self.sparse_density)
        return depth_image * random_mask


# main
if __name__ == "__main__":
    nyu_dataset = NYUDataset(nyu_dataset_root_path="data/nyudepth_hdf5", csv_file_name="nyudepth_hdf5_train.csv", n_samples=100)
    for i in range(1):
        sample = nyu_dataset[i]
        # print(i, sample["gt"].size(), sample["rgbd"].size())
