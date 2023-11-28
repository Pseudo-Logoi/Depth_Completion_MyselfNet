from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import h5py as h5

__all__ = ["VoidDataset", "KittiDataset", "NYUDataset"]


class VoidDataset(Dataset):
    def __init__(self, dataset_root_path, which, split):
        """
        Args:
            dataset_root_path (string): Path to the dataset folder.
            which (string): 'void_150', 'void_500' or 'void_1500'.
            split (string): 'train' or 'test' split.
        """
        self.dataset_root_path = dataset_root_path
        self.split = split
        self.absolute_pose_txt = pd.read_csv(os.path.join(dataset_root_path, which, split + "_absolute_pose.txt"))
        self.ground_truth_txt = pd.read_csv(os.path.join(dataset_root_path, which, split + "_ground_truth.txt"))
        self.image_txt = pd.read_csv(os.path.join(dataset_root_path, which, split + "_image.txt"))
        self.intrinsics_txt = pd.read_csv(os.path.join(dataset_root_path, which, split + "_intrinsics.txt"))
        self.sparse_depth_txt = pd.read_csv(os.path.join(dataset_root_path, which, split + "_sparse_depth.txt"))
        self.validity_map_txt = pd.read_csv(os.path.join(dataset_root_path, which, split + "_validity_map.txt"))

        self.length = len(self.absolute_pose_txt)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.ToPILImage(),
            ]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample.
        """
        a_absolute_pose_name = self.absolute_pose_txt.iloc[idx, 0]
        a_ground_truth_name = self.ground_truth_txt.iloc[idx, 0]
        a_image_name = self.image_txt.iloc[idx, 0]
        a_intrinsics_name = self.intrinsics_txt.iloc[idx, 0]
        a_sparse_depth_name = self.sparse_depth_txt.iloc[idx, 0]
        a_validity_map_name = self.validity_map_txt.iloc[idx, 0]

        a_ground_truth = Image.open(os.path.join(self.dataset_root_path, a_ground_truth_name))
        a_image = Image.open(os.path.join(self.dataset_root_path, a_image_name))
        a_sparse_depth = Image.open(os.path.join(self.dataset_root_path, a_sparse_depth_name))
        a_validity_map = Image.open(os.path.join(self.dataset_root_path, a_validity_map_name))

        # transform
        a_ground_truth = self.transform(a_ground_truth)
        a_image = self.transform(a_image)
        a_sparse_depth = self.transform(a_sparse_depth)
        # a_validity_map = self.transform(a_validity_map)

        a_rgbd = torch.cat((a_image, a_sparse_depth), 0)

        # return {'gt': a_ground_truth, 'img': a_image, 'sd': a_sparse_depth, 'vm': a_validity_map}
        return {"gt": a_ground_truth, "rgbd": a_rgbd}


class KittiDataset:
    def __init__(self, dataset_root_path, which, split):
        yield

    def __len__(self):
        yield

    def __getitem__(self, idx):
        yield


class NYUDataset:
    def __init__(self, dataset_root_path, csv_file_name, sparse_density, split="train"):
        """
        sparse_density: void_1500: 0.005, void_500: 0.0016, void_150: 0.0005
        """
        self.csv_file = pd.read_csv(os.path.join(dataset_root_path, csv_file_name))
        self.dataset_root_path = dataset_root_path
        self.length = len(self.csv_file)
        self.sparse_density = sparse_density

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
        hd5_file_name = os.path.join(self.dataset_root_path, self.csv_file["Name"][idx])
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
    void_dataset = VoidDataset(dataset_root_path="data/void", which="void_1500", split="train")
    for i in range(3):
        sample = void_dataset[i]
        print(i, sample["gt"].size(), sample["rgbd"].size())

    nyu_dataset = NYUDataset(dataset_root_path="data/nyudepth_hdf5", csv_file_name="nyudepth_hdf5_train.csv", n_samples=100)
    for i in range(1):
        sample = nyu_dataset[i]
        # print(i, sample["gt"].size(), sample["rgbd"].size())
