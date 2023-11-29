import torch
from torchvision import transforms
from torch.utils.data import Dataset

import os
import pandas as pd
import h5py as h5

from PIL import Image
import matplotlib.pyplot as plt


class VoidDataset(Dataset):
    def __init__(self, nyu_dataset_root_path, which, split):
        """
        Args:
            nyu_dataset_root_path (string): Path to the dataset folder.
            which (string): 'void_150', 'void_500' or 'void_1500'.
            split (string): 'train' or 'test' split.
        """
        self.nyu_dataset_root_path = nyu_dataset_root_path
        self.split = split
        self.absolute_pose_txt = pd.read_csv(os.path.join(nyu_dataset_root_path, which, split + "_absolute_pose.txt"))
        self.ground_truth_txt = pd.read_csv(os.path.join(nyu_dataset_root_path, which, split + "_ground_truth.txt"))
        self.image_txt = pd.read_csv(os.path.join(nyu_dataset_root_path, which, split + "_image.txt"))
        self.intrinsics_txt = pd.read_csv(os.path.join(nyu_dataset_root_path, which, split + "_intrinsics.txt"))
        self.sparse_depth_txt = pd.read_csv(os.path.join(nyu_dataset_root_path, which, split + "_sparse_depth.txt"))
        self.validity_map_txt = pd.read_csv(os.path.join(nyu_dataset_root_path, which, split + "_validity_map.txt"))

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

        a_ground_truth = Image.open(os.path.join(self.nyu_dataset_root_path, a_ground_truth_name))
        a_image = Image.open(os.path.join(self.nyu_dataset_root_path, a_image_name))
        a_sparse_depth = Image.open(os.path.join(self.nyu_dataset_root_path, a_sparse_depth_name))
        a_validity_map = Image.open(os.path.join(self.nyu_dataset_root_path, a_validity_map_name))

        # transform
        a_ground_truth = self.transform(a_ground_truth)
        a_image = self.transform(a_image)
        a_sparse_depth = self.transform(a_sparse_depth)
        # a_validity_map = self.transform(a_validity_map)

        a_rgbd = torch.cat((a_image, a_sparse_depth), 0)

        # return {'gt': a_ground_truth, 'img': a_image, 'sd': a_sparse_depth, 'vm': a_validity_map}
        return {"gt": a_ground_truth, "rgbd": a_rgbd}


# main
if __name__ == "__main__":
    void_dataset = VoidDataset(nyu_dataset_root_path="data/void", which="void_1500", split="train")
    for i in range(3):
        sample = void_dataset[i]
        print(i, sample["gt"].size(), sample["rgbd"].size())
