import os

import requests
import scipy.io
import torch
from torch.utils.data import Dataset


class FACESDataset(Dataset):
    """Implements the FACES dataset in pytorch."""

    def __init__(self, root, transform=None):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._source_url = (
            "https://www.cs.toronto.edu/~jmartens/newfaces_rot_single.mat"
        )
        mat_file = os.path.basename(self._source_url)
        self.data_dir = os.path.join(root, "FACES")

        local_mat_path = os.path.join(self.data_dir, mat_file)
        if not os.path.exists(local_mat_path):
            print("Downloading dataset...")
            os.makedirs(self.data_dir)
            r = requests.get(self._source_url)
            with open(local_mat_path, "wb") as f:
                f.write(r.content)
            print("Download finished.")

        print("Loading dataset...")
        data = scipy.io.loadmat(local_mat_path)
        self.data = data["newfaces_single"].T.reshape([-1, 25, 25])

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx, :]
        if self.transform:
            sample = self.transform(sample)

        return sample


class CURVESDataset(Dataset):
    """Implements the CURVES dataset in pytorch."""

    def __init__(self, root, train=True, transform=None):
        """
        Args:
            root (string): Directory with all the images.
            train (bool): Whether to return training or test dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._source_url = "https://www.cs.toronto.edu/~jmartens/digs3pts_1.mat"
        mat_file = os.path.basename(self._source_url)
        self.data_dir = os.path.join(root, "CURVES")
        self.train = train

        local_mat_path = os.path.join(self.data_dir, mat_file)
        if not os.path.exists(local_mat_path):
            print("Downloading CURVES dataset...")
            os.makedirs(self.data_dir)
            r = requests.get(self._source_url)
            with open(local_mat_path, "wb") as f:
                f.write(r.content)
            print("Download finished.")

        print("Loading dataset...")
        data = scipy.io.loadmat(local_mat_path)

        if train:
            self.data = data["bdata"]
            self.labels = data["tdata"]
        else:
            self.data = data["bdatatest"]
            self.labels = data["tdatatest"]

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_sample = self.data[idx, :]
        target_sample = self.labels[idx, :]
        if self.transform:
            input_sample = self.transform(input_sample)

        return input_sample, target_sample
