# MIT License
#
# Copyright (c) 2024 Denis Prokopenko

import torch
import h5py
import numpy as np
from torchvision import transforms
from src.transforms import undersampling, ToImage
from glob import glob
import os


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.dir = data_dir
        self.files = sorted(glob(os.path.join(self.dir, "*_*.*")))

        if transform is None:
            self.transform = transforms.Lambda(lambda x: x)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.files)

    def _read_file(self, file, key="data"):
        if file.endswith(".npy"):
            return self._read_numpy(file)
        elif file.endswith(".hdf5"):
            return self._read_hdf5(file, key=key)
        else:
            raise NotImplementedError("Unsupported File Format")

    def _read_numpy(self, file):
        return np.load(file)

    def _read_hdf5(self, file, key="data"):
        with h5py.File(file, "r") as f:
            return f[key][:]

    def __getitem__(self, idx):
        # Read the data as is
        data = self._read_file(self.files[idx])
        return self.transform(data)


class PairedDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        transform=None,
        pattern=None,
        acceleration=8,
        frames=32,
        img_size=96,
        u_coef=None,
        mask_id=None,
        mask_dir=None,
    ):
        self.dir = data_dir
        self.files = sorted(glob(os.path.join(self.dir, "*_*.*")))

        self.frames = frames
        self.img_size = img_size
        self.pattern = pattern
        self.u_coef = u_coef
        self.mask_id = mask_id
        self.acceleration = acceleration
        self.mask_dir = mask_dir

        if transform is None:
            self.transform = transforms.Lambda(lambda x: x)
        else:
            self.transform = transform

    def __getitem__(self, idx):
        data = self._read_file(self.files[idx])

        kspace = self.transform(data)
        target = ToImage()(kspace)

        target_delta = target.abs().max()

        target /= target_delta
        kspace /= target_delta

        mask = undersampling(
            shape=(self.frames, self.img_size, 1),
            acceleration=self.acceleration,
            mask_type=self.pattern,
            idx=self.mask_id,
            mask_dir=self.mask_dir,
        )

        input_data = (kspace, mask)
        target_data = (target,)
        return input_data, target_data
