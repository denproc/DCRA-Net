# MIT License
#
# Copyright (c) 2024 Denis Prokopenko

import torch
from typing import Union
from torchvision import transforms
from mrirecon.functional import k2x, x2k, t2f, f2t
from scipy.io import loadmat
import os
from glob import glob
import random


def tensor2complex(x: torch.Tensor):
    assert x.dim() == 5, "Expected (N, C, T, H, W) tensor"
    if x.size(1) == 1:
        return x
    else:
        return x[:, 0:1] + 1j * x[:, 1:2]


class ToTensor(transforms.Lambda):
    def __init__(self):
        super().__init__(lambda x: torch.tensor(x))


class Undersample(transforms.Lambda):
    def __init__(
        self, acceleration: int, mask_type: str = None, offset: Union[int, float] = None
    ):
        super().__init__(
            lambda x: x
            * undersampling(
                x.shape, acceleration=acceleration, mask_type=mask_type, offset=offset
            )
        )


class ToReal(transforms.Lambda):
    def __init__(self, batched=False):
        super().__init__(
            lambda x: (
                torch.view_as_real(x).permute(0, -1, *range(1, x.ndim))
                if batched
                else torch.view_as_real(x).permute(-1, *range(0, x.ndim))
            )
        )


class ToComplex(transforms.Lambda):
    def __init__(self, batched=False):
        super().__init__(
            lambda x: torch.view_as_complex(
                x.permute(0, *range(2, x.ndim), 1).contiguous()
                if batched
                else x.permute(*range(1, x.ndim), 0).contiguous()
            )
        )


class AddChannel(transforms.Lambda):
    def __init__(self, dim=0):
        super().__init__(lambda x: x.unsqueeze(dim))


class ToImage(transforms.Lambda):
    def __init__(self, norm: str = None):
        super().__init__(lambda x: k2x(x, norm=norm))


class ToKSpace(transforms.Lambda):
    def __init__(self, norm: str = None):
        super().__init__(lambda x: x2k(x, norm=norm))


class CutFrames(transforms.Lambda):
    def __init__(self, frames: int = None, offset: int = 0):
        super().__init__(lambda x: x[offset : offset + frames])


class ToCuda(transforms.Lambda):
    def __init__(self, device="cuda:0"):
        super().__init__(lambda x: x.to(device))


class ToCPU(transforms.Lambda):
    def __init__(self):
        super().__init__(lambda x: x.cpu())


class ToTime(transforms.Lambda):
    def __init__(self, norm: str = None):
        super().__init__(lambda x: f2t(x, norm=norm))


class ToFrequency(transforms.Lambda):
    def __init__(self, norm: str = None):
        super().__init__(lambda x: t2f(x, norm=norm))


class ComplexNormalize(transforms.Lambda):
    def __init__(self):
        super().__init__(lambda x: x / x.abs().max())


class Identity(transforms.Lambda):
    def __init__(self):
        super().__init__(lambda x: x)


def undersampling(
    shape: tuple,
    acceleration: int,
    mask_type: str = "uniform",
    offset: Union[int, float] = None,
    idx: int = None,
    mask_dir: str = None,
):
    """
    Create undersampling mask

    Args:
        shape: shape of the mask
        acceleration: acceleration factor
        mask_type: type of mask to create. Choose from `uniform`|`lattice`|`random`]
        offset: offset. Integer for uniform mask, float for random mask

    Returns:
        torch.Tensor: undersampling mask
    """
    if mask_type == "uniform":
        return uniform_mask(shape=shape, acceleration=acceleration, offset=offset)
    if mask_type == "random":
        return random_mask(shape=shape, acceleration=acceleration, offset=offset)
    if mask_type == "lattice":
        return lattice_mask(shape=shape, acceleration=acceleration, offset=offset)
    if mask_type == "vista":
        return vista_mask(
            shape=shape,
            idx=idx,
            mask_dir=mask_dir,
        )
    raise ValueError(
        f"Unknown mask type: {mask_type}. Choose from `uniform`|`lattice`|`random`]"
    )


def uniform_mask(shape: tuple, acceleration: int, offset: int = None):
    mask = torch.zeros(shape[-2:], dtype=torch.bool)
    if offset is None:
        offset = shape[0] % 2
    mask[offset::acceleration] = 1

    return mask.view(([1] * (len(shape) - 2) + list(mask.shape)))


def random_mask(shape, acceleration, offset: float = None):
    old_shape = shape
    shape = shape[-2:]
    if offset is None:
        offset = 0.32  # From fastMRI paper 1811.08839
    gap_size = int((shape[0] * offset) // acceleration)
    start_idx = int(shape[0] // 2 - gap_size // 2)
    end_idx = int(start_idx + gap_size)

    mask = torch.zeros(shape, dtype=torch.bool)
    mask[start_idx:end_idx] = 1

    rand_idx = torch.cat(
        [torch.arange(start_idx), torch.arange(end_idx, shape[0])], dim=0
    )
    rand_idx = rand_idx[
        torch.randperm(len(rand_idx))[: shape[0] // acceleration - gap_size]
    ]
    mask[rand_idx] = 1

    assert mask.sum() == int(shape[0] // acceleration) * torch.numel(
        mask[0]
    ), f"mask sum: {mask.sum()}, expected: {(shape[0] // acceleration) * torch.numel(mask[0])} "
    return mask.view(([1] * (len(old_shape) - 2) + list(mask.shape)))


def lattice_mask(shape: tuple, acceleration: int, offset: int = None):
    r"""
    Produce 3D Undersampling mask according to lattice pattern.
    (See Tsao, 2005).

    Args:
        acceleration (int): acceleration rate.
        duration (int): size of time direction, i.e. number of frames.
        height (int): Size of phase direction.

    Returns:
        numpy.ndarray: 3D undersampling pattern
    """
    assert len(shape) == 3, "Expected at 3D shape, got {shape}"
    if offset is None:
        offset = 0
    undersampling_step = lattice_undersampling_step(acceleration)
    undersampling_mask = torch.zeros(shape)
    for i in range(acceleration):
        h_offset = (i * undersampling_step + offset) % acceleration
        undersampling_mask[i::acceleration, h_offset::acceleration] = 1
    return undersampling_mask


def lattice_undersampling_step(acceleration_rate: int) -> int:
    r"""
    Returns undersampling step according to the lattice pattern
    (See Tsao, 2015).

    Args:
        acceleration_rate (int): acceleration rate.

    Returns:
        int: required undersampling step according to the lattice pattern.
    """
    if acceleration_rate in [2, 3, 4, 6]:
        return 1
    if acceleration_rate in [5, 7, 9]:
        return 2
    if acceleration_rate in [8, 10]:
        return 3
    raise ValueError(
        f"Expected acceleration rate in range 2-10, got {acceleration_rate}"
    )


def vista_mask(
    shape: tuple,
    idx: int = None,
    mask_dir=".",
):
    assert len(shape) == 3, f"Expected 3D shape, got {len(shape)}D shape {shape}"

    files = sorted(glob(os.path.join(mask_dir, "*.mat")))
    if idx is None:
        idx = random.randint(0, len(files) - 1)
    # print(os.path.basename(files[idx]))
    assert idx < len(files), "Wrong index for mask"
    mask = loadmat(files[idx])["samp"]

    return torch.tensor(mask.T).unsqueeze(-1)
