# MIT License
#
# Copyright (c) 2024 Denis Prokopenko

import torch
from typing import Union


class DataConsistencyKSpace(torch.nn.Module):
    r"""
    PyTorch Data Consistency Layer.
    Data Consistency Layer works with 4D :math:`(N, C, H, W)`
    and 5D :math:`(N, C, T, H, W)`.
    The k-space lines are mixed for last 2 dimensions.

    Args:
        noise_lvl (float): noise level (0, 1). Zero means inserting the data,
            One ignores the consistency term.

    """

    def __init__(self, dc_mode: Union[str, float]):
        super().__init__()

        if dc_mode == "learn":
            noise_lvl = 0.0
            trainable = True
        elif isinstance(dc_mode, float):
            noise_lvl = dc_mode
            trainable = False
        else:
            raise ValueError(
                "Unexpected data conistency mode. ", "Use `learn` or float."
            )

        assert (
            0.0 <= noise_lvl <= 1.0
        ), f"Expected non-negative noise level, got {noise_lvl}"

        self.noise_lvl = torch.nn.Parameter(
            torch.tensor(noise_lvl, dtype=torch.float), requires_grad=trainable
        )

    def forward(
        self, k_prediction: torch.Tensor, k_x: torch.Tensor, k_mask: torch.Tensor
    ) -> torch.Tensor:
        r"""

        Args:
            k_prediction (torch.Tensor): 4D or 5D prediction by the previous
                layers.
            k_x (torch.Tensor): 4D or 5D given values available with input.
            k_mask (torch.Tensor): 4D or 5D mask for non-zero locations in
                the input.

        Returns:
            torch.Tensor: Tensor with data consistency.

        """
        return k_prediction * (1 - k_mask * (1 - self.noise_lvl)) + k_x * k_mask * (
            1 - self.noise_lvl
        )
