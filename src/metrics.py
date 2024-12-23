# MIT License
#
# Copyright (c) 2024 Denis Prokopenko

import torch
from piq import ssim


def mse_complex(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """MSE loss for complex tensors. Follows pytorch notation.

    Args:
        input: Complex valued input tensor.
        target: Complex valued target tensor.
        reduction: Reduction method. Can be `none`, `sum` or `mean`.
        mask: Mask tensor.


    Returns:
        Loss tensor.
    """
    delta = input - target
    mse = (delta * delta.conj()).abs()
    if mask is not None:
        mse = mse * mask

    # reduction over all dimensions
    if reduction == "none":
        return mse
    if reduction == "sum":
        return mse.sum()
    if reduction == "mean":
        if mask is None:
            return mse.mean()
        else:
            return mse.sum() / mask.sum()

    raise NotImplementedError("Unknown key for reduction. Use `none`, `sum` or `mean`")


def nmse_complex(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """NMSE loss for complex tensors. Follows pytorch notation.

    Args:
        input: Complex valued input tensor. Shape (N, ...).
        target: Complex valued target tensor. Shape (N, ...).
        reduction: Reduction method. Can be `none`, `sum` or `mean`.
        mask: Mask tensor.

    Returns:
        Loss tensor.
    """
    mse = mse_complex(input, target, reduction="none", mask=mask)
    norm = mse_complex(target, torch.zeros_like(target), reduction="none", mask=mask)

    averaging_dims = tuple(range(2, len(input.shape)))
    nmse = mse.sum(dim=averaging_dims, keepdim=True) / norm.sum(
        dim=averaging_dims, keepdim=True
    )

    # reduction over batch dimension
    if reduction == "none":
        return nmse
    if reduction == "sum":
        return nmse.sum()
    if reduction == "mean":
        return nmse.mean()

    raise NotImplementedError("Unknown key for reduction. Use `none`")


def ssim_complex(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """SSIM for complex input computed over magnituges. Follows pytorch notation.

    Args:
        input: Complex valued input tensor. (N, C, T, H, W)
        target: Complex valued target tensor. (N, C, T, H, W)
        reduction: Reduction method. Can be `none`, `sum` or `mean`.
        mask: Mask tensor.

    Returns:
        Loss tensor.
    """
    input_magnitude = input.abs()
    target_magnitude = target.abs()
    if mask is not None:
        input_magnitude *= mask
        target_magnitude *= mask

    max_val = max(input_magnitude.max(), target_magnitude.max())
    scores = []
    for frame in range(input_magnitude.shape[2]):
        scores.append(
            ssim(
                x=input_magnitude[:, :, frame],
                y=target_magnitude[:, :, frame],
                data_range=max_val,
                reduction="none",
            )
        )
    scores = torch.stack(scores, dim=-1)

    if reduction == "none":
        return scores.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    if reduction == "sum":
        return scores.sum()
    return scores.mean()


def psnr_complex(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "none",
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """PSNR for complex input computed over magnituges. Follows pytorch notation.

    Args:
        input: Complex valued input tensor.
        target: Complex valued target tensor.
        reduction: Reduction method. Can be `none`, `sum` or `mean`.
        mask: Mask tensor.

    Returns:
        Loss tensor.
    """
    input_magnitude = input.abs()
    target_magnitude = target.abs()
    if mask is not None:
        input_magnitude = input_magnitude * mask
        target_magnitude = target_magnitude * mask

    max_val = max(input_magnitude.max(), target_magnitude.max())
    mse = mse_complex(input, target, reduction="none", mask=mask)

    averaging_dims = tuple(range(1, len(input_magnitude.shape)))
    if mask is None:
        mse = mse.mean(dim=averaging_dims, keepdim=True)
    else:
        mse = mse.sum(dim=averaging_dims, keepdim=True) / mask.sum(
            dim=averaging_dims, keepdim=True
        )

    EPS = torch.finfo(mse.dtype).eps
    score = 20.0 * torch.log10(max_val) - 10.0 * torch.log10(mse + EPS)

    # reduction over batch dimension
    if reduction == "none":
        return score
    if reduction == "sum":
        return score.sum()
    if reduction == "mean":
        return score.mean()
    raise NotImplementedError("Unknown key for reduction. Use `none`, `sum` or `mean`")


def l1_complex(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """L1 loss for complex tensors. Follows pytorch notation.

    Args:
        input: Complex valued input tensor.
        target: Complex valued target tensor.
        reduction: Reduction method. Can be `none`, `sum` or `mean`.
        mask: Mask tensor.


    Returns:
        Loss tensor.
    """
    delta = input - target
    l1_loss = delta.abs()
    if mask is not None:
        l1_loss = l1_loss * mask

    # reduction over all dimensions
    if reduction == "none":
        return l1_loss
    if reduction == "sum":
        return l1_loss.sum()
    if reduction == "mean":
        if mask is None:
            return l1_loss.mean()
        else:
            return l1_loss.sum() / mask.sum()

    raise NotImplementedError("Unknown key for reduction. Use `none`, `sum` or `mean`")


loss_func = {
    "MSE": mse_complex,
    "NMSE": nmse_complex,
    "SSIM": ssim_complex,
    "PSNR": psnr_complex,
    "L1": l1_complex,
}
