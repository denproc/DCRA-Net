# MIT License
#
# Copyright (c) 2024 Denis Prokopenko

import yaml
import json
from tensorboard.backend.event_processing import event_accumulator
import os
from glob import glob
import matplotlib.pyplot as plt
import torch
from src.metrics import (
    mse_complex,
    nmse_complex,
    ssim_complex,
    psnr_complex,
    l1_complex,
)


def read_yml(path: str) -> dict:
    """Interface to read a yml file.

    Args:
        path (str): path to target file
    Returns:
        (dict) content of yml file
    """
    with open(path, "r") as yml_file:
        data = yaml.safe_load(yml_file)
    return data


def dump_yml(data: dict, path: str) -> None:
    """Interface to dump a yml file.

    Args:
        data (dict): data to dump
        path (str): path to target file
    """
    with open(path, "w") as yml_file:
        yaml.safe_dump(data, yml_file, sort_keys=False, default_flow_style=False)


def dump_json(data: dict, path: str) -> None:
    """Interface to dump a json file.

    Args:
        data (dict): data to dump
        path (str): path to target file
    """
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def read_tb(path):
    """Read scalars from the tensorboard file.

    Args:
        path (str): path to directory with events files of tensorboard
    Returns:
        (dict) dictionary with scalars
    """
    data = {}
    files = sorted(glob(os.path.join(path, "events.*")))
    for file in files:
        event_acc = event_accumulator.EventAccumulator(file)
        event_acc.Reload()
        for tag in sorted(event_acc.Tags()["scalars"]):
            if tag not in data.keys():
                data[tag] = {"step": [], "value": []}

            for scalar_event in event_acc.Scalars(tag):
                data[tag]["step"].append(scalar_event.step)
                data[tag]["value"].append(scalar_event.value)

    return data


LOSSES = {
    "MSE": mse_complex,
    "NMSE": nmse_complex,
    "L1": l1_complex,
    "SSIM": ssim_complex,
    "PSNR": psnr_complex,
}


def plot_pred_orig(
    predictions,
    originals,
    title="",
    loss=False,
    show=False,
    save=False,
    width=6,
    normalise=True,
):
    """Plot predictions and originals side by side.
    Args:
        predictions (torch.Tensor): Predictions. Expect 2D image or 3D set of images.
        originals (torch.Tensor): Originals.
        title (str, optional): Title of the plot. Defaults to "".
        loss (bool, optional): Whether to show losses. Defaults to False.
        show (bool, optional): Whether to show the plot. Defaults to False.
        save (bool, optional): Whether to save the plot. Defaults to False.
        width (int, optional): Width of the plot in inches. Defaults to 6.

    """
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
        originals = originals.unsqueeze(0)

    ratio = predictions.shape[-1] / predictions.shape[-2]

    ncols = 3
    if loss:
        ncols += 2
    nrows = predictions.shape[0]
    fontsize = 4 * width / ncols

    gap_w = 0.01
    gap_h = 0.2
    w = ncols + (ncols - 1) * gap_w
    size_w = 1 / w
    gap_w /= w
    h = nrows * (1 + gap_h)
    size_h = 1 / h
    gap_h /= h
    cax_rect = [0.2, -0.07, 0.6, 0.04]
    scale_w = width
    scale_h = size_w * scale_w / (size_h * ratio)
    f = plt.figure(figsize=(scale_w, scale_h))
    for i in range(nrows):
        vmax = originals[i].abs().max() if normalise else 1.0
        prediction = predictions[i] / vmax
        original = originals[i] / vmax
        show_max = max(prediction.abs().max(), original.abs().max())

        rect = (0, (nrows - i - 1) * (size_h + gap_h) + gap_h, size_w, size_h)
        ax = f.add_axes(rect)
        im = ax.imshow(prediction.abs(), cmap="gray", vmax=show_max, vmin=0)
        cax = ax.inset_axes(cax_rect)
        f.colorbar(im, cax=cax, orientation="horizontal")
        ax.axis("off")
        text = f"Prediction {title}\n{prediction.abs().max()}"
        ax.text(
            0,
            0,
            text,
            color="gray",
            fontsize=fontsize,
            horizontalalignment="left",
            verticalalignment="top",
        )

        rect = (
            size_w + gap_w,
            (nrows - i - 1) * (size_h + gap_h) + gap_h,
            size_w,
            size_h,
        )
        ax = f.add_axes(rect)
        im = ax.imshow(original.abs(), cmap="gray", vmax=show_max, vmin=0)
        cax = ax.inset_axes(cax_rect)
        f.colorbar(im, cax=cax, orientation="horizontal")
        ax.axis("off")
        text = f"Original\n{original.abs().max()}"
        ax.text(
            0,
            0,
            text,
            color="gray",
            fontsize=fontsize,
            horizontalalignment="left",
            verticalalignment="top",
        )

        rect = (
            (size_w + gap_w) * 2,
            (nrows - i - 1) * (size_h + gap_h) + gap_h,
            size_w,
            size_h,
        )
        ax = f.add_axes(rect)
        im = ax.imshow((original - prediction).abs(), cmap=None)
        cax = ax.inset_axes(cax_rect)
        f.colorbar(im, cax=cax, orientation="horizontal")
        ax.axis("off")
        text = "DIFF"
        ax.text(
            0,
            0,
            text,
            color="gray",
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=fontsize,
        )

        if loss:
            mask = original.abs() > (0 + 1e-6)
            rect = (
                (size_w + gap_w) * 3,
                (nrows - i - 1) * (size_h + gap_h) + gap_h,
                size_w,
                size_h,
            )
            ax = f.add_axes(rect)
            im = ax.imshow(mask, vmax=1, vmin=0, cmap="gray")
            ax.axis("off")
            ax.text(
                0,
                0,
                "MASK",
                color="gray",
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=fontsize,
            )

            rect = (
                (size_w + gap_w) * 4,
                (nrows - i - 1) * (size_h + gap_h) + gap_h,
                size_w,
                size_h,
            )
            ax = f.add_axes(rect)
            im = ax.imshow(torch.zeros_like(original.abs()), vmax=show_max, cmap="gray")
            ax.axis("off")

            text = f"LOSS: {'Full':>9} {'Masked':>9}"

            for loss_fn in LOSSES.keys():
                text += f"\n{loss_fn:<4}:"
                for mask_val in [None, mask[None, None, None]]:
                    loss_val = LOSSES[loss_fn](
                        input=prediction[None, None, None],
                        target=original[None, None, None],
                        reduction="mean",
                        mask=mask_val,
                    )
                    text += f" {loss_val:>9.6f}"

            ax.text(
                0,
                0,
                text,
                color="white",
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=fontsize,
                fontfamily="monospace",
            )

    if save:
        plt.savefig(f"{save}", bbox_inches="tight", pad_inches=0.0, dpi=300)
    if show:
        plt.show()
    plt.close()
