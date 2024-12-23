# MIT License
#
# Copyright (c) 2024 Denis Prokopenko

from torchvision import transforms
import torch
from torch.optim import Adam
import argparse
import os
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.append(os.path.abspath("."))
from src.datasets import PairedDataset
from src.metrics import loss_func
from src.dcranet import DCRANet
from src.transforms import (
    CutFrames,
    ToTime,
    ToFrequency,
    ToTensor,
    ToImage,
    ToKSpace,
    ToReal,
    ToComplex,
    AddChannel,
    tensor2complex,
)
from sklearn.model_selection import train_test_split
from src.utils import dump_yml
from src.utils import plot_pred_orig
from torchvision.utils import save_image


def parse_args():
    """
    Parse the arguments provided with call.

    Returns:
        Namespace() with arguments
    """
    parser = argparse.ArgumentParser(description="Script to run training")

    parser.add_argument(
        "--backbone",
        help="model",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--dc_mode",
        help="Data consistency mode for old unets",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--image_size", help="resize data to the given size", default=None, type=int
    )
    parser.add_argument(
        "--n_frames",
        help="resize data to the given size in time dim",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--representation_time",
        help="data representation to use: frequency or time",
        default="frequency",
        type=str,
    )
    parser.add_argument(
        "--representation_space",
        help="data representation to use: image or kspace",
        default="image",
        type=str,
    )
    parser.add_argument(
        "--out_channels", help="number of output channels", default=2, type=int
    )
    parser.add_argument(
        "--in_channels", help="number of input channels", default=2, type=int
    )
    parser.add_argument("--batch_size", help="batch size", default=1, type=int)

    parser.add_argument(
        "--start_epoch", help="start with given epoch", default=0, type=int
    )
    parser.add_argument(
        "--n_epochs", help="number of epochs to train", default=10, type=int
    )

    parser.add_argument(
        "--data_dir",
        help="data directory",
        default=None,
    )

    parser.add_argument(
        "--save_dir",
        help="path to save results",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--mask_dir",
        help="path to save results",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--tensorboard_dir",
        help="path to save data for tensorboard",
        type=str,
        default="./tensorboard_data",
    )
    parser.add_argument("--acceleration", help="acceleration rate", default=8, type=int)

    parser.add_argument(
        "--pattern", help="undersampling attern", default="lattice", type=str
    )

    parser.add_argument("--mask_idx", help="vista mask index", default=None, type=str)

    parser.add_argument("--mask_ucoef", help="vista mask index", default=None, type=str)

    parser.add_argument("--seed", help="random seed", default=42, type=int)

    parser.add_argument("--dim", help="inital emb dimension", default=64, type=int)
    parser.add_argument(
        "--dim_mults",
        help="multiplicator for unet-based backbones",
        default=(1, 2, 4),
        type=tuple,
    )

    parser.add_argument(
        "--lr",
        help="learning rate",
        default=1e-4,
        type=float,
    )

    parser.add_argument(
        "--loss_type",
        help="loss type",
        default="L1",
        type=str,
    )

    parser.add_argument(
        "--verbose",
        help="verbose mode",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    return vars(args)


MODELS = {"DCRA-Net": DCRANet}
DC_MODE = {
    "none": 1.0,
    "force": 0.0,
    "learn": "learn",
}


def main(config):
    assert config["backbone"] in MODELS.keys(), "Check the backbone"
    if config["dc_mode"] is not None:
        config["dc_mode"] = DC_MODE[config["dc_mode"]]
    checkpoints_dir = os.path.join(config["save_dir"], "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    dump_yml(
        config,
        os.path.join(
            config["save_dir"],
            f"config_{config['start_epoch']}_{config['n_epochs']}.yml",
        ),
    )

    samples_dir = os.path.join(config["save_dir"], "predictions")
    os.makedirs(samples_dir, exist_ok=True)
    origs_dir = os.path.join(config["save_dir"], "origs")
    os.makedirs(origs_dir, exist_ok=True)
    custom_dir = os.path.join(config["save_dir"], "custom")
    os.makedirs(custom_dir, exist_ok=True)

    inputs_dir = os.path.join(config["save_dir"], "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    assert config["representation_time"] in ["time", "frequency"]
    assert config["representation_space"] == "image"

    transform = transforms.Compose(
        [
            ToTensor(),
            CutFrames(frames=config["n_frames"]),
            ToImage(),
            ToReal(),
            transforms.Resize(config["image_size"], antialias=True),
            transforms.CenterCrop(config["image_size"]),
            ToComplex(),
            ToKSpace(),
        ]
    )

    train_dataset = PairedDataset(
        data_dir=config["data_dir"],
        pattern=config["pattern"],
        transform=transform,
        frames=config["n_frames"],
        img_size=config["image_size"],
        acceleration=config["acceleration"],
        u_coef=config["mask_ucoef"],
        mask_dir=config["mask_dir"],
    )
    val_dataset = PairedDataset(
        data_dir=config["data_dir"],
        transform=transform,
        pattern=config["pattern"],
        frames=config["n_frames"],
        img_size=config["image_size"],
        acceleration=config["acceleration"],
        u_coef=config["mask_ucoef"],
        mask_id=1,
        mask_dir=config["mask_dir"],
    )

    files = train_dataset.files
    patient_ids = set([os.path.basename(file).split("_")[0] for file in files])

    train_ids, val_ids = train_test_split(
        sorted(list(patient_ids)), test_size=0.1, random_state=42
    )

    train_dataset.files = []
    val_dataset.files = []
    for file in files:
        if os.path.basename(file).split("_")[0] in train_ids:
            train_dataset.files.append(file)
        else:
            val_dataset.files.append(file)

    print(sorted(train_ids))
    print(len(train_dataset))
    print(sorted(val_ids))
    print(len(val_dataset))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["batch_size"] % (os.cpu_count() - 1),
        pin_memory=True,
        drop_last=False,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["batch_size"] % (os.cpu_count() - 1),
        pin_memory=True,
        drop_last=False,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = MODELS[config["backbone"]](
        dim=config["dim"],
        channels=config["in_channels"],
        out_dim=config["out_channels"],
        dim_mults=config["dim_mults"],
        representation_time=config["representation_time"],
        norm_fft=None,
        dc_mode=config["dc_mode"],
    )

    print(f"{type(model).__name__} created")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config["lr"])

    if config["start_epoch"] > 0:
        data = torch.load(
            os.path.join(
                checkpoints_dir, f"checkpoint-{config['start_epoch']-1:02d}.pt"
            ),
            map_location=device,
        )
        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["opt"])

    writer = SummaryWriter(
        log_dir=os.path.join(
            config["tensorboard_dir"], os.path.basename(config["save_dir"])
        )
    )

    training_losses = {
        "mse": torch.nn.MSELoss,
        "l1": torch.nn.L1Loss,
        "smothL1": torch.nn.SmoothL1Loss,
        "huber": torch.nn.HuberLoss,
    }

    criterion = training_losses[config["loss_type"].lower()](reduction="mean")

    losses = []
    val_losses = {}
    val_losses_masked = {}

    for k in loss_func.keys():
        val_losses[k] = []
        val_losses_masked[k] = []

    for epoch in range(config["start_epoch"], config["n_epochs"]):
        for step, batch in tqdm(
            enumerate(train_dataloader),
            disable=not config["verbose"],
            total=len(train_dataloader),
            desc="Train Iterations",
        ):
            optimizer.zero_grad()

            # Undersampling
            kspace, mask = [item.cuda() for item in batch[0]][:]
            target = batch[1][0].cuda()

            undersampled = kspace * mask
            undersampled = AddChannel(dim=1)(undersampled)

            target = ToFrequency()(target)
            target = ToReal(batched=True)(target)

            if step == 0 and config["verbose"]:
                print(undersampled.abs().mean(), target.abs().mean())
                print(undersampled.abs().max(), target.abs().max())
                print(undersampled.size(), target.size())

            loss = criterion(
                input=model(undersampled),
                target=target,
            )

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            writer.add_scalar(
                f"Loss_{config['loss_type']}/Train/Iteration",
                losses[-1],
                step + epoch * len(train_dataloader),
            )

        offset = -len(train_dataloader)
        current_mean = sum(losses[offset:]) / len(train_dataloader)
        writer.add_scalar(
            f"Loss_{config['loss_type']}/Train/Epoch", current_mean, epoch
        )

        with torch.inference_mode():
            for v_iter, v_batch in tqdm(
                enumerate(val_dataloader),
                disable=not config["verbose"],
                total=len(val_dataloader),
                desc="Validation",
            ):
                # Undersampling

                v_kspace, v_mask = [item.cuda() for item in v_batch[0]][:]

                v_target = v_batch[1][0].cuda()
                v_target = ToReal(batched=True)(v_target)

                v_undersampled = v_kspace * v_mask
                v_undersampled = AddChannel(dim=1)(v_undersampled)

                v_pred = model(v_undersampled)

                v_undersampled = tensor2complex(v_undersampled)
                v_pred = tensor2complex(v_pred)
                v_target = tensor2complex(v_target)

                # Move to Image-Time for plotting
                v_undersampled = ToImage()(v_undersampled)
                v_pred = ToTime()(v_pred)

                for k in loss_func.keys():
                    val_losses[k].extend(
                        loss_func[k](v_pred, v_target, reduction="none")
                        .mean(dim=(1, 2, 3, 4))
                        .cpu()
                    )
                    mask = v_target.abs() > (0 + 1e-6)
                    val_losses_masked[k].extend(
                        loss_func[k](v_pred, v_target, mask=mask, reduction="none")
                        .mean(dim=(1, 2, 3, 4))
                        .cpu()
                    )

                    for i in range(v_target.size(0), 0, -1):
                        writer.add_scalar(
                            f"{k.upper()}_Loss/Validation/Samples",
                            val_losses[k][-i],
                            epoch * len(val_dataset)
                            + (v_iter + 1) * config["batch_size"]
                            - i,
                        )
                        writer.add_scalar(
                            f"{k.upper()}_Loss_Masked/Validation/Samples",
                            val_losses_masked[k][-i],
                            epoch * len(val_dataset)
                            + (v_iter + 1) * config["batch_size"]
                            - i,
                        )

                if v_iter == 0:
                    n_samples = 4
                    v_pred = v_pred.cpu()
                    v_target = v_target.cpu()
                    v_undersampled = v_undersampled.cpu()
                    plot_pred_orig(
                        predictions=v_pred[:, 0].reshape(-1, *v_pred.shape[-2:])[
                            :n_samples
                        ],
                        originals=v_target[:, 0].reshape(-1, *v_pred.shape[-2:])[
                            :n_samples
                        ],
                        save=os.path.join(custom_dir, f"sample-{epoch:02d}.png"),
                        title=f"{os.path.basename(config['save_dir'])}\n{epoch:02d}",
                        loss=True,
                        normalise=False,
                    )

                    save_image(
                        v_pred[:, 0]
                        .reshape(-1, 1, *v_pred.size()[-2:])[:n_samples]
                        .abs(),
                        os.path.join(samples_dir, f"sample-{epoch:02d}.png"),
                        nrow=4,
                        normalize=True,
                        scale_each=True,
                    )

                    save_image(
                        v_undersampled[:, 0]
                        .reshape(-1, 1, *v_undersampled.size()[-2:])[:n_samples]
                        .abs(),
                        os.path.join(inputs_dir, f"input-{epoch:02d}.png"),
                        nrow=4,
                        normalize=True,
                        scale_each=True,
                    )
                    save_image(
                        v_target[:, 0]
                        .reshape(-1, 1, *v_target.size()[-2:])[:n_samples]
                        .abs(),
                        os.path.join(origs_dir, f"orig-{epoch:02d}.png"),
                        nrow=4,
                        normalize=True,
                        scale_each=True,
                    )

        for k in val_losses.keys():
            offset = -len(val_dataset)
            current_mean = sum(val_losses[k][offset:]) / len(val_dataset)
            writer.add_scalar(f"{k.upper()}_Loss/Validation/Epoch", current_mean, epoch)
            current_mean = sum(val_losses_masked[k][offset:]) / len(val_dataset)
            writer.add_scalar(
                f"{k.upper()}_Loss_Masked/Validation/Epoch", current_mean, epoch
            )

        data = {
            "step": None,
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
        }
        torch.save(data, os.path.join(checkpoints_dir, f"checkpoint-{epoch:02d}.pt"))

    print("Done")


if __name__ == "__main__":
    # config
    config = parse_args()
    print("Strating the script")
    print(config)
    main(config)
