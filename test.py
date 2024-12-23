# MIT License
#
# Copyright (c) 2024 Denis Prokopenko

from torchvision import transforms
import torch
import argparse
import os
from tqdm.auto import tqdm
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


def parse_args():
    """
    Parse the arguments provided with call.

    Returns:
        Namespace() with arguments
    """
    parser = argparse.ArgumentParser(description="Script to run testing")

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
        "--verbose",
        help="verbose mode",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--checkpoint_path",
        help="checkpoint_to_load",
        default=None,
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

    inference_dir = os.path.join(config["save_dir"], "inference")
    os.makedirs(inference_dir, exist_ok=True)

    predictions_dir = os.path.join(inference_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

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

    test_dataset = PairedDataset(
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

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
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

    checkpoint = torch.load(
        config["checkpoint_path"],
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    test_losses = {}
    test_losses_masked = {}

    for k in loss_func.keys():
        test_losses[k] = []
        test_losses_masked[k] = []

    model.eval()
    with torch.inference_mode():

        for test_iter, test_batch in tqdm(
            enumerate(test_dataloader),
            disable=not config["verbose"],
            total=len(test_dataloader),
            desc="Validation",
        ):
            # Undersampling

            test_kspace, test_mask = [item.cuda() for item in test_batch[0]][:]

            test_target = test_batch[1][0].cuda()
            test_target = ToReal(batched=True)(test_target)

            test_undersampled = test_kspace * test_mask
            test_undersampled = AddChannel(dim=1)(test_undersampled)

            test_pred = model(test_undersampled)

            test_undersampled = tensor2complex(test_undersampled)
            test_pred = tensor2complex(test_pred)
            test_target = tensor2complex(test_target)

            # Move to Image-Time for plotting
            test_undersampled = ToImage()(test_undersampled)
            test_pred = ToTime()(test_pred)

            for k in loss_func.keys():
                test_losses[k].extend(
                    loss_func[k](test_pred, test_target, reduction="none")
                    .mean(dim=(1, 2, 3, 4))
                    .cpu()
                )
                mask = test_target.abs() > (0 + 1e-6)
                test_losses_masked[k].extend(
                    loss_func[k](test_pred, test_target, mask=mask, reduction="none")
                    .mean(dim=(1, 2, 3, 4))
                    .cpu()
                )

            for b_idx in range(test_pred.size(0)):
                data_to_save = {
                    "input": test_undersampled[b_idx, 0],
                    "target": test_target[b_idx, 0],
                    "prediction": test_pred[b_idx, 0],
                }
                torch.save(
                    data_to_save,
                    os.path.join(
                        predictions_dir,
                        f"{test_iter * config['batch_size'] +  b_idx:04d}.pt",
                    ),
                )
    torch.save(
        {"test_losses": test_losses, "test_losses_masked": test_losses_masked},
        os.path.join(inference_dir, "test_losses.pt"),
    )

    print("Done")


if __name__ == "__main__":
    config = parse_args()
    print("Strating the script")
    print(config)
    main(config)
