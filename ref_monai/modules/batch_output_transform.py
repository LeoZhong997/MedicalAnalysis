"""
How to use batch_transform and output_transform
- MONAI defines many ignite style Event handlers to provide general purpose logic during training or evaluation,
like: stats logging, MLFlow tracking, TensorBoard, metrics, etc.
- Usually, the handler has a callable arg named batch_transform or output_transform to help prepare expected data
from ignite engine.state.batch or engine.state.output.
- This tutorial shows examples about how to set batch_transform and output_transform for different kinds of handlers
with different data shapes.
"""

import logging
import os
import sys
import tempfile
from glob import glob
import shutil

import nibabel as nib
import numpy as np
import torch
from ignite.metrics import Accuracy

from monai.config import print_config
from monai.data import CacheDataset, create_test_image_3d, DataLoader
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    MeanDice,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    EnsureChannelFirstd,
    AsDiscreted,
    Compose,
    KeepLargestConnectedComponentd,
    LoadImaged,
    RandCropByPosNegLabeld,
    ScaleIntensityd,
    EnsureTyped,
)
from monai.utils import get_torch_version_tuple


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    print_config()

    # Setup data directory
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else os.path.join(directory, 'SyntheticImages')
    os.makedirs(root_dir, exist_ok=True)
    print(f"root dir is: {root_dir}")

    # Prepare synthetic data for test
    if len(os.listdir(root_dir)) == 0:
        for i in range(40):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(root_dir, f"img{i:d}.nii.gz"))
            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(root_dir, f"seg{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(root_dir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(root_dir, "seg*.nii.gz")))
    train_files = [{"image": img, "label": seg} for img, seg in zip(images[:20], segs[:20])]
    val_files = [{"image": img, "label": seg} for img, seg in zip(images[-20:], segs[-20:])]

    # Setup train / val transforms, dataset, DataLoader, post transforms
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim=-1),
            ScaleIntensityd(keys="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim=-1),
            ScaleIntensityd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    # create a training data loader
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
    # create a validation data loader
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    val_post_transforms = Compose(
        [
            EnsureTyped(keys="pred"),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
        ]
    )

    # Setup network, optimizer, loss function, etc.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss = DiceLoss(sigmoid=True)
    opt = torch.optim.Adam(net.parameters(), 3e-4)

    # Setup handlers and metrics for train and validation
    val_handlers = [
        # no stats for iteration of validation
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir="./runs/", output_transform=lambda x: None),
        TensorBoardImageHandler(
            log_dir="./runs/",
            batch_transform=from_engine(["image", "label"]),
            output_transform=from_engine(["pred"]),
        ),
        CheckpointSaver(save_dir="./runs/", save_dict={"net": net}, save_key_metric=True),
    ]

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5),
        postprocessing=val_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=True, output_transform=from_engine(["pred", "label"]))
        },
        additional_metrics={"val_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},
        val_handlers=val_handlers,
        # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP evaluation
        amp=True if get_torch_version_tuple() >= (1, 6) else False,
    )

    train_handlers = [
        ValidationHandler(validator=evaluator, interval=2, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        TensorBoardStatsHandler(
            log_dir="./runs/",
            tag_name="train_loss",
            output_transform=from_engine(["loss"], first=True),
        ),
    ]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=5,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=loss,
        inferer=SimpleInferer(),
        train_handlers=train_handlers,
        # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP training
        amp=True if get_torch_version_tuple() >= (1, 6) else False,
    )

    # Execute training to verify the settings
    trainer.run()

    # Cleanup data directory
    if directory is None:
        shutil.rmtree(root_dir)
