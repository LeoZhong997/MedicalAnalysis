"""
PersistentDataset, CacheDataset, LMDBDataset, and simple Dataset Tutorial and Speed Test

This tutorial shows how to accelerate PyTorch medical DL program based on
how data is loaded and preprocessed using different MONAI Dataset managers.

Dataset provides the simplest model of data loading. Each time a dataset is needed,
it is reloaded from the original datasources, and processed through the all non-random and
random transforms to generate analyzable tensors. This mechanism has the smallest memory footprint,
and the smallest temporary disk footprint.

CacheDataset provides a mechanism to pre-load all original data and apply non-random transforms
into analyzable tensors loaded in memory prior to starting analysis.
The CacheDataset requires all tensor representations of data requested to be loaded into memory at once.
The subset of random transforms is applied to the cached components before use.
This is the highest performance dataset if all data fit in core memory.

PersistentDataset processes original data sources through the non-random transforms on first use,
and stores these intermediate tensor values to an on-disk persistence representation.
The intermediate processed tensors are loaded from disk on each use for processing
by the random-transforms for each analysis request.
The PersistentDataset has a similar memory footprint to the simple Dataset,
with performance characteristics close to the CacheDataset at the expense of disk storage.
Additionally, the cost of first time processing of data is distributed across each first use.

LMDBDataset is a variant of PersistentDataset. It uses an LMDB database as the persistent backend.
"""

# Setup imports
import glob
import os
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import torch
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import (
    CacheDataset,
    Dataset,
    DataLoader,
    LMDBDataset,
    PersistentDataset,
    decollate_batch,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.utils import set_determinism


def train_process(train_ds, val_ds):
    # Define a typical PyTorch training process
    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=8,
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
    device = torch.device("cuda:0")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    loss_function = DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
        batch=True,
        smooth_nr=0.00001,
        smooth_dr=0.00001,
        lambda_dice=0.5,
        lambda_ce=0.5,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.00004,
    )

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # The initial choice is 300~600 epoches, in order to save time, we use 30 to demonstrate
    # the comparison of dataset classes
    max_epochs = 30

    val_interval = 1  # do validation for every epoch
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    epoch_times = []
    total_start = time.time()
    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size},"
                f" train_loss: {loss.item():.4f}"
                f" step time: {(time.time() - step_start):.4f}"
            )
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        os.path.join(root_dir, "best_metric_model.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current"
                    f" mean dice: {metric:.4f}"
                    f" best mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
        print(f"time of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
        epoch_times.append(time.time() - epoch_start)

    print(
        f"train completed, best_metric: {best_metric:.4f}"
        f" at epoch: {best_metric_epoch}"
        f" total time: {(time.time() - total_start):.4f}"
    )
    return (
        max_epochs,
        time.time() - total_start,
        epoch_loss_values,
        metric_values,
        epoch_times,
    )


def transformations():
    """
    Setup transforms for training and validation

    Deterministic transforms during training:
    * LoadImaged
    * EnsureChannelFirstd
    * Spacingd
    * Orientationd
    * ScaleIntensityRanged

    Non-deterministic transforms:
    * RandCropByPosNegLabeld

    All the validation transforms are deterministic.
    The results of all the deterministic transforms will be cached to accelerate training.
    :return: train_transforms, val_transforms
    """
    train_transforms = Compose(
        [
            # LoadImaged with image_only=True is to return the MetaTensors
            # the additional metadata dictionary is not returned.
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # randomly crop out patch samples from big
            # image based on pos / neg ratio
            # the image centers of negative samples
            # must be in valid image area
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )

    # NOTE: No random cropping in the validation data,
    # we will evaluate the entire image using a sliding window.
    val_transforms = Compose(
        [
            # LoadImaged with image_only=True is to return the MetaTensors
            # the additional metadata dictionary is not returned.
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )
    return train_transforms, val_transforms


if __name__ == "__main__":
    print_config()

    # Setup data directory
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)

    # Download dataset
    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

    compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
    data_dir = os.path.join(root_dir, "Task09_Spleen")
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir, md5)

    # Set MSD Spleen dataset path
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in
                  zip(train_images, train_labels)]
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]

    # 1. Enable deterministic training and regular `Dataset`
    set_determinism(seed=0)

    train_trans, val_trans = transformations()
    train_ds = Dataset(data=train_files, transform=train_trans)
    val_ds = Dataset(data=val_files, transform=val_trans)

    (
        max_epochs,
        total_time,
        epoch_loss_values,
        metric_values,
        epoch_times,
    ) = train_process(train_ds, val_ds)
    print(f"total training time of {max_epochs} epochs" f" with regular Dataset: {total_time:.4f}")

    # 2. Enable deterministic training and `PersistentDataset`
    persistent_cache = os.path.join(tempfile.mkdtemp(), "persistent_cache")

    set_determinism(seed=0)
    train_trans, val_trans = transformations()
    train_persitence_ds = PersistentDataset(data=train_files, transform=train_trans, cache_dir=persistent_cache)
    val_persitence_ds = PersistentDataset(data=val_files, transform=val_trans, cache_dir=persistent_cache)

    (
        persistence_epoch_num,
        persistence_total_time,
        persistence_epoch_loss_values,
        persistence_metric_values,
        persistence_epoch_times,
    ) = train_process(train_persitence_ds, val_persitence_ds)
    print(
        f"total training time of {persistence_epoch_num}"
        f" epochs with persistent storage Dataset: {persistence_total_time:.4f}"
    )

    # 3. Enable deterministic training and `LMDBDataset`
    LMDB_cache = os.path.join(tempfile.mkdtemp(), "lmdb_cache")

    set_determinism(seed=0)
    train_trans, val_trans = transformations()
    lmdb_init_start = time.time()
    train_lmdb_ds = LMDBDataset(
        data=train_files, transform=train_trans, cache_dir=LMDB_cache, lmdb_kwargs={"map_async": True}
    )
    val_lmdb_ds = LMDBDataset(data=val_files, transform=val_trans, cache_dir=LMDB_cache,
                              lmdb_kwargs={"map_async": True})
    lmdb_init_time = time.time() - lmdb_init_start

    (
        lmdb_epoch_num,
        lmdb_total_time,
        lmdb_epoch_loss_values,
        lmdb_metric_values,
        lmdb_epoch_times,
    ) = train_process(train_lmdb_ds, val_lmdb_ds)
    print(f"total training time of {lmdb_epoch_num}" f" epochs with LMDB storage Dataset: {lmdb_total_time:.4f}")

    # 4. Enable deterministic training and `CacheDataset`
    """Precompute all non-random transforms of original data and store in memory.
    When `runtime_cache="processes"` the cache initialization time `cache_init_time` is negligible.
    Set `runtime_cache=False` to enable precomputing cache."""
    set_determinism(seed=0)
    train_trans, val_trans = transformations()
    cache_init_start = time.time()
    cache_train_ds = CacheDataset(
        data=train_files, transform=train_trans, cache_rate=1.0, runtime_cache="processes", copy_cache=False
    )
    cache_val_ds = CacheDataset(
        data=val_files, transform=val_trans, cache_rate=1.0, runtime_cache="processes", copy_cache=False
    )
    cache_init_time = time.time() - cache_init_start

    (
        cache_epoch_num,
        cache_total_time,
        cache_epoch_loss_values,
        cache_metric_values,
        cache_epoch_times,
    ) = train_process(cache_train_ds, cache_val_ds)
    print(f"total training time of {cache_epoch_num}" f" epochs with CacheDataset: {cache_total_time:.4f}")

    # Plot training loss and validation metrics
    plt.figure("train", (12, 18))

    plt.subplot(4, 2, 1)
    plt.title("Regular Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.grid(alpha=0.4, linestyle=":")
    plt.plot(x, y, color="red")

    plt.subplot(4, 2, 2)
    plt.title("Regular Val Mean Dice")
    x = [i + 1 for i in range(len(metric_values))]
    y = cache_metric_values
    plt.xlabel("epoch")
    plt.grid(alpha=0.4, linestyle=":")
    plt.plot(x, y, color="red")

    plt.subplot(4, 2, 3)
    plt.title("PersistentDataset Epoch Average Loss")
    x = [i + 1 for i in range(len(persistence_epoch_loss_values))]
    y = persistence_epoch_loss_values
    plt.xlabel("epoch")
    plt.grid(alpha=0.4, linestyle=":")
    plt.plot(x, y, color="blue")

    plt.subplot(4, 2, 4)
    plt.title("PersistentDataset Val Mean Dice")
    x = [i + 1 for i in range(len(persistence_metric_values))]
    y = persistence_metric_values
    plt.xlabel("epoch")
    plt.grid(alpha=0.4, linestyle=":")
    plt.plot(x, y, color="blue")

    plt.subplot(4, 2, 5)
    plt.title("LMDBDataset Epoch Average Loss")
    x = [i + 1 for i in range(len(lmdb_epoch_loss_values))]
    y = lmdb_epoch_loss_values
    plt.xlabel("epoch")
    plt.grid(alpha=0.4, linestyle=":")
    plt.plot(x, y, color="yellow")

    plt.subplot(4, 2, 6)
    plt.title("LMDBDataset Val Mean Dice")
    x = [i + 1 for i in range(len(lmdb_metric_values))]
    y = lmdb_metric_values
    plt.xlabel("epoch")
    plt.grid(alpha=0.4, linestyle=":")
    plt.plot(x, y, color="yellow")

    plt.subplot(4, 2, 7)
    plt.title("Cache Epoch Average Loss")
    x = [i + 1 for i in range(len(cache_epoch_loss_values))]
    y = cache_epoch_loss_values
    plt.xlabel("epoch")
    plt.grid(alpha=0.4, linestyle=":")
    plt.plot(x, y, color="green")

    plt.subplot(4, 2, 8)
    plt.title("Cache Val Mean Dice")
    x = [i + 1 for i in range(len(cache_metric_values))]
    y = cache_metric_values
    plt.xlabel("epoch")
    plt.grid(alpha=0.4, linestyle=":")
    plt.plot(x, y, color="green")

    plt.show()

    # Plot total time and every epoch time
    plt.figure("train", (12, 6))

    plt.subplot(1, 2, 1)
    plt.title(f"Total Train Time ({max_epochs} epochs)")
    plt.bar("regular", total_time, 1, label="Regular Dataset", color="red")
    plt.bar(
        "lmdb",
        lmdb_init_time + lmdb_total_time,
        1,
        label="LMDB cache init",
        color="yellow",
    )
    plt.bar("lmdb", lmdb_total_time, 1, label="LMDB Dataset", color="orange")
    plt.bar(
        "persistent",
        persistence_total_time,
        1,
        label="Persistent Dataset",
        color="blue",
    )
    plt.bar(
        "cache",
        cache_init_time + cache_total_time,
        1,
        label="Cache Dataset",
        color="green",
    )
    if cache_init_time > 1:
        plt.bar("cache", cache_init_time, 1, label="Cache Init", color="orange")
    plt.ylabel("secs")
    plt.grid(alpha=0.4, linestyle=":")
    plt.legend(loc="best")

    plt.subplot(1, 2, 2)
    plt.title("Epoch Time")
    x = [i + 1 for i in range(len(epoch_times))]
    plt.xlabel("epoch")
    plt.ylabel("secs")
    plt.plot(x, epoch_times, label="Regular Dataset", color="red")
    plt.plot(x, persistence_epoch_times, label="Persistent Dataset", color="blue")
    plt.plot(x, lmdb_epoch_times, label="LMDB Dataset", color="yellow")
    plt.plot(x, cache_epoch_times, label="Cache Dataset", color="green")
    plt.grid(alpha=0.4, linestyle=":")
    plt.legend(loc="best")

    plt.show()

    # Cleanup data directory
    if directory is None:
        shutil.rmtree(root_dir)