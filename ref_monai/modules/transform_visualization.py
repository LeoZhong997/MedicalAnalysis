"""
Visualization for images during transform augmentation

This tutorial shows how to visualize image with `matplotlib` based on MONAI `matshow3d` API,
with `TensorBoard` based on MONAI `plot_2d_or_3d_image` API,
with `itkwidgets` for interactive visualization.
Also shows how to blend 2 images with the same shape then use `matplotlib` to plot
`image`, `label`, `blend result` accordingly.
"""

# Setup imports
import torch
from torch.utils.tensorboard import SummaryWriter
from monai.utils import first, set_determinism
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.data import DataLoader, Dataset
from monai.config import print_config
from monai.apps import download_and_extract
from monai.visualize import blend_images, matshow3d, plot_2d_or_3d_image
import tempfile
import shutil
import os
import glob
import matplotlib.pyplot as plt
from itkwidgets import view


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

    # Set deterministic training for reproducibility
    set_determinism(seed=0)

    # Setup MONAI transforms
    transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="PLS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )

    # Get the transform results in DataLoader
    check_ds = Dataset(data=data_dicts, transform=transform)
    check_loader = DataLoader(check_ds, batch_size=1)
    data = first(check_loader)
    print(f"image shape: {data['image'].shape}, label shape: {data['label'].shape}")

    # Visualize the image with MONAI `matshow3d` API
    matshow3d(
        volume=data["image"],
        fig=None,
        title="input image",
        figsize=(100, 100),
        every_n=10,
        frame_dim=-1,
        show=True,
        cmap="gray",
    )

    # Visualize the 3D image in TensorBoard as GIF images
    tb_dir = os.path.join(root_dir, "tensorboard")
    plot_2d_or_3d_image(data=data["image"], step=0, writer=SummaryWriter(log_dir=tb_dir), frame_dim=-1)

    # Leverage `itkwidgets` to interactively visualize `image` and `label`
    view(image=data["image"][0, 0, :, :, :] * 255, label_image=data["label"][0, 0, :, :, :] * 255, gradient_opacity=0.4)

    # Blend the image and label to check the segmentation region
    ret = blend_images(image=data["image"][0], label=data["label"][0], alpha=0.5, cmap="hsv", rescale_arrays=False)
    print(ret.shape)

    for i in range(5, 10):
        # plot the slice 50 - 100 of image, label and blend result
        slice_index = 10 * i
        plt.figure("blend image and label", (12, 4))
        plt.subplot(1, 3, 1)
        plt.title(f"image slice {slice_index}")
        plt.imshow(data["image"][0, 0, :, :, slice_index], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label slice {slice_index}")
        plt.imshow(data["label"][0, 0, :, :, slice_index])
        plt.subplot(1, 3, 3)
        plt.title(f"blend slice {slice_index}")
        # switch the channel dim to the last dim
        plt.imshow(torch.moveaxis(ret[:, :, :, slice_index], 0, -1))
        plt.show()

    # Visualize the image with TensorBoardPlugin3D
    # Create directories to use as TensorBoard log directories
    output_dir = os.path.join(root_dir, "image_with_label")

    # Visualize the input image and label
    sw = SummaryWriter(log_dir=output_dir)
    plot_2d_or_3d_image(data=data["image"], step=0, writer=sw, frame_dim=-1, tag="image")
    plot_2d_or_3d_image(data=data["label"], step=0, writer=sw, frame_dim=-1, tag="label")

    # Cleanup data directory
    # Remove directory if a temporary was used.
    if directory is None:
        shutil.rmtree(root_dir)
