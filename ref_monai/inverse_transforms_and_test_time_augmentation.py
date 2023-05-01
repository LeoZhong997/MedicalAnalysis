"""
For the sake of speed, we’ll use a 2D dataset here,
although needless to say the workflow would be identical for 3D data.

We’ll generate the data by taking Decathlon’s 3D brain tumor dataset,
taking the 2D slice containing the most voxels > 0 (the most label),
and then saving the new dataset to disk.

After that, we’ll do normal 2D training with a few augmentations,
which means that we’ll be able to benefit from the inverse transformations.
"""

# Setup imports
from glob import glob
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import random
import tempfile
import torch
from tqdm import tqdm

import monai
from monai.apps import download_and_extract
from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    pad_list_data_collate,
    TestTimeAugmentation,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks import eval_mode
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    CropForegroundd,
    DivisiblePadd,
    KeepLargestConnectedComponent,
    Lambdad,
    LoadImaged,
    MapTransform,
    RandAffined,
    ScaleIntensityd,
    BatchInverseTransform,
)
from monai.transforms.utils import allow_missing_keys_mode
from monai.utils import first, set_determinism


class SliceWithMaxNumLabelsd(MapTransform):
    def __init__(self, keys, label_key):
        self.keys = keys
        self.label_key = label_key

    def __call__(self, data):
        d = dict(data)
        im = d[self.label_key]
        q = np.sum((im.array > 0).reshape(-1, im.array.shape[-1]), axis=0)
        _slice = np.where(q == np.max(q))[0][0]
        for key in self.keys:
            d[key] = d[key][..., _slice]
        return d


class SaveSliced(MapTransform):
    def __init__(self, keys, path):
        self.keys = keys
        self.path = path

    def __call__(self, data):
        d = {}
        for key in self.keys:
            fname = os.path.basename(data[key + "_meta_dict"]["filename_or_obj"])
            path = os.path.join(self.path, key, fname)
            nib.save(nib.Nifti1Image(data[key].array, np.eye(4)), path)
            d[key] = path
        return d


def imshows(ims):
    nrow = len(ims)
    ncol = len(ims[0])
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3), facecolor="white")
    for i, im_dict in enumerate(ims):
        for j, (title, im) in enumerate(im_dict.items()):
            if isinstance(im, torch.Tensor):
                im = im.detach().cpu().numpy()
            im = np.mean(im, axis=0)  # average across channels
            if len(ims) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            ax.set_title(f"{title}\n{im.shape}")
            im_show = ax.imshow(im)
            ax.axis("off")
            fig.colorbar(im_show, ax=ax)


def plot_range(data, wrapped_generator):
    # Get ax, show plot, etc.
    plt.ion()
    for d in data.values():
        ax = d["line"].axes
    fig = ax.get_figure()
    fig.show()

    for i in wrapped_generator:
        yield i
        # update plots, legend, view
        for d in data.values():
            d["line"].set_data(d["x"], d["y"])
        ax.legend()
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()


def infer_seg(images, model, roi_size=(96, 96), sw_batch_size=4):
    val_outputs = sliding_window_inference(images, roi_size, sw_batch_size, model)
    return pad_list_data_collate([post_trans(i) for i in val_outputs])


if __name__ == "__main__":

    monai.config.print_config()

    # Set deterministic training for reproducibility
    set_determinism(seed=0)

    # Setup data directory
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else os.path.expanduser(directory)
    print(root_dir)

    # Create data
    # # Get data, get 2D slice with most voxels > 0 (most label) and save.
    task = "Task01_BrainTumour"
    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/" + task + ".tar"

    compressed_file = os.path.join(root_dir, task + ".tar")
    data_dir = os.path.join(root_dir, task)
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir)

    images = sorted(glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image, "label": label} for image, label in zip(images, labels)]

    keys = ["image", "label"]
    data_dir = os.path.join(root_dir, task + "_single_slice")
    for key in keys:
        os.makedirs(os.path.join(data_dir, key), exist_ok=True)
    transform_2d_slice = Compose(
        [
            LoadImaged(keys),
            EnsureChannelFirstd(keys),
            SliceWithMaxNumLabelsd(keys, "label"),
            SaveSliced(keys, data_dir),
        ]
    )
    # Running the whole way through the dataset will create the 2D slices and save to file
    ds_2d = Dataset(data_dicts, transform_2d_slice)
    dl_2d = DataLoader(ds_2d, batch_size=1, num_workers=10)     # num_workers=10
    data_dicts_single_slice = list(tqdm(dl_2d))

    random.shuffle(data_dicts_single_slice)
    num_files = len(data_dicts_single_slice)
    num_train_files = round(0.8 * num_files)
    train_files = data_dicts_single_slice[:num_train_files]
    val_files = data_dicts_single_slice[num_train_files:]
    print("total num files:", len(data_dicts_single_slice))
    print("num training files:", len(train_files))
    print("num validation files:", len(val_files))

    train_transforms = Compose(
        [
            LoadImaged(keys),
            Lambdad("label", lambda x: (x > 0).to(torch.float)),
            RandAffined(
                keys,
                prob=1.0,
                spatial_size=(300, 300),
                rotate_range=(np.pi / 3, np.pi / 3),
                translate_range=(3, 3),
                scale_range=((0.8, 1), (0.8, 1)),
                padding_mode="zeros",
                mode=("bilinear", "nearest"),
            ),
            CropForegroundd(keys, source_key="image"),
            DivisiblePadd(keys, 16),
            ScaleIntensityd("image"),
        ]
    )
    val_transforms = train_transforms

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=2, num_workers=10, collate_fn=pad_list_data_collate)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=10, collate_fn=pad_list_data_collate)

    # Display some examples
    to_imshow = []
    for file in np.random.choice(train_files, size=5, replace=False):
        data = train_transforms(file)
        to_imshow.append({"image": data["image"], "label": data["label"]})
    imshows(to_imshow)

    # Function for live plotting whilst running training
    post_trans = Compose(
        [
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5),
            KeepLargestConnectedComponent(applied_labels=1),
        ]
    )

    # Create network, loss fn., etc.
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    in_channels = train_ds[0]["image"].shape[0]
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    best_model_path = "best_model_inverse_transforms.pth"
    skip_training_if_poss = True
    skip_training = skip_training_if_poss and os.path.isfile(best_model_path)

    if not skip_training:
        # Plotting stuff
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor="white")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")

        data = {}
        for i in ["train", "val dice"]:
            data[i] = {"x": [], "y": []}
            (data[i]["line"],) = ax.plot(data[i]["x"], data[i]["y"], label=i)

        # start a typical PyTorch training
        max_epochs = 20
        val_interval = 1
        best_metric = -1
        best_metric_epoch = -1

        for epoch in plot_range(data, range(max_epochs)):
            model.train()
            epoch_loss = 0

            for batch_data in train_loader:
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            data["train"]["x"].append(epoch + 1)
            data["train"]["y"].append(epoch_loss)

            if (epoch + 1) % val_interval == 0:
                with eval_mode(model):
                    val_outputs = None
                    for val_data in val_loader:
                        val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                        val_outputs = infer_seg(val_images, model)
                        dice_metric(y_pred=val_outputs, y=val_labels)

                    # aggregate the final mean dice result
                    metric = dice_metric.aggregate().item()
                    # reset the status for next validation round
                    dice_metric.reset()

                    data["val dice"]["x"].append(epoch + 1)
                    data["val dice"]["y"].append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), best_model_path)

        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    model.load_state_dict(torch.load(best_model_path))
    _ = model.eval()

    # Check segmentations
    # Load validation files, apply validation transforms and display (no inverses yet!).
    to_imshow = []
    for file in np.random.choice(val_files, size=5, replace=False):
        data = val_transforms(file)
        inferred = infer_seg(pad_list_data_collate([data["image"].to(device)]), model)[0]
        to_imshow.append(
            {
                "image": data["image"],
                "GT label": data["label"],
                "inferred label": inferred,
            }
        )
    imshows(to_imshow)

    # Inverse transformation
    # Need minimal transforms just to be able to show the unmodified originals
    minimal_transforms = Compose(
        [
            LoadImaged(keys, image_only=True),
            Lambdad("label", lambda x: (x > 0).to(torch.float)),
            ScaleIntensityd("image"),
        ]
    )

    to_imshow = []
    for file in np.random.choice(val_files, size=5, replace=False):
        unmodified_data = minimal_transforms(file)
        transformed_data = val_transforms(file)
        _img = pad_list_data_collate([transformed_data["image"].to(device)])
        seg = infer_seg(_img, model)[0].detach().cpu()
        seg.applied_operations = transformed_data["label"].applied_operations
        seg_dict = {"label": seg}
        with allow_missing_keys_mode(val_transforms):
            inverted_seg = val_transforms.inverse(seg_dict)
        to_imshow.append(
            {
                "orig image": unmodified_data["image"],
                "orig GT label": unmodified_data["label"],
                "trans image": transformed_data["image"],
                "trans GT label": transformed_data["label"],
                "trans inferred label": seg,
                "inverted inferred label": inverted_seg["label"],
            }
        )
    imshows(to_imshow)

    # Batch inverse
    val_loader = DataLoader(val_ds, batch_size=5, num_workers=0, collate_fn=pad_list_data_collate)
    batch_val = first(val_loader)
    batch_infer = torch.sigmoid(sliding_window_inference(batch_val["image"].to(device), (96, 96), 4, model))
    batch_infer.applied_operations = batch_val["label"].applied_operations
    segs_dict = {"label": batch_infer}
    batch_inverter = BatchInverseTransform(val_transforms, val_loader)
    with allow_missing_keys_mode(val_transforms):
        fwd_bck_batch_labels = batch_inverter(segs_dict)

    # visualise it batch_val["label_meta_dict"]["filename_or_obj"]
    to_imshow = []
    for idx, inverted_seg in enumerate(fwd_bck_batch_labels):
        file = {
            "image": batch_val["image_meta_dict"]["filename_or_obj"][idx],
            "label": batch_val["label_meta_dict"]["filename_or_obj"][idx],
        }
        unmodified_data = minimal_transforms(file)
        transformed_data = val_transforms(file)

        to_imshow.append(
            {
                "orig image": unmodified_data["image"],
                "orig GT label": unmodified_data["label"],
                "trans image": transformed_data["image"],
                "trans GT label": transformed_data["label"],
                "inverted inferred label": inverted_seg["label"],
            }
        )
    imshows(to_imshow)

    # Test-time augmentations
    tt_aug = TestTimeAugmentation(
        val_transforms, batch_size=5, num_workers=0, inferrer_fn=lambda x: torch.sigmoid(model(x)), device=device
    )

    # Get images
    to_imshow = []
    for file in np.random.choice(val_files, size=5, replace=False):
        mode_tta, mean_tta, std_tta, vvc_tta = tt_aug(file, num_examples=10)
        unmodified_data = minimal_transforms(file)

        to_imshow.append(
            {
                "orig image": unmodified_data["image"],
                "orig GT label": unmodified_data["label"],
                "mode tta, vvc: %.2f" % vvc_tta: mode_tta,
                "mean tta, vvc: %.2f" % vvc_tta: mean_tta,
                "std tta, vvc: %.2f" % vvc_tta: std_tta,
            }
        )
    imshows(to_imshow)

