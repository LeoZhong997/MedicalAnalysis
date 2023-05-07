"""
Quick Start With Public Datasets and add new Dataset

In this tutorial, we introduce how to quickly set up workflows with MONAI public Datasets and how to add new Dataset.
Currently, MONAI provides MedNISTDataset and DecathlonDataset to automatically download and extract
the MedNIST and Decathlon datasets, and act as PyTorch datasets to generate training/validation/test data.

We'll cover the following topics in this tutorial:
- Create training experiment with MedNISTDataset and workflow
- Create training experiment with DecathlonDataset and workflow
- Share other public data and add Dataset in MONAI
"""
# Setup imports
from monai.transforms import (
    EnsureChannelFirstd,
    AsDiscreted,
    Compose,
    LoadImaged,
    Orientationd,
    Randomizable,
    Resized,
    ScaleIntensityd,
    Spacingd,
    EnsureTyped,
)
from monai.networks.nets import UNet, DenseNet121
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.inferers import SimpleInferer
from monai.handlers import MeanDice, StatsHandler, from_engine
from monai.engines import SupervisedTrainer
from monai.data import CacheDataset, DataLoader
from monai.config import print_config
from monai.apps import DecathlonDataset, MedNISTDataset, download_and_extract
import torch
import matplotlib.pyplot as plt
import ignite
import tempfile
import sys
import shutil
import os
import logging


if __name__ == "__main__":
    print_config()

    # Setup data directory
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(f"root_dir: {root_dir}")

    # Setup logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    task = 'other'  # MedNISTDataset, DecathlonDataset, other

    if task == 'MedNISTDataset':
        # 1. Create training experiment with MedNISTDataset and workflow
        # # Set up pre-processing transforms
        transform = Compose(
            [
                LoadImaged(keys="image"),
                EnsureChannelFirstd(keys="image"),
                ScaleIntensityd(keys="image"),
                EnsureTyped(keys="image"),
            ]
        )

        # # Create MedNISTDataset for training,
        # MedNISTDataset inherits from MONAI CacheDataset and provides rich parameters to achieve expected behavior
        train_ds = MedNISTDataset(root_dir=root_dir, transform=transform, section="training", download=True)
        # the dataset can work seamlessly with the pytorch native dataset loader,
        # but using monai.data.DataLoader has additional benefits of mutli-process
        # random seeds handling, and the customized collate functions
        train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=10)

        # # Pick images from MedNISTDataset to visualize and check
        # plt.subplots(3, 3, figsize=(8, 8))
        # for i in range(9):
        #     plt.subplot(3, 3, i + 1)
        #     plt.imshow(train_ds[i * 5000]["image"][0].detach().cpu(), cmap="gray")
        # plt.tight_layout()
        # plt.show()

        # # Create training components
        # # images classification: AbdomenCT, BreastMRI, ChestCT, CXR, Hand, HeadCT
        device = torch.device("cuda:0")
        net = DenseNet121(spatial_dims=2, in_channels=1, out_channels=6).to(device)
        loss = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(net.parameters(), 1e-5)

        # # Define the easiest training workflow and run
        # # Use MONAI SupervisedTrainer handlers to quickly set up a training workflow
        trainer = SupervisedTrainer(
            device=device,
            max_epochs=5,
            train_data_loader=train_loader,
            network=net,
            optimizer=opt,
            loss_function=loss,
            inferer=SimpleInferer(),
            key_train_metric={"train_acc": ignite.metrics.Accuracy(output_transform=from_engine(["pred", "label"]))},
            train_handlers=StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        )
        trainer.run()

    elif task == 'DecathlonDataset':
        # 2. Create training experiment with DecathlonDataset and workflow
        # # Set up pre-processing transforms
        transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                ScaleIntensityd(keys="image"),
                Resized(keys=["image", "label"], spatial_size=(32, 64, 32), mode=("trilinear", "nearest")),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

        # # Create DeccathlonDataset for training
        train_ds = DecathlonDataset(
            root_dir=root_dir,
            task="Task04_Hippocampus",
            transform=transform,
            section="training",
            download=True,
        )
        # the dataset can work seamlessly with the pytorch native dataset loader,
        # but using monai.data.DataLoader has additional benefits of mutli-process
        # random seeds handling, and the customized collate functions
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=16)

        # # Pick images from DecathlonDataset to visualize and check
        # plt.subplots(3, 3, figsize=(8, 8))
        # for i in range(9):
        #     plt.subplot(3, 3, i + 1)
        #     plt.imshow(train_ds[i * 20]["image"][0, :, :, 10].detach().cpu(), cmap="gray")
        # plt.tight_layout()
        # plt.show()

        # # Create training components
        device = torch.device("cuda:0")
        net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)
        loss = DiceLoss(to_onehot_y=True, softmax=True)
        opt = torch.optim.Adam(net.parameters(), 1e-2)

        # # Define the easiest training workflow and run
        trainer = SupervisedTrainer(
            device=device,
            max_epochs=5,
            train_data_loader=train_loader,
            network=net,
            optimizer=opt,
            loss_function=loss,
            inferer=SimpleInferer(),
            postprocessing=AsDiscreted(
                keys=["pred", "label"],
                argmax=(True, False),
                to_onehot=3,
            ),
            key_train_metric={"train_meandice": MeanDice(output_transform=from_engine(["pred", "label"]))},
            train_handlers=StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        )
        trainer.run()

    else:
        # 3. Share other public data and add Dataset in MONAI
        # # Define IXIDataset as an example
        class IXIDataset(Randomizable, CacheDataset):
            resource = "http://biomedic.doc.ic.ac.uk/" + "brain-development/downloads/IXI/IXI-T1.tar"
            md5 = "34901a0593b41dd19c1a1f746eac2d58"

            def __init__(
                    self,
                    root_dir,
                    section,
                    transform,
                    download=False,
                    seed=0,
                    val_frac=0.2,
                    test_frac=0.2,
                    cache_num=sys.maxsize,
                    cache_rate=1.0,
                    num_workers=0,
            ):
                if not os.path.isdir(root_dir):
                    raise ValueError("Root directory root_dir must be a directory.")
                self.section = section
                self.val_frac = val_frac
                self.test_frac = test_frac
                self.set_random_state(seed=seed)
                dataset_dir = os.path.join(root_dir, "ixi")
                tarfile_name = f"{dataset_dir}.tar"
                if download:
                    download_and_extract(self.resource, tarfile_name, dataset_dir, self.md5)
                # as a quick demo, we just use 10 images to show

                self.datalist = [
                    {"image": os.path.join(dataset_dir, "IXI314-IOP-0889-T1.nii.gz"), "label": 0},
                    {"image": os.path.join(dataset_dir, "IXI249-Guys-1072-T1.nii.gz"), "label": 0},
                    {"image": os.path.join(dataset_dir, "IXI609-HH-2600-T1.nii.gz"), "label": 0},
                    {"image": os.path.join(dataset_dir, "IXI173-HH-1590-T1.nii.gz"), "label": 1},
                    {"image": os.path.join(dataset_dir, "IXI020-Guys-0700-T1.nii.gz"), "label": 0},
                    {"image": os.path.join(dataset_dir, "IXI342-Guys-0909-T1.nii.gz"), "label": 0},
                    {"image": os.path.join(dataset_dir, "IXI134-Guys-0780-T1.nii.gz"), "label": 0},
                    {"image": os.path.join(dataset_dir, "IXI577-HH-2661-T1.nii.gz"), "label": 1},
                    {"image": os.path.join(dataset_dir, "IXI066-Guys-0731-T1.nii.gz"), "label": 1},
                    {"image": os.path.join(dataset_dir, "IXI130-HH-1528-T1.nii.gz"), "label": 0},
                ]
                data = self._generate_data_list()
                super().__init__(
                    data,
                    transform,
                    cache_num=cache_num,
                    cache_rate=cache_rate,
                    num_workers=num_workers,
                )

            def randomize(self, data=None):
                self.rann = self.R.random()

            def _generate_data_list(self):
                data = []
                for d in self.datalist:
                    self.randomize()
                    if self.section == "training":
                        if self.rann < self.val_frac + self.test_frac:
                            continue
                    elif self.section == "validation":
                        if self.rann >= self.val_frac:
                            continue
                    elif self.section == "test":
                        if self.rann < self.val_frac or self.rann >= self.val_frac + self.test_frac:
                            continue
                    else:
                        raise ValueError(
                            f"Unsupported section: {self.section}, "
                            f"available options are ['training', 'validation', 'test']."
                        )
                    data.append(d)
                return data

        # # Pick images from IXIDataset to visualize and check
        train_ds = IXIDataset(
            root_dir=root_dir,
            section="training",
            transform=Compose([LoadImaged("image"), EnsureTyped("image")]),
            download=True,
        )
        plt.figure("check", (18, 6))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(train_ds[i]["image"][:, :, 80].detach().cpu(), cmap="gray")
        plt.show()

    if directory is None:
        shutil.rmtree(root_dir)






