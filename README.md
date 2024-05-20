
# UNet Segmentation Model Training

This repository contains the code for training a UNet segmentation model using PyTorch. The model is designed to segment images with a focus on identifying specific features such as teeth and glasses. The training process includes data augmentation, custom loss functions, and model checkpointing.
I have also included py files on how to make the validation dataset: make_validationDataset.py, save_model.py to save the model, and code to run the model, and apply the predicted masks onto images: apply_mask.py

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset Preparation](#dataset-preparation)
- [Training and Validation](#training-and-validation)
- [Plotting Predictions](#plotting-predictions)
- [Checkpoints](#checkpoints)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Features

- Custom UNet architecture for image segmentation.
- Data augmentation techniques to enhance model generalization.
- Combined binary cross-entropy and dice loss for improved training.
- Training and validation data loaders with on-the-fly transformations.
- Model checkpointing to save the best performing model.
- Visualization of predictions during training.

## Installation

Ensure you have PyTorch installed. You can install it from the official [PyTorch website](https://pytorch.org/).

## Usage

### Training the Model

To train the model, run `train.py`:

```bash
python train.py
```

This will start the training process, which includes loading the dataset, performing data augmentation, training the UNet model, and saving the best model checkpoints.
Make sure to add the files for training:

.data/mask  <- for the masked images
.data/truth <- for the original images that are not masked.

Make sure these images are 1 to 1. Meaning that every image is named the same in both folders, and that they are an exact match. Failure to do this will result in a more or less useless model in the end.

### Checkpoints

The training process includes saving the model checkpoints with the best validation loss. The checkpoints are saved in the file `checkpoint.pth.tar`.

## Model Architecture

The model architecture is based on the UNet structure, which is widely used for image segmentation tasks. Here is an overview of the architecture:

```python

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.dconv_down1 = DoubleConv(in_channels, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(64 + 128, 64)
        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = torch.sigmoid(self.conv_last(x))
        return out
```

## Dataset Preparation

The dataset should be split into training and validation sets and stored in a pickle file. The dataset includes paths to the images and their corresponding mask images.

### Custom Dataset Class

The \`CustomDataset\` class handles data loading and on-the-fly transformations including horizontal and vertical flips, rotations, brightness, contrast, saturation, and hue adjustments. It also includes a mechanism for generating grayscale masks based on specific color ranges.

## Training and Validation

The training process includes loading the dataset, applying transformations, and using a combined loss function (binary cross-entropy and dice loss) to optimize the model.

### Loss Function

```python
class CombinedLoss(nn.Module):
    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        smooth = 1
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return bce_loss + dice_loss
```

### Training Loop

The `train_and_validate` function handles the training and validation process, saving the best model based on validation loss.

## Plotting Predictions

The `plot_and_save_predictions` function generates and saves plots of the model's predictions alongside the true masks and input images. This helps in visualizing the model's performance during training.

## Results

The best model checkpoint and training logs can be found in the specified directory. The validation loss and training loss are printed during the training process.

## Acknowledgments

This project is based on the UNet architecture for image segmentation. Special thanks to the PyTorch community for providing excellent resources and tutorials.
