import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
from pathlib import Path
import random
import torch.nn.functional as F

# Set up a path for saving plots
save_plots = True
plotSavePath = Path("plot_images")
plotSavePath.mkdir(exist_ok=True)

def plot_and_save_predictions(images, true_masks, preds_masks, epoch, val_loss, figsize_per_image=(3, 3)):
    num_images = len(images)
    aspect_ratio = 720 / 401  # Example aspect ratio
    total_fig_width = figsize_per_image[0] * 3 * aspect_ratio
    total_fig_height = figsize_per_image[1] * num_images
    fig, axs = plt.subplots(num_images, 3, figsize=(total_fig_width, total_fig_height))

    if num_images == 1:
        axs = np.array([axs]).reshape(-1, 3)

    for i in range(num_images):
        image = images[i].cpu().detach().numpy().transpose(1, 2, 0)
        true_mask = true_masks[i].cpu().detach().numpy().squeeze()
        pred_mask = preds_masks[i].cpu().detach().numpy().squeeze()  # Now preds_masks should already be sigmoid-applied

        axs[i, 0].imshow(image, aspect='auto')
        axs[i, 0].set_title('Image')

        axs[i, 1].imshow(true_mask, cmap='gray', aspect='auto')
        axs[i, 1].set_title('True Mask')

        axs[i, 2].imshow(pred_mask, cmap='gray', aspect='auto')
        axs[i, 2].set_title(f'Predicted Mask\nEpoch: {epoch}, Val Loss: {val_loss:.6f}')

        for ax in axs[i]:
            ax.axis('off')

    plt.tight_layout()
    fig.suptitle(f'Predictions at Epoch {epoch}', y=1.02)
    fig.canvas.draw()
    plt.savefig(plotSavePath / f'predictions_epoch_{epoch}.png', bbox_inches='tight')
    plt.close(fig)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

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
        if x.size()[2:] != conv3.size()[2:]:
            x = F.interpolate(x, size=conv3.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        if x.size()[2:] != conv2.size()[2:]:
            x = F.interpolate(x, size=conv2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        if x.size()[2:] != conv1.size()[2:]:
            x = F.interpolate(x, size=conv1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        out = torch.sigmoid(self.conv_last(x))
        return out

class CustomDataset(Dataset):
    def __init__(self, image_paths, target_paths, transform=None, target_transform=None, target_size=(720, 401)):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform
        self.target_transform = target_transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.target_paths[idx]).convert("RGB")

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        angle = random.uniform(-30, 30)
        image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)
        mask_array = np.array(mask)  # Convert the rotated mask back to a numpy array


        # Define color tolerance for teeth and glasses
        teeth_tolerance = np.array([20, 20, 20])
        glasses_tolerance = np.array([15, 15, 15])  # Adjust as needed for your images

        brightness_factor = random.uniform(0.95, 1.05)
        contrast_factor = random.uniform(0.95, 1.05)
        saturation_factor = random.uniform(0.95, 1.05)
        hue_factor = random.uniform(-0.02, 0.02)

        image = TF.adjust_brightness(image, brightness_factor)
        image = TF.adjust_contrast(image, contrast_factor)
        image = TF.adjust_saturation(image, saturation_factor)
        image = TF.adjust_hue(image, hue_factor)

        if self.transform:
            image = self.transform(image)

        image = TF.resize(image, self.target_size, interpolation=Image.BILINEAR)


        # function to check if the colors are within the range for teeth
        def is_teeth_color(rgbs):
            return np.logical_or.reduce([
                np.all(np.abs(rgbs - np.array([114, 170, 170])) <= teeth_tolerance, axis=-1),
                np.all(np.abs(rgbs - np.array([178, 255, 255])) <= teeth_tolerance, axis=-1),
                np.all(np.abs(rgbs - np.array([146, 234, 237])) <= teeth_tolerance, axis=-1),
                np.all(np.abs(rgbs - np.array([150, 243, 246])) <= teeth_tolerance, axis=-1),
                np.all(np.abs(rgbs - np.array([182, 255, 255])) <= teeth_tolerance, axis=-1),
                np.all(np.abs(rgbs - np.array([177, 242, 246])) <= teeth_tolerance, axis=-1),
            ])

        # Function to check if the colors are within the range for glasses
        def is_glasses_color(rgbs):
            return np.logical_or.reduce([
                np.all(np.abs(rgbs - np.array([248, 255, 255])) <= glasses_tolerance, axis=-1),
                np.all(np.abs(rgbs - np.array([251, 255, 255])) <= glasses_tolerance, axis=-1),
                np.all(np.abs(rgbs - np.array([255, 251, 255])) <= glasses_tolerance, axis=-1),
                np.all(np.abs(rgbs - np.array([251, 244, 254])) <= glasses_tolerance, axis=-1),
                np.all(np.abs(rgbs - np.array([254, 252, 253])) <= glasses_tolerance, axis=-1),
                np.all(np.abs(rgbs - np.array([243, 254, 253])) <= glasses_tolerance, axis=-1),
                np.all(np.abs(rgbs - np.array([224, 248, 252])) <= glasses_tolerance, axis=-1),
                np.all(np.abs(rgbs - np.array([235, 255, 255])) <= glasses_tolerance, axis=-1),
                np.all(np.abs(rgbs - np.array([251, 251, 255])) <= glasses_tolerance, axis=-1),
                np.all(np.abs(rgbs - np.array([157, 186, 188])) <= glasses_tolerance, axis=-1),
                np.all(np.abs(rgbs - np.array([196, 213, 215])) <= glasses_tolerance, axis=-1),
            ])
 

        # Create masks based on the above functions
        teeth_mask = is_teeth_color(mask_array)
        glasses_mask = is_glasses_color(mask_array)

        # Initialize the grayscale mask
        grayscale_mask = np.zeros(mask_array.shape[:2], dtype=np.float32)
        
        # Assign the grayscale intensity values for teeth and glasses
        grayscale_mask[teeth_mask] = 1.0  # Teeth
        grayscale_mask[glasses_mask] = 1.0  # Glasses

        # Ensure the rest of the mask is zero
        grayscale_mask[~(teeth_mask | glasses_mask)] = 0.0

        # Convert the grayscale mask to a tensor
        mask_tensor = torch.from_numpy(grayscale_mask).float()
        mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
        mask_tensor = TF.resize(mask_tensor, self.target_size, interpolation=Image.NEAREST)

        return image, mask_tensor
    
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        smooth = 1
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return bce_loss + dice_loss

def prepare_loaders(pickle_file, transform, batch_size, target_size):
    with open(pickle_file, 'rb') as f:
        split_data = pickle.load(f)
    train_ds = CustomDataset(split_data['train_img_paths'], split_data['train_mask_paths'], transform=transform, target_size=target_size)
    val_ds = CustomDataset(split_data['val_img_paths'], split_data['val_mask_paths'], transform=transform, target_size=target_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    start_epoch = 1
    best_loss = float('inf')
    if os.path.isfile(checkpoint_path):
        print(f"=> Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resumed training from epoch {start_epoch} with best validation loss {best_loss}")
    else:
        print("=> No checkpoint found, starting from scratch.")
    return start_epoch, best_loss

def train_and_validate(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, device, checkpoint_path):
    best_loss = np.inf
    start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Training"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.float()
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Validation"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                outputs = outputs.float()
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            print(f"Validation Loss Decreased({best_loss:.6f}--->{val_loss:.6f}) \nSaving The Model")
            best_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
            }
            save_checkpoint(checkpoint, filename="checkpoint.pth.tar")

            if images.size(0) >= 2:
                images_sub = images[:3]  # Select the first three images for visualization
                masks_sub = masks[:3]  # Corresponding true masks
                preds_sub = torch.sigmoid(outputs)[:3]  # Apply sigmoid to scale outputs to [0, 1] range without thresholding
                plot_and_save_predictions(images_sub, masks_sub, preds_sub, epoch, val_loss)

    return best_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    loss_fn = CombinedLoss()
    target_size = (720, 401)
    
    transform = transforms.Compose([transforms.ToTensor()])
    batch_size = 2
    train_loader, val_loader = prepare_loaders('train_val_split.pkl', transform, batch_size, target_size=(720, 401))

    num_epochs = 40000
    checkpoint_path = "checkpoint.pth.tar"
    train_and_validate(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, device, checkpoint_path)

if __name__ == '__main__':
    main()
