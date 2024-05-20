import torch
import torch.nn as nn


# Define the U-Net model
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
    
# model architecture
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

        # Upsample and potentially resize before concatenation
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
        out = self.conv_last(x)
        return out

def load_checkpoint(checkpoint_path):
    """
    Load an existing checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    return checkpoint

def save_model_only(model, path):
    """
    Save only the model state dictionary to simplify deployment.
    """
    torch.save(model.state_dict(), path)

def main():
    checkpoint_path = 'checkpoint.pth.tar'
    model_path = 'unet_model.pth'

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)

    # Initialize model
    model = UNet()  # Make sure to initialize your model the same way it was done for training
    model.load_state_dict(checkpoint['state_dict'])  # Load state dict from checkpoint

    # Save only the model's state_dict
    save_model_only(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
