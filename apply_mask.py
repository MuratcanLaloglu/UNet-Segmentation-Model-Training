import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
from pathlib import Path
import numpy as np


#  U-Net model
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
    
# U-Net architecture
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


# transformation using the same parameters as during training
transform = transforms.Compose([
    transforms.Resize((720, 401)),
    transforms.ToTensor()
])

# load the trained model
def load_trained_model(model_path, device):
    model = UNet(in_channels=3, out_channels=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    return model

# process an individual image
def process_image(model, device, image_path, output_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)  # Get probabilities

    # Apply soft threshold from training, adjust based on the distribution of model's sigmoid outputs
    # scale of 0-1
    soft_threshold = 0.5  # Adjust this threshold to match performance as in training
    mask_np = (output.squeeze().cpu().numpy() > soft_threshold).astype(np.uint8) * 255

    # Resize mask to original image dimensions
    original_image = cv2.imread(str(image_path))
    mask_resized = cv2.resize(mask_np, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert the single channel binary mask to a 3-channel image for overlay
    mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

    # Overlay the mask onto the original image
    masked_image = cv2.addWeighted(original_image, 1, mask_colored, 0.5, 0)

    # Save the image with mask applied
    cv2.imwrite(str(output_path), masked_image)



def apply_mask_to_directory(model, device, input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_file in input_dir.glob("*.png"):
        output_image_path = output_dir / image_file.name
        process_image(model, device, image_file, output_image_path)

# Main script execution
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "checkpoint.pth.tar"
    
    input_dir = "testImages"  # Directory containing images to mask
    output_dir = "maskedImages"  # Directory to save masked images

    # Load the trained model
    model = load_trained_model(model_path, device)

    # Apply the mask to all images in the directory
    apply_mask_to_directory(model, device, input_dir, output_dir)

if __name__ == "__main__": 
    main()
