"""
Working U-Net Implementation for Image Segmentation
==================================================

This module implements a simple but working U-Net architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and dropout"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout_rate)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        return x


class ModernUNet(nn.Module):
    """
    Working U-Net implementation
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.1
    ):
        super(ModernUNet, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels, dropout_rate)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, dropout_rate)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, dropout_rate)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, dropout_rate)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16, dropout_rate)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8, dropout_rate)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4, dropout_rate)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2, dropout_rate)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels, dropout_rate)
        
        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        # Final output
        x = self.final_conv(d1)
        return torch.sigmoid(x)


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks"""
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE and Dice Loss"""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate Intersection over Union (IoU) metric"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()


def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate Dice Score metric"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    dice = (2. * intersection) / (pred_binary.sum() + target_binary.sum())
    
    return dice.item() if dice.item() == dice.item() else 0.0  # Handle NaN


if __name__ == "__main__":
    # Test the working U-Net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = ModernUNet(
        in_channels=3,
        out_channels=1,
        base_channels=64,
        depth=4,
        dropout_rate=0.1
    ).to(device)
    
    # Test with dummy data
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    dummy_target = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ Input shape: {dummy_input.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test loss functions
    loss_fn = CombinedLoss()
    loss = loss_fn(output, dummy_target)
    print(f"✅ Combined loss: {loss.item():.4f}")
    
    # Test metrics
    iou = calculate_iou(output, dummy_target)
    dice = calculate_dice_score(output, dummy_target)
    print(f"✅ IoU: {iou:.4f}")
    print(f"✅ Dice Score: {dice:.4f}")
    
    print("\n🎉 Working U-Net implementation ready!")