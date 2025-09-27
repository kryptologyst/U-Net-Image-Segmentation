"""
Mock Dataset for Image Segmentation
==================================

This module creates synthetic datasets for testing U-Net segmentation models.
It generates images with geometric shapes and their corresponding masks.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from typing import Tuple, List, Optional
import random
import os


class SyntheticSegmentationDataset(Dataset):
    """
    Synthetic dataset that generates images with geometric shapes and their masks
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: Tuple[int, int] = (256, 256),
        num_shapes_per_image: int = 3,
        shape_types: List[str] = None,
        background_color: Tuple[int, int, int] = (50, 50, 50),
        foreground_colors: List[Tuple[int, int, int]] = None
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_shapes_per_image = num_shapes_per_image
        
        if shape_types is None:
            self.shape_types = ['circle', 'rectangle', 'triangle', 'ellipse']
        else:
            self.shape_types = shape_types
            
        self.background_color = background_color
        
        if foreground_colors is None:
            self.foreground_colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
            ]
        else:
            self.foreground_colors = foreground_colors
    
    def _create_circle(self, image: Image.Image, mask: Image.Image, 
                      center: Tuple[int, int], radius: int, color: Tuple[int, int, int]):
        """Create a circle shape"""
        draw_img = ImageDraw.Draw(image)
        draw_mask = ImageDraw.Draw(mask)
        
        # Draw circle
        bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
        draw_img.ellipse(bbox, fill=color)
        draw_mask.ellipse(bbox, fill=255)
    
    def _create_rectangle(self, image: Image.Image, mask: Image.Image,
                         top_left: Tuple[int, int], bottom_right: Tuple[int, int], 
                         color: Tuple[int, int, int]):
        """Create a rectangle shape"""
        draw_img = ImageDraw.Draw(image)
        draw_mask = ImageDraw.Draw(mask)
        
        bbox = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
        draw_img.rectangle(bbox, fill=color)
        draw_mask.rectangle(bbox, fill=255)
    
    def _create_triangle(self, image: Image.Image, mask: Image.Image,
                        points: List[Tuple[int, int]], color: Tuple[int, int, int]):
        """Create a triangle shape"""
        draw_img = ImageDraw.Draw(image)
        draw_mask = ImageDraw.Draw(mask)
        
        draw_img.polygon(points, fill=color)
        draw_mask.polygon(points, fill=255)
    
    def _create_ellipse(self, image: Image.Image, mask: Image.Image,
                       center: Tuple[int, int], axes: Tuple[int, int], 
                       color: Tuple[int, int, int]):
        """Create an ellipse shape"""
        draw_img = ImageDraw.Draw(image)
        draw_mask = ImageDraw.Draw(mask)
        
        bbox = [center[0] - axes[0], center[1] - axes[1], 
                center[0] + axes[0], center[1] + axes[1]]
        draw_img.ellipse(bbox, fill=color)
        draw_mask.ellipse(bbox, fill=255)
    
    def _generate_single_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single image and its corresponding mask"""
        # Create blank images
        image = Image.new('RGB', self.image_size, self.background_color)
        mask = Image.new('L', self.image_size, 0)
        
        # Add random shapes
        for _ in range(self.num_shapes_per_image):
            shape_type = random.choice(self.shape_types)
            color = random.choice(self.foreground_colors)
            
            if shape_type == 'circle':
                center = (random.randint(50, self.image_size[0] - 50),
                         random.randint(50, self.image_size[1] - 50))
                radius = random.randint(20, 60)
                self._create_circle(image, mask, center, radius, color)
            
            elif shape_type == 'rectangle':
                top_left = (random.randint(20, self.image_size[0] - 100),
                           random.randint(20, self.image_size[1] - 100))
                bottom_right = (top_left[0] + random.randint(40, 80),
                               top_left[1] + random.randint(40, 80))
                self._create_rectangle(image, mask, top_left, bottom_right, color)
            
            elif shape_type == 'triangle':
                center_x = random.randint(50, self.image_size[0] - 50)
                center_y = random.randint(50, self.image_size[1] - 50)
                size = random.randint(30, 60)
                points = [
                    (center_x, center_y - size),
                    (center_x - size, center_y + size),
                    (center_x + size, center_y + size)
                ]
                self._create_triangle(image, mask, points, color)
            
            elif shape_type == 'ellipse':
                center = (random.randint(50, self.image_size[0] - 50),
                         random.randint(50, self.image_size[1] - 50))
                axes = (random.randint(20, 50), random.randint(15, 40))
                self._create_ellipse(image, mask, center, axes, color)
        
        # Convert to numpy arrays
        image_array = np.array(image)
        mask_array = np.array(mask)
        
        return image_array, mask_array
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset"""
        image, mask = self._generate_single_image()
        
        # Convert to tensors and normalize
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        return image_tensor, mask_tensor


class MedicalImageDataset(Dataset):
    """
    Simulated medical image dataset (e.g., cell segmentation)
    """
    
    def __init__(
        self,
        num_samples: int = 500,
        image_size: Tuple[int, int] = (256, 256),
        num_cells_per_image: int = 5
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_cells_per_image = num_cells_per_image
    
    def _create_cell(self, image: np.ndarray, mask: np.ndarray,
                    center: Tuple[int, int], radius: int):
        """Create a cell-like structure"""
        y, x = np.ogrid[:self.image_size[1], :self.image_size[0]]
        
        # Create circular cell
        cell_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        # Add some texture to the cell
        cell_intensity = np.random.uniform(0.3, 0.8)
        image[cell_mask] = cell_intensity
        
        # Add membrane
        membrane_mask = ((x - center[0])**2 + (y - center[1])**2 <= (radius + 2)**2) & \
                       ((x - center[0])**2 + (y - center[1])**2 > (radius - 2)**2)
        image[membrane_mask] = 0.9
        
        # Update mask
        mask[cell_mask] = 1.0
    
    def _generate_single_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single medical image"""
        # Create background with noise
        image = np.random.normal(0.1, 0.05, self.image_size)
        image = np.clip(image, 0, 1)
        mask = np.zeros(self.image_size)
        
        # Add cells
        for _ in range(self.num_cells_per_image):
            center = (random.randint(30, self.image_size[0] - 30),
                     random.randint(30, self.image_size[1] - 30))
            radius = random.randint(15, 35)
            self._create_cell(image, mask, center, radius)
        
        return image, mask
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset"""
        image, mask = self._generate_single_image()
        
        # Convert to RGB (grayscale repeated)
        image_rgb = np.stack([image, image, image], axis=2)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image_tensor, mask_tensor


def create_data_loaders(
    dataset_type: str = "synthetic",
    batch_size: int = 8,
    train_split: float = 0.8,
    num_samples: int = 1000,
    image_size: Tuple[int, int] = (256, 256)
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders
    """
    if dataset_type == "synthetic":
        dataset = SyntheticSegmentationDataset(
            num_samples=num_samples,
            image_size=image_size
        )
    elif dataset_type == "medical":
        dataset = MedicalImageDataset(
            num_samples=num_samples,
            image_size=image_size
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


def visualize_dataset_samples(dataset: Dataset, num_samples: int = 4):
    """Visualize samples from the dataset"""
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        image, mask = dataset[i]
        
        # Convert tensors to numpy for visualization
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().numpy()
        
        # Plot image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"Image {i+1}")
        axes[0, i].axis('off')
        
        # Plot mask
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f"Mask {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test synthetic dataset
    print("Testing Synthetic Dataset...")
    synthetic_dataset = SyntheticSegmentationDataset(num_samples=10)
    print(f"✅ Synthetic dataset created with {len(synthetic_dataset)} samples")
    
    # Test medical dataset
    print("\nTesting Medical Dataset...")
    medical_dataset = MedicalImageDataset(num_samples=10)
    print(f"✅ Medical dataset created with {len(medical_dataset)} samples")
    
    # Test data loaders
    print("\nTesting Data Loaders...")
    train_loader, val_loader = create_data_loaders(
        dataset_type="synthetic",
        batch_size=4,
        num_samples=100
    )
    
    print(f"✅ Train loader: {len(train_loader)} batches")
    print(f"✅ Val loader: {len(val_loader)} batches")
    
    # Test a batch
    for images, masks in train_loader:
        print(f"✅ Batch shape - Images: {images.shape}, Masks: {masks.shape}")
        break
    
    print("\n🎉 Mock datasets ready for training!")
