"""
Training Pipeline for U-Net Image Segmentation
=============================================

This module provides a complete training pipeline with:
- Model training and validation
- Loss tracking and metrics
- Model checkpointing
- Learning rate scheduling
- Early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import wandb
from pathlib import Path

from unet_model import ModernUNet, CombinedLoss, calculate_iou, calculate_dice_score
from dataset import create_data_loaders


class Trainer:
    """Main trainer class for U-Net model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Initialize loss function
        self.criterion = CombinedLoss(
            bce_weight=config.get('bce_weight', 0.5),
            dice_weight=config.get('dice_weight', 0.5)
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('scheduler_patience', 10)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_ious = []
        self.val_ious = []
        self.train_dices = []
        self.val_dices = []
        
        # Create output directory
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('project_name', 'unet-segmentation'),
                config=config,
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                iou = calculate_iou(outputs, masks)
                dice = calculate_dice_score(outputs, masks)
            
            total_loss += loss.item()
            total_iou += iou
            total_dice += dice
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{iou:.4f}',
                'Dice': f'{dice:.4f}'
            })
        
        return {
            'loss': total_loss / num_batches,
            'iou': total_iou / num_batches,
            'dice': total_dice / num_batches
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Calculate metrics
                iou = calculate_iou(outputs, masks)
                dice = calculate_dice_score(outputs, masks)
                
                total_loss += loss.item()
                total_iou += iou
                total_dice += dice
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'IoU': f'{iou:.4f}',
                    'Dice': f'{dice:.4f}'
                })
        
        return {
            'loss': total_loss / num_batches,
            'iou': total_iou / num_batches,
            'dice': total_dice / num_batches
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_ious': self.train_ious,
            'val_ious': self.val_ious,
            'train_dices': self.train_dices,
            'val_dices': self.val_dices,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"✅ New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_ious = checkpoint['train_ious']
        self.val_ious = checkpoint['val_ious']
        self.train_dices = checkpoint['train_dices']
        self.val_dices = checkpoint['val_dices']
        
        print(f"✅ Checkpoint loaded from epoch {self.current_epoch}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # IoU plot
        axes[0, 1].plot(self.train_ious, label='Train IoU')
        axes[0, 1].plot(self.val_ious, label='Val IoU')
        axes[0, 1].set_title('Training and Validation IoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Dice plot
        axes[1, 0].plot(self.train_dices, label='Train Dice')
        axes[1, 0].plot(self.val_dices, label='Val Dice')
        axes[1, 0].set_title('Training and Validation Dice Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        axes[1, 1].plot(lrs)
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train(self, num_epochs: int, early_stopping_patience: int = 20):
        """Main training loop"""
        print(f"🚀 Starting training for {num_epochs} epochs...")
        print(f"📊 Training on {len(self.train_loader)} batches")
        print(f"📊 Validation on {len(self.val_loader)} batches")
        
        early_stopping_counter = 0
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Store metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.train_ious.append(train_metrics['iou'])
            self.val_ious.append(val_metrics['iou'])
            self.train_dices.append(train_metrics['dice'])
            self.val_dices.append(val_metrics['dice'])
            
            # Check for best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Log metrics
            print(f"\n📈 Epoch {epoch+1}/{num_epochs}")
            print(f"   Train Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, Dice: {train_metrics['dice']:.4f}")
            print(f"   Val Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}")
            print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_iou': train_metrics['iou'],
                    'val_iou': val_metrics['iou'],
                    'train_dice': train_metrics['dice'],
                    'val_dice': val_metrics['dice'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break
        
        print("✅ Training completed!")
        self.plot_training_history()
        
        # Save final metrics
        metrics = {
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_train_iou': self.train_ious[-1],
            'final_val_iou': self.val_ious[-1],
            'final_train_dice': self.train_dices[-1],
            'final_val_dice': self.val_dices[-1],
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.train_losses)
        }
        
        with open(self.output_dir / 'final_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics


def create_config() -> Dict:
    """Create default training configuration"""
    return {
        # Model parameters
        'in_channels': 3,
        'out_channels': 1,
        'base_channels': 64,
        'depth': 4,
        'dropout_rate': 0.1,
        
        # Training parameters
        'batch_size': 8,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 50,
        'early_stopping_patience': 20,
        
        # Loss parameters
        'bce_weight': 0.5,
        'dice_weight': 0.5,
        
        # Scheduler parameters
        'scheduler_patience': 10,
        
        # Dataset parameters
        'dataset_type': 'synthetic',
        'num_samples': 1000,
        'image_size': (256, 256),
        'train_split': 0.8,
        
        # Output parameters
        'output_dir': 'outputs',
        'project_name': 'unet-segmentation',
        'use_wandb': False
    }


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Create configuration
    config = create_config()
    print("⚙️  Configuration created")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset_type=config['dataset_type'],
        batch_size=config['batch_size'],
        train_split=config['train_split'],
        num_samples=config['num_samples'],
        image_size=config['image_size']
    )
    print("📊 Data loaders created")
    
    # Create model
    model = ModernUNet(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        base_channels=config['base_channels'],
        depth=config['depth'],
        dropout_rate=config['dropout_rate']
    )
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    print("🏋️  Trainer initialized")
    
    # Start training
    final_metrics = trainer.train(
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    print("\n🎉 Training completed successfully!")
    print(f"📊 Final metrics: {final_metrics}")
