"""
Test Script for U-Net Image Segmentation Project
================================================

This script tests all major components of the project to ensure
everything is working correctly.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from unet_model import ModernUNet, CombinedLoss, calculate_iou, calculate_dice_score
        print("✅ unet_model imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import unet_model: {e}")
        return False
    
    try:
        from dataset import SyntheticSegmentationDataset, MedicalImageDataset, create_data_loaders
        print("✅ dataset imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import dataset: {e}")
        return False
    
    try:
        from config import Config, load_config
        print("✅ config imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import config: {e}")
        return False
    
    return True


def test_model():
    """Test U-Net model creation and forward pass"""
    print("\n🧪 Testing U-Net model...")
    
    try:
        from unet_model import ModernUNet
        
        # Create model
        model = ModernUNet(
            in_channels=3,
            out_channels=1,
            base_channels=64,
            depth=4,
            dropout_rate=0.1
        )
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 256, 256)
        output = model(dummy_input)
        
        # Check output shape
        expected_shape = (2, 1, 256, 256)
        if output.shape == expected_shape:
            print(f"✅ Model forward pass successful: {output.shape}")
        else:
            print(f"❌ Unexpected output shape: {output.shape}, expected: {expected_shape}")
            return False
        
        # Check output values are in [0, 1] range (sigmoid output)
        if torch.all(output >= 0) and torch.all(output <= 1):
            print("✅ Output values in correct range [0, 1]")
        else:
            print("❌ Output values not in range [0, 1]")
            return False
        
        print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def test_loss_functions():
    """Test loss functions"""
    print("\n🧪 Testing loss functions...")
    
    try:
        from unet_model import CombinedLoss, DiceLoss
        
        # Create dummy data
        pred = torch.rand(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        # Test CombinedLoss
        combined_loss = CombinedLoss()
        loss_value = combined_loss(pred, target)
        
        if loss_value.item() > 0:
            print(f"✅ CombinedLoss working: {loss_value.item():.4f}")
        else:
            print("❌ CombinedLoss returned non-positive value")
            return False
        
        # Test DiceLoss
        dice_loss = DiceLoss()
        dice_value = dice_loss(pred, target)
        
        if 0 <= dice_value.item() <= 1:
            print(f"✅ DiceLoss working: {dice_value.item():.4f}")
        else:
            print("❌ DiceLoss returned value outside [0, 1]")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        return False


def test_metrics():
    """Test evaluation metrics"""
    print("\n🧪 Testing evaluation metrics...")
    
    try:
        from unet_model import calculate_iou, calculate_dice_score
        
        # Create dummy data
        pred = torch.rand(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        # Test IoU
        iou = calculate_iou(pred, target)
        if 0 <= iou <= 1:
            print(f"✅ IoU calculation working: {iou:.4f}")
        else:
            print(f"❌ IoU value outside [0, 1]: {iou}")
            return False
        
        # Test Dice Score
        dice = calculate_dice_score(pred, target)
        if 0 <= dice <= 1:
            print(f"✅ Dice Score calculation working: {dice:.4f}")
        else:
            print(f"❌ Dice Score value outside [0, 1]: {dice}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False


def test_datasets():
    """Test dataset creation and data loading"""
    print("\n🧪 Testing datasets...")
    
    try:
        from dataset import SyntheticSegmentationDataset, MedicalImageDataset, create_data_loaders
        
        # Test Synthetic Dataset
        synthetic_dataset = SyntheticSegmentationDataset(num_samples=10)
        image, mask = synthetic_dataset[0]
        
        if image.shape == (3, 256, 256) and mask.shape == (1, 256, 256):
            print("✅ Synthetic dataset working correctly")
        else:
            print(f"❌ Synthetic dataset shape mismatch: image={image.shape}, mask={mask.shape}")
            return False
        
        # Test Medical Dataset
        medical_dataset = MedicalImageDataset(num_samples=10)
        image, mask = medical_dataset[0]
        
        if image.shape == (3, 256, 256) and mask.shape == (1, 256, 256):
            print("✅ Medical dataset working correctly")
        else:
            print(f"❌ Medical dataset shape mismatch: image={image.shape}, mask={mask.shape}")
            return False
        
        # Test Data Loaders
        train_loader, val_loader = create_data_loaders(
            dataset_type='synthetic',
            batch_size=4,
            num_samples=20
        )
        
        # Test a batch
        for images, masks in train_loader:
            if images.shape[0] <= 4 and images.shape[1:] == (3, 256, 256):
                print("✅ Data loaders working correctly")
                break
        else:
            print("❌ Data loaders not working")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False


def test_config():
    """Test configuration management"""
    print("\n🧪 Testing configuration...")
    
    try:
        from config import Config, load_config
        
        # Test default config
        config = Config()
        batch_size = config.get('training.batch_size')
        
        if batch_size == 8:
            print("✅ Default configuration working")
        else:
            print(f"❌ Default configuration issue: batch_size={batch_size}")
            return False
        
        # Test config modification
        config.set('training.batch_size', 16)
        new_batch_size = config.get('training.batch_size')
        
        if new_batch_size == 16:
            print("✅ Configuration modification working")
        else:
            print(f"❌ Configuration modification failed: {new_batch_size}")
            return False
        
        # Test validation
        if config.validate():
            print("✅ Configuration validation working")
        else:
            print("❌ Configuration validation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_training_setup():
    """Test training setup without actually training"""
    print("\n🧪 Testing training setup...")
    
    try:
        from train import create_config, Trainer
        from unet_model import ModernUNet
        from dataset import create_data_loaders
        
        # Create small config for testing
        config = create_config()
        config['num_samples'] = 20
        config['batch_size'] = 4
        config['num_epochs'] = 1
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            dataset_type=config['dataset_type'],
            batch_size=config['batch_size'],
            num_samples=config['num_samples']
        )
        
        # Create model
        model = ModernUNet(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            base_channels=config['base_channels'],
            depth=config['depth'],
            dropout_rate=config['dropout_rate']
        )
        
        # Create trainer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config
        )
        
        print("✅ Training setup successful")
        return True
        
    except Exception as e:
        print(f"❌ Training setup test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("🚀 Starting U-Net Project Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Model", test_model),
        ("Loss Functions", test_loss_functions),
        ("Metrics", test_metrics),
        ("Datasets", test_datasets),
        ("Configuration", test_config),
        ("Training Setup", test_training_setup)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
