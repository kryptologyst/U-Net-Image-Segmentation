"""
Configuration Management for U-Net Project
==========================================

This module provides configuration management with YAML support,
environment variables, and validation.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import json


class Config:
    """Configuration management class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        self.config_path = config_path
        
        if config_path:
            self.load_from_file(config_path)
        else:
            self.load_defaults()
    
    def load_defaults(self):
        """Load default configuration"""
        self.config = {
            # Model configuration
            'model': {
                'in_channels': 3,
                'out_channels': 1,
                'base_channels': 64,
                'depth': 4,
                'dropout_rate': 0.1,
                'use_attention': True
            },
            
            # Training configuration
            'training': {
                'batch_size': 8,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'num_epochs': 50,
                'early_stopping_patience': 20,
                'scheduler_patience': 10
            },
            
            # Loss configuration
            'loss': {
                'bce_weight': 0.5,
                'dice_weight': 0.5,
                'smooth': 1.0
            },
            
            # Dataset configuration
            'dataset': {
                'type': 'synthetic',
                'num_samples': 1000,
                'image_size': [256, 256],
                'train_split': 0.8,
                'num_shapes_per_image': 3,
                'num_cells_per_image': 5
            },
            
            # Data augmentation
            'augmentation': {
                'enabled': True,
                'horizontal_flip': 0.5,
                'vertical_flip': 0.5,
                'rotation': 15,
                'brightness': 0.2,
                'contrast': 0.2
            },
            
            # Output configuration
            'output': {
                'dir': 'outputs',
                'save_interval': 5,
                'log_interval': 10,
                'visualize_interval': 20
            },
            
            # Logging configuration
            'logging': {
                'use_wandb': False,
                'project_name': 'unet-segmentation',
                'log_level': 'INFO'
            },
            
            # Hardware configuration
            'hardware': {
                'device': 'auto',  # auto, cpu, cuda
                'num_workers': 2,
                'pin_memory': True
            },
            
            # Evaluation configuration
            'evaluation': {
                'metrics': ['iou', 'dice', 'accuracy', 'precision', 'recall'],
                'threshold': 0.5,
                'save_predictions': True
            }
        }
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        
        # Load defaults first, then override with file config
        self.load_defaults()
        self._update_config(self.config, file_config)
    
    def _update_config(self, base_config: Dict, update_config: Dict):
        """Recursively update configuration"""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update_from_env(self):
        """Update configuration from environment variables"""
        env_mappings = {
            'UNET_BATCH_SIZE': 'training.batch_size',
            'UNET_LEARNING_RATE': 'training.learning_rate',
            'UNET_NUM_EPOCHS': 'training.num_epochs',
            'UNET_DEVICE': 'hardware.device',
            'UNET_OUTPUT_DIR': 'output.dir',
            'UNET_USE_WANDB': 'logging.use_wandb',
            'UNET_PROJECT_NAME': 'logging.project_name'
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if config_key in ['training.batch_size', 'training.num_epochs']:
                    self.set(config_key, int(env_value))
                elif config_key in ['training.learning_rate']:
                    self.set(config_key, float(env_value))
                elif config_key in ['logging.use_wandb']:
                    self.set(config_key, env_value.lower() in ['true', '1', 'yes'])
                else:
                    self.set(config_key, env_value)
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Validate model configuration
        if self.get('model.base_channels') <= 0:
            errors.append("Model base_channels must be positive")
        
        if self.get('model.depth') <= 0:
            errors.append("Model depth must be positive")
        
        if not 0 <= self.get('model.dropout_rate') <= 1:
            errors.append("Model dropout_rate must be between 0 and 1")
        
        # Validate training configuration
        if self.get('training.batch_size') <= 0:
            errors.append("Training batch_size must be positive")
        
        if self.get('training.learning_rate') <= 0:
            errors.append("Training learning_rate must be positive")
        
        if self.get('training.num_epochs') <= 0:
            errors.append("Training num_epochs must be positive")
        
        # Validate dataset configuration
        if self.get('dataset.num_samples') <= 0:
            errors.append("Dataset num_samples must be positive")
        
        if not 0 < self.get('dataset.train_split') < 1:
            errors.append("Dataset train_split must be between 0 and 1")
        
        # Validate loss configuration
        if not 0 <= self.get('loss.bce_weight') <= 1:
            errors.append("Loss bce_weight must be between 0 and 1")
        
        if not 0 <= self.get('loss.dice_weight') <= 1:
            errors.append("Loss dice_weight must be between 0 and 1")
        
        if abs(self.get('loss.bce_weight') + self.get('loss.dice_weight') - 1.0) > 1e-6:
            errors.append("Loss weights must sum to 1.0")
        
        if errors:
            print("❌ Configuration validation errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        print("✅ Configuration validation passed")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.config.copy()
    
    def print_config(self):
        """Print configuration in a readable format"""
        print("📋 Configuration:")
        print(json.dumps(self.config, indent=2))


def create_config_file(config_path: str = "config.yaml"):
    """Create a sample configuration file"""
    config = Config()
    config.save_to_file(config_path)
    print(f"✅ Sample configuration file created: {config_path}")


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or create default"""
    if config_path and Path(config_path).exists():
        config = Config(config_path)
        print(f"✅ Configuration loaded from: {config_path}")
    else:
        config = Config()
        print("✅ Using default configuration")
    
    # Update from environment variables
    config.update_from_env()
    
    # Validate configuration
    config.validate()
    
    return config


if __name__ == "__main__":
    # Test configuration management
    print("🧪 Testing Configuration Management")
    
    # Create default config
    config = Config()
    print("\n📋 Default Configuration:")
    config.print_config()
    
    # Test get/set methods
    print(f"\n🔍 Testing get/set methods:")
    print(f"   Model base_channels: {config.get('model.base_channels')}")
    config.set('model.base_channels', 128)
    print(f"   Updated base_channels: {config.get('model.base_channels')}")
    
    # Test environment variable updates
    print(f"\n🌍 Testing environment variable updates:")
    os.environ['UNET_BATCH_SIZE'] = '16'
    os.environ['UNET_LEARNING_RATE'] = '0.001'
    config.update_from_env()
    print(f"   Batch size from env: {config.get('training.batch_size')}")
    print(f"   Learning rate from env: {config.get('training.learning_rate')}")
    
    # Create sample config file
    create_config_file("sample_config.yaml")
    
    # Test loading from file
    loaded_config = Config("sample_config.yaml")
    print(f"\n📁 Configuration loaded from file:")
    print(f"   Model depth: {loaded_config.get('model.depth')}")
    
    print("\n🎉 Configuration management test completed!")
