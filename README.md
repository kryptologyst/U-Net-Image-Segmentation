# U-Net Image Segmentation

A modern implementation of U-Net for image segmentation with PyTorch, featuring attention mechanisms, comprehensive training pipeline, and interactive web interface.

## Features

- **Modern U-Net Architecture**: Enhanced with batch normalization, dropout, and attention gates
- **Comprehensive Training Pipeline**: Complete training with validation, checkpointing, and metrics tracking
- **Mock Datasets**: Synthetic and medical image datasets for testing and demonstration
- **Interactive Web UI**: Streamlit-based interface for image upload and real-time segmentation
- **Configuration Management**: YAML-based configuration with environment variable support
- **Advanced Loss Functions**: Combined BCE and Dice loss for better segmentation performance
- **Evaluation Metrics**: IoU, Dice Score, and other segmentation metrics
- **Model Checkpointing**: Automatic saving of best models and training history

##  Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/U-Net-Image-Segmentation.git
cd U-Net-Image-Segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

#### 1. Train a Model

```python
from train import create_config, Trainer
from unet_model import ModernUNet
from dataset import create_data_loaders
import torch

# Create configuration
config = create_config()

# Create data loaders
train_loader, val_loader = create_data_loaders(
    dataset_type='synthetic',
    batch_size=8,
    num_samples=1000
)

# Create model
model = ModernUNet(
    in_channels=3,
    out_channels=1,
    base_channels=64,
    depth=4,
    use_attention=True
)

# Create trainer and start training
trainer = Trainer(model, train_loader, val_loader, torch.device('cuda'), config)
trainer.train(num_epochs=50)
```

#### 2. Run Web Interface

```bash
streamlit run app.py
```

#### 3. Use Configuration Management

```python
from config import Config

# Load configuration
config = Config('config.yaml')

# Access configuration values
batch_size = config.get('training.batch_size')
learning_rate = config.get('training.learning_rate')

# Update configuration
config.set('model.base_channels', 128)
config.save_to_file('updated_config.yaml')
```

## 📁 Project Structure

```
unet-image-segmentation/
├── 0131.py                 # Original simple implementation
├── unet_model.py          # Modern U-Net architecture
├── dataset.py             # Mock datasets and data loaders
├── train.py               # Training pipeline
├── app.py                 # Streamlit web interface
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── config.yaml            # Configuration file
├── README.md              # This file
└── outputs/               # Training outputs and checkpoints
    ├── checkpoint_best.pth
    ├── checkpoint_latest.pth
    ├── training_history.png
    └── final_metrics.json
```

## Model Architecture

The modern U-Net implementation includes:

- **Encoder-Decoder Structure**: Contracting and expanding paths
- **Skip Connections**: Direct connections between encoder and decoder
- **Attention Gates**: Focus on relevant features during upsampling
- **Batch Normalization**: Stabilize training and improve convergence
- **Dropout**: Prevent overfitting
- **Modern Activation Functions**: ReLU with proper initialization

### Architecture Details

```
Input (3, 256, 256)
    ↓
Encoder Blocks (64 → 128 → 256 → 512 channels)
    ↓
Bottleneck (512 channels)
    ↓
Decoder Blocks (512 → 256 → 128 → 64 channels)
    ↓
Output (1, 256, 256) - Binary segmentation mask
```

## Datasets

### Synthetic Dataset
- Geometric shapes (circles, rectangles, triangles, ellipses)
- Configurable number of shapes per image
- Random colors and positions
- Perfect ground truth masks

### Medical Dataset
- Simulated cell-like structures
- Realistic medical image appearance
- Variable cell sizes and positions
- Noise and texture simulation

## Training Features

- **Combined Loss Function**: BCE + Dice loss for better segmentation
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Early Stopping**: Prevent overfitting with patience-based stopping
- **Model Checkpointing**: Save best and latest models
- **Metrics Tracking**: IoU, Dice Score, and loss monitoring
- **Visualization**: Training history plots and sample predictions

## Web Interface

The Streamlit app provides:

- **Image Upload**: Support for PNG, JPG, JPEG formats
- **Synthetic Generation**: Generate test images on-the-fly
- **Real-time Segmentation**: Instant model inference
- **Interactive Visualization**: Overlay masks on original images
- **Download Results**: Save segmentation masks and overlays
- **Performance Metrics**: Inference time and accuracy metrics

## Configuration

Configuration is managed through YAML files with support for:

- **Model Parameters**: Architecture, channels, depth, dropout
- **Training Settings**: Batch size, learning rate, epochs
- **Dataset Options**: Type, size, augmentation
- **Hardware Configuration**: Device, workers, memory
- **Logging**: Wandb integration, log levels

### Example Configuration

```yaml
model:
  in_channels: 3
  out_channels: 1
  base_channels: 64
  depth: 4
  dropout_rate: 0.1
  use_attention: true

training:
  batch_size: 8
  learning_rate: 0.0001
  num_epochs: 50
  early_stopping_patience: 20

dataset:
  type: synthetic
  num_samples: 1000
  image_size: [256, 256]
  train_split: 0.8
```

## Performance Metrics

The implementation tracks multiple metrics:

- **IoU (Intersection over Union)**: Standard segmentation metric
- **Dice Score**: Similarity measure between predicted and ground truth
- **Loss**: Combined BCE and Dice loss
- **Inference Time**: Model prediction speed

## 🔧 Advanced Features

### Attention Mechanisms
- Attention gates in skip connections
- Focus on relevant features during upsampling
- Improved segmentation accuracy

### Data Augmentation
- Horizontal and vertical flips
- Rotation and brightness adjustment
- Contrast modification
- Configurable augmentation parameters

### Model Optimization
- Xavier weight initialization
- Batch normalization for stable training
- Dropout for regularization
- Learning rate scheduling

## Getting Started Examples

### Example 1: Quick Training

```python
# Quick training with default settings
python train.py
```

### Example 2: Custom Configuration

```python
from config import Config
from train import Trainer
from unet_model import ModernUNet
from dataset import create_data_loaders

# Load custom configuration
config = Config('my_config.yaml')

# Create data loaders
train_loader, val_loader = create_data_loaders(
    dataset_type=config.get('dataset.type'),
    batch_size=config.get('training.batch_size'),
    num_samples=config.get('dataset.num_samples')
)

# Create model
model = ModernUNet(**config.get('model'))

# Train
trainer = Trainer(model, train_loader, val_loader, device, config.to_dict())
trainer.train(config.get('training.num_epochs'))
```

### Example 3: Model Inference

```python
import torch
from unet_model import ModernUNet
from PIL import Image

# Load trained model
model = ModernUNet()
checkpoint = torch.load('outputs/checkpoint_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess image
image = Image.open('test_image.jpg')
image_tensor = preprocess_image(image)

# Run inference
with torch.no_grad():
    output = model(image_tensor)
    mask = (output > 0.5).float()

# Save result
mask_pil = Image.fromarray((mask.squeeze().numpy() * 255).astype('uint8'))
mask_pil.save('segmentation_mask.png')
```

## 🛠️ Development

### Running Tests

```bash
# Test individual components
python unet_model.py
python dataset.py
python config.py
```

### Adding New Features

1. **New Loss Functions**: Add to `unet_model.py`
2. **New Datasets**: Extend `dataset.py`
3. **New Metrics**: Add to training pipeline
4. **UI Improvements**: Modify `app.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Acknowledgments

- Original U-Net paper by Ronneberger et al.
- PyTorch team for the excellent framework
- Streamlit team for the web interface framework
- Open source community for inspiration and tools


# U-Net-Image-Segmentation
