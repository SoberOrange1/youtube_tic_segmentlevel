# Two-Stream Online Tic Detection

A deep learning framework for real-time tic detection using a two-stream neural network architecture that combines visual (face) and facial landmark/blendshape features.

## Overview

This project implements a sophisticated two-stream neural network for detecting tics in video sequences. The system processes both visual facial features through a Vision Transformer (ViT) and temporal facial landmark/blendshape data through Temporal Convolutional Networks (TCNs), enabling robust tic detection with high accuracy.


## Architecture

### Model Variants

1.  **FrameProbAvgFusionModel** **(Use this directly)**: Experimental approach that averages per-frame probabilities across streams
2.  **LateFusionModel** (Optional but not the best performance): Traditional late fusion approach where features are combined after temporal processing


### Core Components

- **ViTFrameEncoder**: Vision Transformer encoder with optional LoRA fine-tuning
- **TemporalTCN**: Temporal Convolutional Network for sequence modeling
- **Segment Aggregation**: Peak-aware scoring for segment-level classification

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)

### Dependencies

```bash
pip install torch torchvision transformers
pip install peft  # for LoRA fine-tuning
pip install scikit-learn tqdm numpy pillow
pip install matplotlib seaborn  # for visualization
```

### Custom Dependencies

The project includes custom implementations:
- `pytorch_tcn/`: Custom TCN implementation
- `losses/`: Ranking loss functions
- `utils/`: Utility functions for segment aggregation

## Dataset Structure

The dataset should follow this directory structure:

```
data_folder/cropped_faces/
├── tic/
│   ├── 01/  # two-digit video ID
│   │   ├── segment_001/
│   │   │   ├── frame_001.jpg
│   │   │   ├── frame_002.jpg
│   │   │   ├── ...
│   │   │   ├── blendshapes/ 
│   │   │   │   ├── frame_001.npy
│   │   │   │   ├── frame_002.npy
│   │   │   │   └── ...
│   │   │   └── face_landmarks_blendshapes.json  # JSON format (preferred)
│   │   └── segment_002/
│   └── 02/
└── non-tic/
    ├── 01/
    └── 02/
```

### Data Formats

1. **JSON Format** (Recommended): Single JSON file per segment with frame-by-frame blendshape data
2. **Per-frame Files**: Individual `.npy`, `.txt`, or `.csv` files for each frame
3. **Segment-level Files**: Single file with temporal sequence data

## Configuration

The main configuration is defined in `main_two_stream.py`. Key parameters include:

```python
config = {
    # Dataset
    'segment_root': '/path/to/data_folder/cropped_faces',
    'target_len': 30,  # frames per segment
    'blend_dim': 52,   # landmark feature dimension
    
    # Training
    'epochs': 150,
    'batch_size': 8,
    'lr': 1e-4,
    'seed': 42,
    
    # Model Architecture
    'vit_model_name': 'trpakov/vit-face-expression',
    'model_variant': 'frame_prob_avg',  # Best on 'frame_prob_avg'
    'fusion_mode': 'concat',  # or 'add'
    
    # LoRA Fine-tuning
    'vit_use_lora': True,
    'vit_lora_r': 8,
    'vit_lora_alpha': 32,
    'vit_lora_targets': ['query', 'value'],
    
    # Ranking Loss (Set it to False)
    'rank_use': False,
    'rank_margin': 0.5,
    'rank_alpha': 0.1,
    
    # CE Threshold Analysis
    'threshold_analysis': True,
    'threshold_start': 0.1,
    'threshold_end': 0.9,
    'threshold_step': 0.05,
}
```

## Usage

### Training

```bash
python main_two_stream.py
```

The script automatically:
- Performs 5-fold cross-validation
- Saves best models for each fold
- Generates comprehensive training logs
- Performs threshold sensitivity analysis
- Creates inference-ready model packages

### Testing

```bash
# Modify mode in main_two_stream.py
mode = 'test'
python main_two_stream.py
```

### Key Training Features

- **Early Stopping**: Automatically stops training when validation loss plateaus
- **Checkpointing**: Regular model checkpoints for training resumption
- **Video-aware Sampling**: Custom sampler ensures proper negative-first ordering for ranking loss
- **Memory Optimization**: Gradient checkpointing reduces memory usage during training

## Model Outputs

The framework generates several output files:

- `best_model_fold_X.pth`: Best model by validation loss
- `best_model_by_acc_fold_X.pth`: Best model by accuracy
- `best_inference_fold_X.pt`: Complete inference package with optimal threshold
- `checkpoint_fold_X.pth`: Training checkpoint for resumption
- `training.log`: Comprehensive training logs
- `confusion_matrix.png` & `roc_curve.png`: Evaluation visualizations

## Advanced Features

### Ranking Loss

Optional pairwise ranking loss improves discrimination between tic and non-tic segments:

```python
config['rank_use'] = True
config['rank_margin'] = 0.5  # hinge margin
config['rank_alpha'] = 0.1   # peak vs local average blend
```

### Temporal Smoothness Loss

Encourages smooth temporal transitions in per-frame predictions:

```python
config['smooth_lambda'] = 0.01  # smoothness loss weight
```

### Threshold Analysis

Comprehensive threshold sensitivity analysis finds optimal classification thresholds:

- Coarse-grained search across specified range
- Fine-grained search around best threshold
- Separate analysis for training and validation sets
- Multiple evaluation metrics (accuracy, precision, recall, F1)

### LoRA Fine-tuning

Efficient adaptation of pre-trained ViT models:

```python
config['vit_use_lora'] = True
config['vit_lora_r'] = 8          # rank
config['vit_lora_alpha'] = 32     # scaling factor
config['vit_lora_targets'] = ['query', 'value']  # target modules
```

## Performance Monitoring

The framework provides extensive monitoring:

- Per-epoch loss decomposition (CE, ranking, smoothness)
- Training and validation accuracy tracking
- Threshold sensitivity analysis
- Video-level pairing statistics for ranking loss
- Memory usage optimization logs

## Model Architecture Details

### Vision Stream
1. **ViTFrameEncoder**: Processes cropped face images
   - Pre-trained on facial expression data
   - Optional LoRA fine-tuning for domain adaptation
   - Gradient checkpointing for memory efficiency

2. **ViT TCN**: Temporal modeling of visual features
   - Causal convolutions for online processing
   - Configurable depth and channel dimensions

### Landmark Stream
1. **Facial Features**: Processes blendshape/landmark data
   - Supports multiple input formats (JSON, NPY, TXT, CSV)
   - Automatic temporal alignment with visual stream

2. **Blend TCN**: Temporal modeling of facial landmarks
   - Lightweight architecture for real-time processing
   - Batch normalization and GELU activations

### Fusion Strategies
- **Concatenation**: Combines features in feature space
- **Addition**: Element-wise fusion (requires matching dimensions)
- **Frame-level averaging**: Averages probabilities across streams

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Missing Landmark Files**: Check dataset structure and enable `strict_blend=False`
3. **Corrupted Checkpoints**: Training automatically detects and removes corrupted files
4. **Threshold Convergence**: Adjust threshold analysis range and step size

### Performance Tips

1. Use mixed precision training for faster convergence
2. Enable gradient checkpointing for large models
3. Optimize data loading with appropriate `num_workers`
4. Use LoRA for efficient fine-tuning of large ViT models

For additional support, please check the training logs and ensure your dataset follows the expected structure.