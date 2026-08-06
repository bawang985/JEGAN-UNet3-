# JEGAN-UNet3+

**Joint super-resolution and semantic segmentation network for photovoltaic panel detection from remote sensing imagery.**

JEGAN-UNet3+ is a multi-task GAN framework that simultaneously performs
super-resolution (SR) reconstruction and semantic segmentation of
photovoltaic (PV) panels from low-resolution satellite or aerial imagery.
The model uses a shared encoder with two task-specific decoders and
adversarial training to achieve high-quality results on both tasks.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset and Data Availability](#dataset-availability)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Ablation Study Configuration](#ablation-study-configuration)
- [Project Structure](#project-structure)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Features

- Joint multi-task learning: SR and segmentation in a single forward pass
- GAN-based SR branch: PatchGAN discriminator + perceptual losses for realistic texture
- UNet3+ style segmentation: Full-scale skip connections with ASPP bottleneck
- Ablation-friendly: Modular design with CBAM, CoordAtt, Sobel edge guidance, and SR feature injection
- Mixed precision training: AMP support for reduced GPU memory usage
- Curriculum learning: Gradual introduction of segmentation loss

## Requirements

| Component | Minimum Requirement | Recommended |
|-----------|-------------------|-------------|
| GPU Memory | 8 GB | 16 GB+ |
| CUDA | 11.0 | 11.8+ |
| Python | 3.8 | 3.10 |
| PyTorch | 2.0.0 | 2.1.0+ |
| Disk Space | 10 GB | 50 GB+ (for full dataset) |

### Dependencies

Core dependencies are listed in 
equirements.txt:

- torch >= 2.0.0
- torchvision >= 0.15.0
- rasterio (for reading GeoTIFF files)
- numpy, scikit-image, scikit-learn
- matplotlib, imageio, tifffile, Pillow
- tqdm for progress bars

## Installation

Clone the repository and install dependencies:

`
git clone https://github.com/yourusername/JEGAN-UNet3Plus.git
cd JEGAN-UNet3Plus
pip install -r requirements.txt
`

## Dataset and Data Availability

Due to the data protection policies of the ongoing academic project and
graduation thesis research, the complete large-scale Jilin-1 satellite
dataset covering the entire study area cannot be fully open-sourced at
this stage.

However, to fully support the reproducibility of this paper and demonstrate
the expected behavior of our framework, we have provided a representative
**Sample Dataset (Toy Dataset)** in the sample_data/ directory. This
subset includes paired high-resolution imagery, low-resolution inputs, and
ground-truth pixel-level annotations for rooftop photovoltaic targets.

Reviewers and users can directly run the training and testing scripts using
this sample dataset to verify the computational environment and the
algorithmic pipeline.

### Sample Dataset Structure

`
sample_data/
  train/
    LR/       Low-resolution-source RGB images (*.tif)
    HR/       High-resolution reference images (*.tif)
    label/    Binary segmentation masks (*.png)
  val/
    LR/
    HR/
    label/
  test/
    LR/
    HR/
    label/
`

### Using the Sample Dataset

Configure the data path and start training:

`
# Edit configs/default_config.py
data_dir = Path('./sample_data')

# Or override at runtime by modifying the config
python train.py
`

### Using Your Own Dataset

If you have access to remote sensing data, organize it in the same structure
as above:

- LR/HR images: GeoTIFF files with RGB bands, values in [0, 255]. If files contain an alpha or extra band, the code uses the first three bands.
- Label masks: PNG files, binary (0 = background, 1 = PV panel)
- All three folders under each split must contain the same number of files
- Filenames must match across LR, HR, and label directories

### Synthetic Data for Testing

If neither the original nor sample data is available, you can generate
synthetic data for pipeline verification:

`
python demo/generate_synthetic_data.py --output_dir ./demo_data --num_samples 100
`



## Quick Start

### Run a full demo

`
python demo/run_demo.py
`

This generates synthetic data, trains for a few epochs, and runs inference.

### Step-by-step workflow

1. Configure paths in configs/default_config.py
2. Start training: python train.py
3. Run inference on the test set: python predict.py
4. Evaluate segmentation results: python evaluate.py --pred_dir ./results/seg --gt_dir ./dataset/test/label

## Training

### Basic Training

`
python train.py
`

This uses the default configuration in configs/default_config.py.
The checkpoint is saved every epoch and the best model (lowest validation
MSE) is kept as savepath_joint/train/best.pth.

### Resume from Checkpoint

Set 
esume = True in the config to automatically find and load the latest
checkpoint (joint_model.pth). Training resumes from the saved epoch.

### Custom Configuration

Edit configs/default_config.py to modify data paths, hyperparameters,
hardware settings, and ablation switches.

### Curriculum Learning

- Epochs 0-9: Only SR loss is active (segmentation loss weight = 0)
- Epochs 10-50: Segmentation loss weight linearly increases from 0 to 1
- Epochs 50+: Full joint training with both losses at full weight

## Inference

To run inference on the test set:

`
python predict.py
`

### Outputs

| Directory | Contents |
|-----------|----------|
| results/sr/ | Super-resolved images (TIF format) |
| results/seg/ | Binary segmentation masks (PNG format) |
| results/vis/ | Side-by-side comparison plots |

Each visualization shows: LR Input, SR Output, HR Reference, Predicted Mask, Ground Truth.

Modify CHECKPOINT_PATH in predict.py to use a different checkpoint.

## Evaluation

The evaluation script computes global metrics across the entire test set:

`
python evaluate.py --pred_dir ./results/seg --gt_dir ./dataset/test/label
`

### Metrics

- Global IoU (Intersection over Union)
- Global Precision
- Global Recall
- Global F1-score

All metrics are computed at the dataset level (accumulating TP, FP, FN
across all images), following standard remote sensing evaluation practices.

## Model Architecture

The JEGAN-UNet3+ architecture consists of four main components:

### 1. Shared Feature Encoder (DFeatureExtract)

A 5-stage UNet3+ encoder with optional CBAM attention. Extracts
multi-scale features from the input LR image at 5 different resolutions.

### 2. Super-Resolution Decoder (TextureDecoder)

Reconstructs the HR image from shallow encoder features (h1, h2, h3)
using Coordinate Attention, RCAB blocks, and ESPCN sub-pixel upsampling.

### 3. Segmentation Decoder (PVSegmentationHeadUNet3Plus)

A UNet3+ style decoder with full-scale skip connections, ASPP bottleneck,
Sobel edge guidance, and SR feature injection.

### 4. PatchGAN Discriminator

A 70x70 PatchGAN that classifies local image patches as real/fake,
driving the SR branch to generate realistic textures.

## Ablation Study Configuration

Four ablation experiments are controlled by boolean flags:

| Flag | Default | Effect |
|------|---------|--------|
| use_cbam | True | CBAM channel-spatial attention in encoder |
| use_coordatt | True | Coordinate attention in SR decoder |
| use_sobel | True | Sobel edge guidance in segmentation |
| use_sr_inject | True | SR feature injection into segmentation |

## Project Structure

`
JEGAN-UNet3Plus/
  LICENSE, README.md, requirements.txt
  train.py           Joint training script
  predict.py         Inference script
  evaluate.py        Evaluation script
  configs/           Configuration files
  models/            Model architecture modules
  utils/             Training utilities
  demo/              Synthetic data generation and demo
  docs/              Detailed documentation
  comparison_models/ Baseline model implementations
  checkpoints/       Saved model checkpoints
`

## Results

### Super-Resolution Quality

Metrics: MSE approx 0.002, SSIM approx 0.92.

### Segmentation Performance

Metrics: Global IoU approx 0.85, Precision approx 0.88,
Recall approx 0.87, F1-score approx 0.87.

Note: Results depend on dataset characteristics and training configuration.

## Citation

If you use this code in your research, please cite the appropriate paper.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This work builds upon the UNet3+ architecture by Huang et al.
- PyTorch framework and torchvision for pre-trained models
- Rasterio for GeoTIFF data handling

