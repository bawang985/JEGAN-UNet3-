from pathlib import Path


class Config:
    """Default configuration for training JEGAN-UNet3+.

    Modify the fields below to match your dataset paths, hardware,
    and ablation experiment settings.
    """
    # ---------- Data parameters ----------
    num_bands = 3            # RGB bands used by the model; extra bands are discarded
    batch_size = 2
    num_workers = 4
    data_dir = Path('sample_data')  # Root directory containing train/val/test sub-folders
    image_size = [512, 512]

    # ---------- Training hyperparameters ----------
    lr = 1e-4
    epochs = 200
    cuda = True
    ngpu = 2

    # ---------- Checkpoint / save ----------
    save_dir = Path('savepath_joint')
    resume = True           # Automatically resume from latest checkpoint if True

    # ---------- Ablation switches ----------
    use_cbam = True         # Channel-Spatial Attention Module (CBAM)
    use_coordatt = True     # Coordinate Attention for directional priors
    use_sobel = True        # Sobel edge guidance for segmentation
    use_sr_inject = True    # SR feature injection into segmentation branch
