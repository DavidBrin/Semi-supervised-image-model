# How to run the project

This project trains and compares semi-supervised segmentation models (U-Net, ViT, and cross-teaching) on the **Oxford-IIIT Pet** dataset (trimap: 3 classes). All data is loaded via TensorFlow Datasets, so no local image folders are required.

## Setup

1. **Environment**

   From the project root:

   ```bash
   pip install -r requirements.txt
   ```

   Main dependencies: `torch`, `torchvision`, `tensorflow-datasets`, `tensorflow`, `segmentation-models-pytorch`, `timm`.

2. **Checkpoints**

   Models are saved under `checkpoints/`. The scripts create this directory automatically.

## Data

- **Source:** [Oxford-IIIT Pet](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet) via `tensorflow_datasets`.
- **Fixed split:** All experiments use the same deterministic split (`DATA_SEED = 42`): 90% of the TFDS train split for training and 10% for validation.
- **Fair supervised baselines:** By default, supervised U-Net and supervised ViT both train on the same labeled 20% subset of the training split.
- **Semi-supervised setting:** Cross-teaching uses that same labeled 20% subset plus the remaining 80% as unlabeled images only.
- **Epoch fairness:** Cross-teaching defines one epoch as one full pass over the labeled loader, so it does not receive extra supervised updates compared with the U-Net and ViT baselines.

## GPU

- The scripts automatically use `cuda` when `torch.cuda.is_available()` is `True`; otherwise they fall back to CPU.
- A CPU-only PyTorch install will never use your GPU, even if your machine has one. If that happens, install a CUDA-enabled PyTorch build that matches your NVIDIA driver/CUDA setup.
- Each training script prints the selected device at startup.

## Running the scripts

Run from the project root.

1. **Train U-Net (supervised)**

   ```bash
   python Unet_TransferLearn.py
   ```

   Saves: `checkpoints/unet_oxford_pet.pth` and `checkpoints/unet_metrics.json`.

2. **Train ViT (supervised)**

   ```bash
   python ViT_train.py
   ```

   Saves: `checkpoints/vit_oxford_pet.pth` and `checkpoints/vit_metrics.json`.

3. **Train cross-teaching (semi-supervised)**

   ```bash
   python CrossTeachingTraining.py
   ```

   Saves: `checkpoints/unet_cross_teaching_best.pth`, `checkpoints/vit_cross_teaching_best.pth`, and `checkpoints/cross_teaching_metrics.json`.

4. **Comparison notebook**

   Open `Segmentation_Models_Comparison.ipynb` and run all cells.

## Optional

- To change the split or labeled fraction, edit `data_oxford_pet.py`.
- To change training hyperparameters, edit the `Config` class in the corresponding script.
