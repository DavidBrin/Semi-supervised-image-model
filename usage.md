# How to run the project

This project trains and compares semi-supervised segmentation models (U-Net, ViT, and cross-teaching) on the **Oxford-IIIT Pet** dataset (trimap: 3 classes). All data is loaded via TensorFlow Datasets; no local data folders are required.

## Setup

1. **Environment**

   From the project root:

   ```bash
   pip install -r requirements.txt
   ```

   Main dependencies: `torch`, `torchvision`, `tensorflow-datasets`, `tensorflow`, `segmentation-models-pytorch`, `timm`.

2. **Checkpoints**

   Models are saved under the project folder in a `checkpoints/` directory. The scripts create this directory if it does not exist. You do **not** need any pre-saved models: training creates them.

## Data

- **Source:** [Oxford-IIIT Pet](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet) via `tensorflow_datasets` (train/test splits).
- **Split:** All scripts use the same train/val/test split (seed in `data_oxford_pet.DATA_SEED`), so supervised U-Net, ViT, and the comparison notebook see the same training data.
- **Cross-teaching:** Labeled data uses masks for the supervised loss; the “unlabeled” stream is **images only** (no masks). Pseudo-labels come only from the other model.

## Running the scripts

Run from the **project root** (the directory that contains `Unet_TransferLearn.py`, `ViT_train.py`, etc.).

1. **Train U-Net (supervised)**

   ```bash
   python Unet_TransferLearn.py
   ```

   Saves: `checkpoints/unet_oxford_pet.pth`.

2. **Train ViT (supervised)**

   ```bash
   python ViT_train.py
   ```

   Saves: `checkpoints/vit_oxford_pet.pth`.

3. **Train cross-teaching (semi-supervised)**

   ```bash
   python CrossTeachingTraining.py
   ```

   - Does **not** require existing checkpoints: if `unet_oxford_pet.pth` / `vit_oxford_pet.pth` are missing, it starts from fresh U-Net and ViT (ImageNet init for encoders).
   - Saves: `checkpoints/unet_oxford_pet.pth`, `checkpoints/vit_oxford_pet.pth`, and optional `checkpoints/unet_epoch_*.pth`, `checkpoints/vit_epoch_*.pth` every 10 epochs.

4. **Comparison notebook**

   Open `Segmentation_Models_Comparison.ipynb` and run all cells.

   - Loads data via `data_oxford_pet` (same split as training).
   - Loads U-Net and ViT from `checkpoints/` (via `comparison_utils`). If checkpoints are missing, the notebook will report that you need to run the training scripts first.
   - Evaluates U-Net, ViT, and their ensemble on the test set and prints a **best-case** comparison (which model has the best Dice/IoU).

## Paths

- **Checkpoints:** `./checkpoints/` (relative to the project root). Scripts use `Path(__file__).resolve().parent / "checkpoints"` so the folder lives next to the scripts.
- **Data:** No local paths; dataset is downloaded by TFDS (e.g. under `~/tensorflow_datasets/` or as set by `TFDS_DATA_DIR`).

## Optional

- To change the data split or labeled/unlabeled ratio, edit `data_oxford_pet.py` (`DATA_SEED`, `val_fraction`, `unlabeled_fraction` in the loader functions).
- To change training hyperparameters, edit the `Config` class or `main()` in the corresponding script.
