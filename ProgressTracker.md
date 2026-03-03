# Progress Tracker

Track changes and to-dos for the Oxford-IIIT Pet redesign.

## Done

- Migrate data loading to TFDS `oxford_iiit_pet`; shared module `data_oxford_pet.py`; train/val/test and cross-teaching loaders.
- Update all segmentation models and configs to 3 classes (trimap); macro Dice loss/coefficient.
- Add dependency `tensorflow-datasets` (and `tensorflow`) in `requirements.txt`.
- Unlabeled-data policy: use subset of TFDS train as "unlabeled" for consistency loss (`unlabeled_fraction` in cross-teaching).
- Comment pass: shorter, more casual comments across scripts.

## In progress

- (none)

## To-do

- **Object detection:** Add object detection using `head_bbox` and class `label` (37 breeds) from Oxford-IIIT Pet. Options: (1) Use head_bbox as ground truth for a bbox regression head (e.g. on top of U-Net/ViT encoder); (2) Train a small detector (e.g. one anchor per image, or use existing detection head) with label as class and head_bbox as target; (3) Document in a new script or notebook and track in ProgressTracker.
- remove all comments in python files

## Notes

- **3 vs 37:** Seg output is **3 classes** (trimap). The `label` field is **37 breeds** (image-level). Use `label` for an optional classification head or logging; keep seg head at 3.
