    # Helpers for Segmentation_Models_Comparison notebook: load models, metrics, viz.
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from data_oxford_pet import NUM_SEGMENTATION_CLASSES

# Reuse ViT from CrossTeachingTraining (same as ViT_train)
from CrossTeachingTraining import ViTSegmentation, load_unet_model as _load_unet


def get_checkpoints_dir():
    return Path(__file__).resolve().parent / "checkpoints"


def load_unet(device="cuda", path=None):
    path = path or get_checkpoints_dir() / "unet_oxford_pet.pth"
    return _load_unet(str(path), device)


def load_vit(device="cuda", path=None):
    path = path or get_checkpoints_dir() / "vit_oxford_pet.pth"
    model = ViTSegmentation(num_classes=3, img_size=224, freeze_backbone=False, use_pretrained=True).to(device)
    p = Path(path)
    if p.exists():
        state = torch.load(p, map_location=device)
        state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


class EnsembleInference:
    """Ensemble prediction for eval (U-Net + ViT, no training)."""

    def __init__(self, unet, vit, device="cuda"):
        self.unet = unet.eval()
        self.vit = vit.eval()
        self.device = device

    @torch.no_grad()
    def predict(self, images):
        images = images.to(self.device)
        unet_in = F.interpolate(images, size=(512, 512), mode="bilinear", align_corners=False)
        vit_in = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        unet_pred = self.unet(unet_in)
        vit_pred = self.vit(vit_in)
        unet_up = F.interpolate(unet_pred, size=images.shape[-2:], mode="bilinear", align_corners=False)
        vit_up = F.interpolate(vit_pred, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return (unet_up + vit_up) / 2


def predict_unet(unet, image_tensor, device="cuda"):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        x = F.interpolate(image_tensor, size=(512, 512), mode="bilinear", align_corners=False)
        pred = unet(x)
        return F.interpolate(pred, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)


def predict_vit(vit, image_tensor, device="cuda"):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        x = F.interpolate(image_tensor, size=(224, 224), mode="bilinear", align_corners=False)
        pred = vit(x)
        return F.interpolate(pred, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)


def dice_score_macro(pred, target, num_classes=3, eps=1e-6):
    pred = np.asarray(pred).squeeze()
    target = np.asarray(target).squeeze()
    if pred.ndim == 3:
        pred = pred.argmax(axis=0)
    d = []
    for c in range(num_classes):
        pc = (pred == c).astype(np.float32)
        tc = (target == c).astype(np.float32)
        inter = (pc * tc).sum()
        tot = pc.sum() + tc.sum()
        d.append((2 * inter + eps) / (tot + eps))
    return float(np.mean(d))


def iou_score_macro(pred, target, num_classes=3, eps=1e-6):
    pred = np.asarray(pred).squeeze()
    target = np.asarray(target).squeeze()
    if pred.ndim == 3:
        pred = pred.argmax(axis=0)
    ious = []
    for c in range(num_classes):
        pc = (pred == c).astype(np.float32)
        tc = (target == c).astype(np.float32)
        inter = (pc * tc).sum()
        union = pc.sum() + tc.sum() - inter
        ious.append(inter / (union + eps))
    return float(np.mean(ious))


def evaluate_models(dataset, unet, vit, ensemble, device="cuda"):
    """Returns dict of model name -> {dice, iou} and summary table."""
    metrics = {"U-Net": {"dice": [], "iou": []}, "ViT": {"dice": [], "iou": []}, "Ensemble": {"dice": [], "iou": []}}
    unet.eval()
    vit.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            img_t, mask_t = dataset[i]
            img = img_t.to(device).unsqueeze(0)
            target = mask_t.unsqueeze(0)
            pred_unet = predict_unet(unet, img, device).argmax(dim=1)
            pred_vit = predict_vit(vit, img, device).argmax(dim=1)
            pred_ens = ensemble.predict(img).argmax(dim=1)
            for name, pred in [("U-Net", pred_unet), ("ViT", pred_vit), ("Ensemble", pred_ens)]:
                metrics[name]["dice"].append(dice_score_macro(pred.cpu().numpy(), target.numpy()))
                metrics[name]["iou"].append(iou_score_macro(pred.cpu().numpy(), target.numpy()))
    summary = {
        "U-Net": (np.mean(metrics["U-Net"]["dice"]), np.mean(metrics["U-Net"]["iou"])),
        "ViT": (np.mean(metrics["ViT"]["dice"]), np.mean(metrics["ViT"]["iou"])),
        "Ensemble": (np.mean(metrics["Ensemble"]["dice"]), np.mean(metrics["Ensemble"]["iou"])),
    }
    return metrics, summary


def show_image(img_tensor, title="Image"):
    import matplotlib.pyplot as plt
    img = img_tensor.squeeze().cpu().numpy()
    if img.ndim == 3:
        img = np.moveaxis(img, 0, -1)
    plt.figure(figsize=(4, 4))
    plt.imshow(img if img.shape[-1] == 3 else img.squeeze(), cmap="gray" if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] != 3) else None)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_image_and_mask(img, mask, title=""):
    import matplotlib.pyplot as plt
    img = img.squeeze().cpu().numpy()
    if img.ndim == 3:
        img = np.moveaxis(img, 0, -1)
    mask = mask.squeeze().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.suptitle(title or "Oxford Pet sample")
    plt.subplot(1, 2, 1)
    plt.imshow(img if img.shape[-1] == 3 else img, cmap=None if img.ndim == 3 and img.shape[-1] == 3 else "gray")
    plt.title("Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="tab10", vmin=0, vmax=NUM_SEGMENTATION_CLASSES)
    plt.title("Mask (trimap)")
    plt.axis("off")
    plt.show()


def compare_predictions(img, mask, unet_pred, vit_pred, ensemble_pred, title=""):
    import matplotlib.pyplot as plt
    img = img.squeeze().cpu().numpy()
    if img.ndim == 3:
        img = np.moveaxis(img, 0, -1)
    mask = mask.squeeze().cpu().numpy()

    def to_display(p):
        p = p.squeeze().cpu()
        return p.argmax(dim=0).numpy() if p.dim() == 3 else p.numpy()

    u, v, e = to_display(unet_pred), to_display(vit_pred), to_display(ensemble_pred)
    plt.figure(figsize=(16, 6))
    plt.suptitle(f"Segmentation – {title or 'sample'}", fontsize=14)
    for i, (arr, t) in enumerate(zip([img, mask, u, v, e], ["Input", "GT Mask", "U-Net", "ViT", "Ensemble"])):
        plt.subplot(1, 5, i + 1)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            plt.imshow(arr)
        else:
            plt.imshow(arr, cmap="tab10", vmin=0, vmax=max(2, NUM_SEGMENTATION_CLASSES))
        plt.title(t)
        plt.axis("off")
    plt.show()
