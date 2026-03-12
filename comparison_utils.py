import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("Please install segmentation-models-pytorch first.")
    raise

from data_oxford_pet import NUM_CLASSES


ROOT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"


class ViTSegmentationHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=3):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # remove cls token and reshape patches into feature map
        x = x[:, 1:, :]
        b, n, c = x.shape
        hw = int(n ** 0.5)
        x = x.transpose(1, 2).reshape(b, c, hw, hw)
        return self.decoder(x)


class ViTSegmentation(nn.Module):
    def __init__(self, num_classes=3, img_size=224, freeze_backbone=False, use_pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=use_pretrained,
            num_classes=0,
            img_size=img_size,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.seg_head = ViTSegmentationHead(
            embed_dim=self.backbone.embed_dim,
            num_classes=num_classes,
        )

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        return self.seg_head(feats)


def create_unet():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    return model


def load_checkpoint_model(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()
    return model


def load_unet(device="cpu", checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINTS_DIR / "unet_oxford_pet.pth"

    model = create_unet()
    return load_checkpoint_model(model, checkpoint_path, device)


def load_vit(device="cpu", checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINTS_DIR / "vit_oxford_pet.pth"

    model = ViTSegmentation(
        num_classes=NUM_CLASSES,
        img_size=224,
        freeze_backbone=False,
        use_pretrained=True,
    )
    return load_checkpoint_model(model, checkpoint_path, device)


class EnsembleInference:
    def __init__(self, unet, vit, device="cpu", unet_size=512, vit_size=224):
        self.unet = unet.eval()
        self.vit = vit.eval()
        self.device = device
        self.unet_size = unet_size
        self.vit_size = vit_size

    @torch.no_grad()
    def predict_logits(self, images):
        images = images.to(self.device)

        unet_in = F.interpolate(
            images,
            size=(self.unet_size, self.unet_size),
            mode="bilinear",
            align_corners=False,
        )

        vit_in = F.interpolate(
            images,
            size=(self.vit_size, self.vit_size),
            mode="bilinear",
            align_corners=False,
        )

        unet_logits = self.unet(unet_in)
        vit_logits = self.vit(vit_in)

        vit_logits = F.interpolate(
            vit_logits,
            size=(self.unet_size, self.unet_size),
            mode="bilinear",
            align_corners=False,
        )

        return (unet_logits + vit_logits) / 2.0

    @torch.no_grad()
    def predict_mask(self, images):
        logits = self.predict_logits(images)
        return logits.argmax(dim=1)


@torch.no_grad()
def predict_unet(unet, image_tensor, device="cpu", image_size=512):
    image_tensor = image_tensor.to(device)

    x = F.interpolate(
        image_tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    logits = unet(x)
    return logits.argmax(dim=1)


@torch.no_grad()
def predict_vit(vit, image_tensor, device="cpu", image_size=224):
    image_tensor = image_tensor.to(device)

    x = F.interpolate(
        image_tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    logits = vit(x)
    return logits.argmax(dim=1)


def dice_score_macro(pred, target, num_classes=NUM_CLASSES, eps=1e-6):
    pred = np.asarray(pred).squeeze()
    target = np.asarray(target).squeeze()

    dice_scores = []

    for cls in range(num_classes):
        pred_cls = (pred == cls).astype(np.float32)
        target_cls = (target == cls).astype(np.float32)

        inter = (pred_cls * target_cls).sum()
        denom = pred_cls.sum() + target_cls.sum()

        dice = (2.0 * inter + eps) / (denom + eps)
        dice_scores.append(dice)

    return float(np.mean(dice_scores))


def iou_score_macro(pred, target, num_classes=NUM_CLASSES, eps=1e-6):
    pred = np.asarray(pred).squeeze()
    target = np.asarray(target).squeeze()

    iou_scores = []

    for cls in range(num_classes):
        pred_cls = (pred == cls).astype(np.float32)
        target_cls = (target == cls).astype(np.float32)

        inter = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - inter

        iou = (inter + eps) / (union + eps)
        iou_scores.append(iou)

    return float(np.mean(iou_scores))


def pixel_accuracy(pred, target):
    pred = np.asarray(pred).squeeze()
    target = np.asarray(target).squeeze()
    return float((pred == target).mean())


@torch.no_grad()
def evaluate_models(dataset, unet, vit, ensemble, device="cpu"):
    unet.eval()
    vit.eval()

    metrics = {
        "U-Net": {"dice": [], "iou": [], "pixel_accuracy": []},
        "ViT": {"dice": [], "iou": [], "pixel_accuracy": []},
        "Ensemble": {"dice": [], "iou": [], "pixel_accuracy": []},
    }

    for i in range(len(dataset)):
        image, mask = dataset[i]

        image_batch = image.unsqueeze(0).to(device)
        target = mask.cpu().numpy()

        pred_unet = predict_unet(unet, image_batch, device=device).squeeze(0).cpu().numpy()
        pred_vit = predict_vit(vit, image_batch, device=device).squeeze(0).cpu().numpy()
        pred_ens = ensemble.predict_mask(image_batch).squeeze(0).cpu().numpy()

        for name, pred in [("U-Net", pred_unet), ("ViT", pred_vit), ("Ensemble", pred_ens)]:
            metrics[name]["dice"].append(dice_score_macro(pred, target))
            metrics[name]["iou"].append(iou_score_macro(pred, target))
            metrics[name]["pixel_accuracy"].append(pixel_accuracy(pred, target))

    summary = {}

    for name in metrics:
        summary[name] = {
            "dice": float(np.mean(metrics[name]["dice"])),
            "iou": float(np.mean(metrics[name]["iou"])),
            "pixel_accuracy": float(np.mean(metrics[name]["pixel_accuracy"])),
        }

    return metrics, summary


def load_metrics_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_history(metrics_json, model_name="Model"):
    history = metrics_json["history"]

    epochs = [row["epoch"] for row in history]
    train_loss = [row.get("train_loss", None) for row in history]
    val_dice = [row.get("val_dice", row.get("ensemble_val_dice", None)) for row in history]
    val_iou = [row.get("val_iou", row.get("ensemble_val_iou", None)) for row in history]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss)
    plt.title(f"{model_name} Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_dice)
    plt.title(f"{model_name} Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")

    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_iou)
    plt.title(f"{model_name} Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")

    plt.tight_layout()
    plt.show()


def show_image_and_mask(image, mask, title=""):
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    mask = mask.detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    if title:
        plt.suptitle(title)

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="tab10", vmin=0, vmax=NUM_CLASSES - 1)
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def compare_predictions(image, mask, unet_pred, vit_pred, ensemble_pred, title=""):
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    mask = mask.detach().cpu().numpy()

    unet_pred = unet_pred.detach().cpu().numpy().squeeze()
    vit_pred = vit_pred.detach().cpu().numpy().squeeze()
    ensemble_pred = ensemble_pred.detach().cpu().numpy().squeeze()

    plt.figure(figsize=(16, 4))
    if title:
        plt.suptitle(title)

    items = [
        (image, "Image"),
        (mask, "Ground Truth"),
        (unet_pred, "U-Net"),
        (vit_pred, "ViT"),
        (ensemble_pred, "Ensemble"),
    ]

    for i, (arr, name) in enumerate(items):
        plt.subplot(1, 5, i + 1)

        if i == 0:
            plt.imshow(arr)
        else:
            plt.imshow(arr, cmap="tab10", vmin=0, vmax=NUM_CLASSES - 1)

        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()