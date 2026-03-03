import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import timm

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- ViT seg head ---
class ViTSegmentationHead(nn.Module):
    """Decoder on ViT features."""
    def __init__(self, embed_dim=768, num_classes=3, img_size=224, patch_size=16):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.img_size = img_size

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

            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, num_patches + 1, embed_dim]
        x = x[:, 1:, :]  # drop CLS
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        x = self.decoder(x)
        return x


# ---------------------------------------------------------------------
#  ViT backbone + head
# ---------------------------------------------------------------------
class ViTSegmentation(nn.Module):
    """Vision Transformer for Segmentation with timm pretrained backbone"""
    def __init__(
        self,
        num_classes: int = 3,
        img_size: int = 224,
        freeze_backbone: bool = False,
        use_pretrained: bool = True,
        vit_npz_path: Optional[str] = None,  # optional, not needed in practice
    ):
        super().__init__()

        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=use_pretrained,
            num_classes=0,      # remove classifier head
            img_size=img_size
        )

        # Optional: override with .npz weights (not really needed)
        if vit_npz_path:
            self.load_vit_weights(vit_npz_path)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.seg_head = ViTSegmentationHead(
            embed_dim=self.backbone.embed_dim,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=self.backbone.patch_embed.patch_size[0]
        )

    def load_vit_weights(self, npz_path: str):
        logger.info(f"Loading ViT weights from {npz_path}")
        try:
            weights = np.load(npz_path)
            state_dict = {}
            for key in weights.files:
                if 'Transformer/encoderblock' in key:
                    new_key = key.replace('Transformer/encoderblock_', 'blocks.')
                    new_key = new_key.replace('/LayerNorm_0/', '.norm1.')
                    new_key = new_key.replace('/LayerNorm_2/', '.norm2.')
                    new_key = new_key.replace('/MlpBlock_3/Dense_0/', '.mlp.fc1.')
                    new_key = new_key.replace('/MlpBlock_3/Dense_1/', '.mlp.fc2.')
                    state_dict[new_key] = torch.from_numpy(weights[key])
                elif 'embedding' in key:
                    new_key = key.replace('embedding/', 'patch_embed.')
                    state_dict[new_key] = torch.from_numpy(weights[key])
            self.backbone.load_state_dict(state_dict, strict=False)
            logger.info("ViT .npz weights loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load .npz weights: {e}. Using timm weights instead.")

    def forward(self, x):
        feats = self.backbone.forward_features(x)   # [B, num_patches+1, C]
        seg_map = self.seg_head(feats)
        return seg_map


# --- Cross-teaching trainer ---
class CrossTeachingTrainer:
    """Semi-supervised: U-Net and ViT swap pseudo-labels."""
    def __init__(
        self,
        unet_model: nn.Module,
        vit_model: nn.Module,
        device: str = "cuda",
        lr: float = 1e-4,
        consistency_weight: float = 0.5,
        confidence_threshold: float = 0.9,
        unet_size: int = 512,
        vit_size: int = 224,
    ):
        self.unet = unet_model.to(device)
        self.vit = vit_model.to(device)
        self.device = device

        self.unet_size = unet_size
        self.vit_size = vit_size

        self.unet_optimizer = torch.optim.Adam(self.unet.parameters(), lr=lr)
        self.vit_optimizer = torch.optim.Adam(self.vit.parameters(), lr=lr)

        self.consistency_weight = consistency_weight
        self.confidence_threshold = confidence_threshold

        self.supervised_loss = nn.CrossEntropyLoss()

    # -------- utilities --------
    def get_confidence_mask(self, pred, threshold):
        probs = F.softmax(pred, dim=1)
        max_probs, _ = probs.max(dim=1)
        return (max_probs > threshold).float()

    # -------- labeled step --------
    def train_step_labeled(self, images, masks):
        images = images.to(self.device)
        masks = masks.to(self.device)
        # keep 3-class mask 0,1,2 (no binarize)

        # U-Net: 3-channel RGB, 512x512
        unet_images = images
        if unet_images.shape[-1] != self.unet_size:
            unet_images = F.interpolate(
                unet_images, size=(self.unet_size, self.unet_size),
                mode="bilinear", align_corners=False
            )
            masks_unet = F.interpolate(
                masks.float().unsqueeze(1),
                size=(self.unet_size, self.unet_size),
                mode="nearest"
            ).squeeze(1).long()
        else:
            masks_unet = masks

        # ViT expects 3 channels, 224x224
        vit_images = images
        if vit_images.shape[-1] != self.vit_size:
            vit_images = F.interpolate(
                vit_images, size=(self.vit_size, self.vit_size),
                mode="bilinear", align_corners=False
            )
            masks_vit = F.interpolate(
                masks.unsqueeze(1).float(),
                size=(self.vit_size, self.vit_size),
                mode="nearest"
            ).squeeze(1).round().long()
        else:
            masks_vit = masks

        # forward
        unet_pred = self.unet(unet_images)   # [B,C,512,512]
        vit_pred = self.vit(vit_images)      # [B,C,224,224]

        unet_loss = self.supervised_loss(unet_pred, masks_unet)
        vit_loss = self.supervised_loss(vit_pred, masks_vit)

        # backward
        self.unet_optimizer.zero_grad()
        unet_loss.backward()
        self.unet_optimizer.step()

        self.vit_optimizer.zero_grad()
        vit_loss.backward()
        self.vit_optimizer.step()

        return {"unet_loss": unet_loss.item(), "vit_loss": vit_loss.item()}

    # -------- unlabeled step (cross-teaching) --------
    def train_step_unlabeled(self, images):
        images = images.to(self.device)  # [B,3,H,W]

        # U-Net input: 3-channel
        unet_images = images
        if unet_images.shape[-1] != self.unet_size:
            unet_images = F.interpolate(
                unet_images, size=(self.unet_size, self.unet_size),
                mode="bilinear", align_corners=False
            )

        # Prepare ViT input
        vit_images = images
        if vit_images.shape[-1] != self.vit_size:
            vit_images = F.interpolate(
                vit_images, size=(self.vit_size, self.vit_size),
                mode="bilinear", align_corners=False
            )

        with torch.no_grad():
            # teacher predictions
            unet_teacher = self.unet(unet_images)        # [B,C,512,512]
            vit_teacher = self.vit(vit_images)          # [B,C,224,224]

            # confidence (for logging)
            unet_conf = self.get_confidence_mask(unet_teacher, self.confidence_threshold)
            vit_conf = self.get_confidence_mask(vit_teacher, self.confidence_threshold)

            # create pseudo-labels in opposite model's resolution
            # ViT -> U-Net (upsample to 512)
            vit_teacher_512 = F.interpolate(
                vit_teacher, size=(self.unet_size, self.unet_size),
                mode="bilinear", align_corners=False
            )
            vit_pseudo_labels_512 = vit_teacher_512.argmax(dim=1)   # [B,512,512]

            # U-Net -> ViT (downsample to 224)
            unet_teacher_224 = F.interpolate(
                unet_teacher, size=(self.vit_size, self.vit_size),
                mode="bilinear", align_corners=False
            )
            unet_pseudo_labels_224 = unet_teacher_224.argmax(dim=1) # [B,224,224]

        # ------- Train U-Net with ViT pseudo-labels -------
        unet_pred = self.unet(unet_images)  # [B,C,512,512]
        unet_consistency = self.supervised_loss(unet_pred, vit_pseudo_labels_512)

        self.unet_optimizer.zero_grad()
        (self.consistency_weight * unet_consistency).backward()
        self.unet_optimizer.step()

        # ------- Train ViT with U-Net pseudo-labels -------
        vit_pred = self.vit(vit_images)     # [B,C,224,224]
        vit_consistency = self.supervised_loss(vit_pred, unet_pseudo_labels_224)

        self.vit_optimizer.zero_grad()
        (self.consistency_weight * vit_consistency).backward()
        self.vit_optimizer.step()

        return {
            "unet_consistency_loss": unet_consistency.item(),
            "vit_consistency_loss": vit_consistency.item(),
            "unet_confidence": unet_conf.mean().item(),
            "vit_confidence": vit_conf.mean().item(),
        }

    # -------- epoch loop --------
    def train_epoch(self, labeled_loader, unlabeled_loader, epoch: int):
        self.unet.train()
        self.vit.train()

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        num_batches = max(len(labeled_loader), len(unlabeled_loader))

        metrics = {
            "unet_loss": 0.0,
            "vit_loss": 0.0,
            "unet_consistency": 0.0,
            "vit_consistency": 0.0,
        }

        for batch_idx in range(num_batches):
            # labeled step
            try:
                images_l, masks_l = next(labeled_iter)
                labeled_metrics = self.train_step_labeled(images_l, masks_l)
                metrics["unet_loss"] += labeled_metrics["unet_loss"]
                metrics["vit_loss"] += labeled_metrics["vit_loss"]
            except StopIteration:
                labeled_iter = iter(labeled_loader)

            # unlabeled step
            try:
                images_u = next(unlabeled_iter)
                if isinstance(images_u, (list, tuple)):
                    images_u = images_u[0]
                unlabeled_metrics = self.train_step_unlabeled(images_u)
                metrics["unet_consistency"] += unlabeled_metrics["unet_consistency_loss"]
                metrics["vit_consistency"] += unlabeled_metrics["vit_consistency_loss"]
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}")

        for k in metrics:
            metrics[k] /= num_batches

        logger.info(f"Epoch {epoch} - Metrics: {metrics}")
        return metrics

    # -------- ensemble prediction (for evaluation / visualization) --------
    def ensemble_predict(self, images):
        self.unet.eval()
        self.vit.eval()
        with torch.no_grad():
            images = images.to(self.device)  # [B,3,H,W]

            # U-Net input
            unet_images = images.mean(dim=1, keepdim=True)
            unet_images = F.interpolate(
                unet_images, size=(self.unet_size, self.unet_size),
                mode="bilinear", align_corners=False
            )
            unet_logits = self.unet(unet_images)  # [B,C,512,512]
            unet_probs = F.softmax(unet_logits, dim=1)

            # ViT input
            vit_images = images
            vit_images = F.interpolate(
                vit_images, size=(self.vit_size, self.vit_size),
                mode="bilinear", align_corners=False
            )
            vit_logits = self.vit(vit_images)      # [B,C,224,224]
            vit_logits_512 = F.interpolate(
                vit_logits, size=(self.unet_size, self.unet_size),
                mode="bilinear", align_corners=False
            )
            vit_probs = F.softmax(vit_logits_512, dim=1)

            ensemble_probs = (unet_probs + vit_probs) / 2
            return ensemble_probs  # [B,C,512,512]


# --- U-Net: create fresh or load checkpoint ---
def create_fresh_unet(device: str = "cuda"):
    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
    )
    model.to(device)
    logger.info("Created fresh U-Net (no checkpoint).")
    return model


def load_unet_model(unet_path: str, device: str = "cuda"):
    unet_model = create_fresh_unet(device)
    path = Path(unet_path)
    if not path.exists():
        logger.info(f"No checkpoint at {unet_path}; using fresh U-Net.")
        return unet_model
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    unet_model.load_state_dict(state_dict, strict=False)
    unet_model.eval()
    logger.info(f"Loaded U-Net from {unet_path}")
    return unet_model

# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------
def main():
    _root = Path(__file__).resolve().parent
    checkpoints_dir = _root / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    unet_path = str(checkpoints_dir / "unet_oxford_pet.pth")
    vit_path = str(checkpoints_dir / "vit_oxford_pet.pth")

    num_classes = 3
    unet_img_size = 512
    vit_img_size = 224
    batch_size = 8
    num_epochs = 30
    learning_rate = 1e-4
    consistency_weight = 0.5
    confidence_threshold = 0.9
    unlabeled_fraction = 0.5  # half of train used as "unlabeled" for consistency

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")

    from data_oxford_pet import get_oxford_pet_datasets_for_cross_teaching
    labeled_loader, unlabeled_loader = get_oxford_pet_datasets_for_cross_teaching(
        unet_img_size=unet_img_size,
        batch_size=batch_size,
        num_workers=0,
        unlabeled_fraction=unlabeled_fraction,
    )
    logger.info(f"Labeled batches: {len(labeled_loader)}, Unlabeled batches: {len(unlabeled_loader)}")

    # ===== Models =====
    unet_model = load_unet_model(unet_path, device)

    vit_model = ViTSegmentation(
        num_classes=num_classes,
        img_size=vit_img_size,
        freeze_backbone=False,
        use_pretrained=True,
        vit_npz_path=None,
    ).to(device)

    # ===== Trainer =====
    trainer = CrossTeachingTrainer(
        unet_model=unet_model,
        vit_model=vit_model,
        device=device,
        lr=learning_rate,
        consistency_weight=consistency_weight,
        confidence_threshold=confidence_threshold,
        unet_size=unet_img_size,
        vit_size=vit_img_size,
    )

    # ===== Training loop =====
    for epoch in range(num_epochs):
        metrics = trainer.train_epoch(labeled_loader, unlabeled_loader, epoch)

        if (epoch + 1) % 10 == 0:
            torch.save(trainer.unet.state_dict(), checkpoints_dir / f"unet_epoch_{epoch+1}.pth")
            torch.save(trainer.vit.state_dict(), checkpoints_dir / f"vit_epoch_{epoch+1}.pth")

    torch.save(trainer.unet.state_dict(), unet_path)
    torch.save(trainer.vit.state_dict(), vit_path)
    logger.info(f"Saved final models to {checkpoints_dir}")


if __name__ == "__main__":
    main()
