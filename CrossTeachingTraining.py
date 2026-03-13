import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("Please install segmentation-models-pytorch first.")
    raise

from data_oxford_pet import (
    DEFAULT_LABELED_FRACTION,
    DEFAULT_VAL_FRACTION,
    NUM_CLASSES,
    get_oxford_pet_datasets_for_cross_teaching,
    set_seed,
)


class Config:
    unet_image_size = 512
    vit_image_size = 224
    batch_size = 4
    epochs = 8
    learning_rate = 1e-4
    val_fraction = DEFAULT_VAL_FRACTION
    labeled_fraction = DEFAULT_LABELED_FRACTION
    confidence_threshold = 0.75
    consistency_weight = 0.5
    freeze_unet_encoder = True
    freeze_vit_backbone = True
    num_workers = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
UNET_SAVE_PATH = os.path.join(CHECKPOINTS_DIR, "unet_cross_teaching_best.pth")
VIT_SAVE_PATH = os.path.join(CHECKPOINTS_DIR, "vit_cross_teaching_best.pth")
METRICS_SAVE_PATH = os.path.join(CHECKPOINTS_DIR, "cross_teaching_metrics.json")


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def describe_device(device):
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device.index or 0)})"
    if device.type == "mps":
        return "mps"
    return "cpu"


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)

        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * target_one_hot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + target_one_hot.sum(dim=dims)

        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        dice_loss = 1.0 - dice.mean()

        return ce_loss + dice_loss


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, pseudo_labels, valid_mask):
        # only train on pseudo labels from images that pass the confidence threshold
        per_pixel_loss = self.ce(logits, pseudo_labels)
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.view(-1, 1, 1)
        masked_loss = per_pixel_loss * valid_mask
        denom = valid_mask.sum().clamp_min(1.0)
        return masked_loss.sum() / denom


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
        # remove cls token and reshape patch tokens back into 2d
        x = x[:, 1:, :]
        b, n, c = x.shape
        hw = int(n ** 0.5)
        x = x.transpose(1, 2).reshape(b, c, hw, hw)
        return self.decoder(x)


class ViTSegmentation(nn.Module):
    def __init__(self, num_classes=3, img_size=224, freeze_backbone=True, use_pretrained=True):
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


def create_unet_tl():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )

    if config.freeze_unet_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    return model


@torch.no_grad()
def compute_metrics(logits, target, num_classes):
    pred = logits.argmax(dim=1)

    pixel_acc = (pred == target).float().mean().item()

    dice_scores = []
    iou_scores = []

    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        inter = (pred_cls * target_cls).sum().item()
        pred_sum = pred_cls.sum().item()
        target_sum = target_cls.sum().item()
        union = pred_sum + target_sum - inter

        dice = (2.0 * inter + 1e-6) / (pred_sum + target_sum + 1e-6)
        iou = (inter + 1e-6) / (union + 1e-6)

        dice_scores.append(dice)
        iou_scores.append(iou)

    return {
        "dice": float(sum(dice_scores) / len(dice_scores)),
        "iou": float(sum(iou_scores) / len(iou_scores)),
        "pixel_accuracy": float(pixel_acc),
    }


@torch.no_grad()
def evaluate_unet(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_batches = 0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")

        logits = model(images)
        loss = criterion(logits, masks)
        metrics = compute_metrics(logits, masks, NUM_CLASSES)

        total_loss += loss.item()
        total_dice += metrics["dice"]
        total_iou += metrics["iou"]
        total_acc += metrics["pixel_accuracy"]
        total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "dice": total_dice / max(total_batches, 1),
        "iou": total_iou / max(total_batches, 1),
        "pixel_accuracy": total_acc / max(total_batches, 1),
    }


@torch.no_grad()
def evaluate_vit(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_batches = 0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")

        vit_images = F.interpolate(
            images,
            size=(config.vit_image_size, config.vit_image_size),
            mode="bilinear",
            align_corners=False,
        )

        vit_masks = F.interpolate(
            masks.unsqueeze(1).float(),
            size=(config.vit_image_size, config.vit_image_size),
            mode="nearest",
        ).squeeze(1).long()

        logits = model(vit_images)
        loss = criterion(logits, vit_masks)
        metrics = compute_metrics(logits, vit_masks, NUM_CLASSES)

        total_loss += loss.item()
        total_dice += metrics["dice"]
        total_iou += metrics["iou"]
        total_acc += metrics["pixel_accuracy"]
        total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "dice": total_dice / max(total_batches, 1),
        "iou": total_iou / max(total_batches, 1),
        "pixel_accuracy": total_acc / max(total_batches, 1),
    }


@torch.no_grad()
def evaluate_ensemble(unet_model, vit_model, dataloader, device):
    unet_model.eval()
    vit_model.eval()

    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_batches = 0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")

        unet_images = F.interpolate(
            images,
            size=(config.unet_image_size, config.unet_image_size),
            mode="bilinear",
            align_corners=False,
        )

        vit_images = F.interpolate(
            images,
            size=(config.vit_image_size, config.vit_image_size),
            mode="bilinear",
            align_corners=False,
        )

        unet_logits = unet_model(unet_images)
        vit_logits = vit_model(vit_images)

        # bring vit output to unet size before averaging
        vit_logits = F.interpolate(
            vit_logits,
            size=(config.unet_image_size, config.unet_image_size),
            mode="bilinear",
            align_corners=False,
        )

        ensemble_logits = (unet_logits + vit_logits) / 2.0
        metrics = compute_metrics(ensemble_logits, masks, NUM_CLASSES)

        total_dice += metrics["dice"]
        total_iou += metrics["iou"]
        total_acc += metrics["pixel_accuracy"]
        total_batches += 1

    return {
        "dice": total_dice / max(total_batches, 1),
        "iou": total_iou / max(total_batches, 1),
        "pixel_accuracy": total_acc / max(total_batches, 1),
    }


class CrossTeachingTrainer:
    def __init__(self):
        self.unet = create_unet_tl().to(config.device)

        self.vit = ViTSegmentation(
            num_classes=NUM_CLASSES,
            img_size=config.vit_image_size,
            freeze_backbone=config.freeze_vit_backbone,
            use_pretrained=True,
        ).to(config.device)

        self.supervised_loss = CombinedSegmentationLoss(num_classes=NUM_CLASSES)
        self.consistency_loss = MaskedCrossEntropyLoss()

        self.unet_optimizer = optim.Adam(
            [p for p in self.unet.parameters() if p.requires_grad],
            lr=config.learning_rate,
        )

        self.vit_optimizer = optim.Adam(
            [p for p in self.vit.parameters() if p.requires_grad],
            lr=config.learning_rate,
        )

    @staticmethod
    def get_confidence_and_labels(logits):
        probs = torch.softmax(logits, dim=1)
        pixel_confidence, labels = probs.max(dim=1)
        image_confidence = pixel_confidence.mean(dim=(1, 2))
        return image_confidence, labels

    def train_labeled_step(self, images, masks):
        images = images.to(config.device, non_blocking=config.device.type == "cuda")
        masks = masks.to(config.device, non_blocking=config.device.type == "cuda")

        unet_images = F.interpolate(
            images,
            size=(config.unet_image_size, config.unet_image_size),
            mode="bilinear",
            align_corners=False,
        )

        vit_images = F.interpolate(
            images,
            size=(config.vit_image_size, config.vit_image_size),
            mode="bilinear",
            align_corners=False,
        )

        vit_masks = F.interpolate(
            masks.unsqueeze(1).float(),
            size=(config.vit_image_size, config.vit_image_size),
            mode="nearest",
        ).squeeze(1).long()

        self.unet_optimizer.zero_grad()
        unet_logits = self.unet(unet_images)
        unet_loss = self.supervised_loss(unet_logits, masks)
        unet_loss.backward()
        self.unet_optimizer.step()

        self.vit_optimizer.zero_grad()
        vit_logits = self.vit(vit_images)
        vit_loss = self.supervised_loss(vit_logits, vit_masks)
        vit_loss.backward()
        self.vit_optimizer.step()

        return {
            "unet_supervised_loss": unet_loss.item(),
            "vit_supervised_loss": vit_loss.item(),
        }

    def train_unlabeled_step(self, images):
        images = images.to(config.device, non_blocking=config.device.type == "cuda")

        unet_images = F.interpolate(
            images,
            size=(config.unet_image_size, config.unet_image_size),
            mode="bilinear",
            align_corners=False,
        )

        vit_images = F.interpolate(
            images,
            size=(config.vit_image_size, config.vit_image_size),
            mode="bilinear",
            align_corners=False,
        )

        with torch.no_grad():
            unet_teacher_logits = self.unet(unet_images)
            vit_teacher_logits = self.vit(vit_images)

            # vit teaches unet, so resize vit output up to unet size
            vit_teacher_up = F.interpolate(
                vit_teacher_logits,
                size=(config.unet_image_size, config.unet_image_size),
                mode="bilinear",
                align_corners=False,
            )
            vit_confidence, vit_labels = self.get_confidence_and_labels(vit_teacher_up)
            vit_valid = (vit_confidence >= config.confidence_threshold).float()

            # unet teaches vit, so resize unet output down to vit size
            unet_teacher_down = F.interpolate(
                unet_teacher_logits,
                size=(config.vit_image_size, config.vit_image_size),
                mode="bilinear",
                align_corners=False,
            )
            unet_confidence, unet_labels = self.get_confidence_and_labels(unet_teacher_down)
            unet_valid = (unet_confidence >= config.confidence_threshold).float()

        self.unet_optimizer.zero_grad()
        unet_student_logits = self.unet(unet_images)
        unet_consistency_loss = self.consistency_loss(unet_student_logits, vit_labels, vit_valid)
        (config.consistency_weight * unet_consistency_loss).backward()
        self.unet_optimizer.step()

        self.vit_optimizer.zero_grad()
        vit_student_logits = self.vit(vit_images)
        vit_consistency_loss = self.consistency_loss(vit_student_logits, unet_labels, unet_valid)
        (config.consistency_weight * vit_consistency_loss).backward()
        self.vit_optimizer.step()

        return {
            "unet_consistency_loss": unet_consistency_loss.item(),
            "vit_consistency_loss": vit_consistency_loss.item(),
            "vit_confident_image_ratio": vit_valid.mean().item(),
            "unet_confident_image_ratio": unet_valid.mean().item(),
        }

    def train_epoch(self, labeled_loader, unlabeled_loader):
        self.unet.train()
        self.vit.train()

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        # Keep one supervised pass per epoch so labeled exposure matches the baselines.
        num_steps = len(labeled_loader)

        totals = {
            "unet_supervised_loss": 0.0,
            "vit_supervised_loss": 0.0,
            "unet_consistency_loss": 0.0,
            "vit_consistency_loss": 0.0,
            "vit_confident_image_ratio": 0.0,
            "unet_confident_image_ratio": 0.0,
        }

        for _ in range(num_steps):
            try:
                images_l, masks_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                images_l, masks_l = next(labeled_iter)

            labeled_stats = self.train_labeled_step(images_l, masks_l)

            try:
                images_u = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                images_u = next(unlabeled_iter)

            unlabeled_stats = self.train_unlabeled_step(images_u)

            for key, value in {**labeled_stats, **unlabeled_stats}.items():
                totals[key] += value

        for key in totals:
            totals[key] /= max(num_steps, 1)

        return totals


def main():
    set_seed()
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    unlabeled_fraction = 1.0 - config.labeled_fraction
    print(f"[Cross-Teaching] using device: {describe_device(config.device)}")

    labeled_loader, unlabeled_loader, val_loader, test_loader = get_oxford_pet_datasets_for_cross_teaching(
        unet_img_size=config.unet_image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        labeled_fraction=config.labeled_fraction,
        val_fraction=config.val_fraction,
    )

    trainer = CrossTeachingTrainer()

    best_val_dice = -1.0
    history = []

    for epoch in range(config.epochs):
        train_stats = trainer.train_epoch(labeled_loader, unlabeled_loader)

        unet_val_metrics = evaluate_unet(
            trainer.unet,
            val_loader,
            trainer.supervised_loss,
            config.device,
        )

        vit_val_metrics = evaluate_vit(
            trainer.vit,
            val_loader,
            trainer.supervised_loss,
            config.device,
        )

        ensemble_val_metrics = evaluate_ensemble(
            trainer.unet,
            trainer.vit,
            val_loader,
            config.device,
        )

        row = {
            "epoch": epoch + 1,
            "unet_supervised_loss": train_stats["unet_supervised_loss"],
            "vit_supervised_loss": train_stats["vit_supervised_loss"],
            "unet_consistency_loss": train_stats["unet_consistency_loss"],
            "vit_consistency_loss": train_stats["vit_consistency_loss"],
            "unet_confident_image_ratio": train_stats["unet_confident_image_ratio"],
            "vit_confident_image_ratio": train_stats["vit_confident_image_ratio"],
            "unet_val_loss": unet_val_metrics["loss"],
            "unet_val_dice": unet_val_metrics["dice"],
            "unet_val_iou": unet_val_metrics["iou"],
            "vit_val_loss": vit_val_metrics["loss"],
            "vit_val_dice": vit_val_metrics["dice"],
            "vit_val_iou": vit_val_metrics["iou"],
            "ensemble_val_dice": ensemble_val_metrics["dice"],
            "ensemble_val_iou": ensemble_val_metrics["iou"],
            "ensemble_val_pixel_accuracy": ensemble_val_metrics["pixel_accuracy"],
        }
        history.append(row)

        print(
            f"[Cross-Teaching] epoch {epoch + 1}/{config.epochs} "
            f"ensemble_val_dice={ensemble_val_metrics['dice']:.4f} "
            f"unet_val_dice={unet_val_metrics['dice']:.4f} "
            f"vit_val_dice={vit_val_metrics['dice']:.4f}"
        )

        if ensemble_val_metrics["dice"] > best_val_dice:
            best_val_dice = ensemble_val_metrics["dice"]

            torch.save(
                {
                    "model_state_dict": trainer.unet.state_dict(),
                    "best_val_dice": best_val_dice,
                },
                UNET_SAVE_PATH,
            )

            torch.save(
                {
                    "model_state_dict": trainer.vit.state_dict(),
                    "best_val_dice": best_val_dice,
                },
                VIT_SAVE_PATH,
            )

    unet_checkpoint = torch.load(UNET_SAVE_PATH, map_location=config.device)
    vit_checkpoint = torch.load(VIT_SAVE_PATH, map_location=config.device)

    trainer.unet.load_state_dict(unet_checkpoint["model_state_dict"])
    trainer.vit.load_state_dict(vit_checkpoint["model_state_dict"])

    unet_test_metrics = evaluate_unet(
        trainer.unet,
        test_loader,
        trainer.supervised_loss,
        config.device,
    )

    vit_test_metrics = evaluate_vit(
        trainer.vit,
        test_loader,
        trainer.supervised_loss,
        config.device,
    )

    ensemble_test_metrics = evaluate_ensemble(
        trainer.unet,
        trainer.vit,
        test_loader,
        config.device,
    )

    save_json(
        METRICS_SAVE_PATH,
        {
            "model": "Cross-Teaching U-Net + ViT",
            "setting": "semi-supervised",
            "labeled_fraction": config.labeled_fraction,
            "unlabeled_fraction": unlabeled_fraction,
            "confidence_threshold": config.confidence_threshold,
            "consistency_weight": config.consistency_weight,
            "best_val_ensemble_dice": best_val_dice,
            "history": history,
            "test_metrics": {
                "unet": unet_test_metrics,
                "vit": vit_test_metrics,
                "ensemble": ensemble_test_metrics,
            },
        },
    )

    print(f"Saved best U-Net cross-teaching model to {UNET_SAVE_PATH}")
    print(f"Saved best ViT cross-teaching model to {VIT_SAVE_PATH}")
    print(f"Saved metrics to {METRICS_SAVE_PATH}")


if __name__ == "__main__":
    main()
