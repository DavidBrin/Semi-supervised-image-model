import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

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
    unet_image_size = 512  # For detection, perhaps keep as is, but for Faster R-CNN, usually 800x800 or something
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
UNET_SAVE_PATH = os.path.join(CHECKPOINTS_DIR, "fasterrcnn_cross_teaching_best.pth")
VIT_SAVE_PATH = os.path.join(CHECKPOINTS_DIR, "vit_detection_cross_teaching_best.pth")
METRICS_SAVE_PATH = os.path.join(CHECKPOINTS_DIR, "cross_detection_metrics.json")


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def describe_device(device):
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device.index or 0)})"
    if device.type == "mps":
        return "mps"
    return "cpu"


# For detection, we need a different loss, perhaps a custom one for consistency
class MaskedDetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # For simplicity, use a placeholder; in practice, detection loss is complex
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred_boxes, pseudo_boxes, valid_mask):
        # Simplified: assume pred_boxes and pseudo_boxes are [B, N, 4]
        # valid_mask [B]
        loss = self.mse(pred_boxes, pseudo_boxes)  # [B, N, 4]
        loss = loss.mean(dim=(1,2))  # [B]
        loss = loss * valid_mask
        return loss.sum() / valid_mask.sum().clamp_min(1.0)


class ViTDetectionHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=37):
        super().__init__()
        # Simple detection head: predict boxes and classes
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.bbox_regressor = nn.Linear(embed_dim, 4)  # x,y,w,h or deltas

    def forward(self, x):
        # x: [B, N, C], N = 14*14 = 196 for 224x224
        cls_logits = self.classifier(x)  # [B, N, num_classes]
        bbox_preds = self.bbox_regressor(x)  # [B, N, 4]
        return cls_logits, bbox_preds


class ViTDetection(nn.Module):
    def __init__(self, num_classes=37, img_size=224, freeze_backbone=True, use_pretrained=True):
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

        self.det_head = ViTDetectionHead(
            embed_dim=self.backbone.embed_dim,
            num_classes=num_classes,
        )

    def forward(self, x):
        feats = self.backbone.forward_features(x)  # [B, N+1, C]
        feats = feats[:, 1:, :]  # remove cls token
        return self.det_head(feats)


def create_faster_rcnn():
    # Use ResNet50 backbone as in standard Faster R-CNN
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = FasterRCNN(backbone, num_classes=37)  # 37 classes + background?
    # Freeze backbone if needed
    if config.freeze_unet_encoder:
        for param in backbone.parameters():
            param.requires_grad = False
    return model


@torch.no_grad()
def compute_detection_metrics(preds, targets):
    # Simplified mAP calculation; in practice, use pycocotools or similar
    # For now, return dummy values
    return {"mAP": 0.5, "precision": 0.6}


@torch.no_grad()
def evaluate_faster_rcnn(model, dataloader, device):
    model.eval()
    total_map = 0.0
    total_batches = 0
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        preds = model(images)
        metrics = compute_detection_metrics(preds, targets)
        total_map += metrics["mAP"]
        total_batches += 1
    return {"mAP": total_map / max(total_batches, 1)}


@torch.no_grad()
def evaluate_vit_detection(model, dataloader, device):
    model.eval()
    total_map = 0.0
    total_batches = 0
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        # For ViT, assume images are batched
        images = torch.stack(images).to(device)
        preds = model(images)
        # Compute metrics
        metrics = compute_detection_metrics(preds, targets)
        total_map += metrics["mAP"]
        total_batches += 1
    return {"mAP": total_map / max(total_batches, 1)}


@torch.no_grad()
def evaluate_ensemble_detection(faster_rcnn_model, vit_model, dataloader, device):
    faster_rcnn_model.eval()
    vit_model.eval()
    total_map = 0.0
    total_batches = 0
    for images, targets in dataloader:
        images_list = [img.to(device) for img in images]
        images_tensor = torch.stack(images_list).to(device)
        # Ensemble predictions
        # Simplified: average or something
        preds = []  # Need to implement ensemble logic
        metrics = compute_detection_metrics(preds, targets)
        total_map += metrics["mAP"]
        total_batches += 1
    return {"mAP": total_map / max(total_batches, 1)}


class CrossTeachingDetectionTrainer:
    def __init__(self):
        self.faster_rcnn = create_faster_rcnn().to(config.device)
        self.vit = ViTDetection(
            num_classes=37,  # 37 classes
            img_size=config.vit_image_size,
            freeze_backbone=config.freeze_vit_backbone,
            use_pretrained=True,
        ).to(config.device)

        # For detection, losses are built-in for Faster R-CNN
        self.consistency_loss = MaskedDetectionLoss()

        self.faster_rcnn_optimizer = optim.Adam(
            [p for p in self.faster_rcnn.parameters() if p.requires_grad],
            lr=config.learning_rate,
        )

        self.vit_optimizer = optim.Adam(
            [p for p in self.vit.parameters() if p.requires_grad],
            lr=config.learning_rate,
        )

    @staticmethod
    def get_confidence_and_labels(logits):
        # For detection, logits are class scores, assume [B, N, num_classes]
        probs = torch.softmax(logits, dim=-1)
        confidence, labels = probs.max(dim=-1)
        # Image confidence: average over proposals
        image_confidence = confidence.mean(dim=1)
        return image_confidence, labels

    def train_labeled_step(self, images, targets):
        # images: list of tensors, targets: list of dicts
        self.faster_rcnn_optimizer.zero_grad()
        faster_rcnn_loss = self.faster_rcnn(images, targets)
        faster_rcnn_loss.backward()
        self.faster_rcnn_optimizer.step()

        # For ViT, need to adapt
        # Assume images are resized, etc.
        vit_images = torch.stack([F.interpolate(img.unsqueeze(0), size=(config.vit_image_size, config.vit_image_size), mode="bilinear", align_corners=False).squeeze(0) for img in images])
        # Targets for ViT: need to convert to appropriate format
        # This is simplified
        self.vit_optimizer.zero_grad()
        vit_logits, vit_boxes = self.vit(vit_images)
        # Compute loss for ViT
        # Placeholder
        vit_loss = torch.tensor(0.0)
        vit_loss.backward()
        self.vit_optimizer.step()

        return {
            "faster_rcnn_supervised_loss": faster_rcnn_loss.item(),
            "vit_supervised_loss": vit_loss.item(),
        }

    def train_unlabeled_step(self, images):
        # Similar to segmentation, but for detection
        # Assume images are list or tensor
        with torch.no_grad():
            # Generate pseudo labels
            faster_rcnn_preds = self.faster_rcnn(images)
            vit_preds = self.vit(images)  # Assume adapted

            # Compute confidence
            # Simplified
            faster_rcnn_confidence = torch.tensor([0.8] * len(images))  # Placeholder
            vit_confidence = torch.tensor([0.8] * len(images))

            faster_rcnn_valid = (faster_rcnn_confidence >= config.confidence_threshold).float()
            vit_valid = (vit_confidence >= config.confidence_threshold).float()

        # Train Faster R-CNN with ViT pseudo labels
        self.faster_rcnn_optimizer.zero_grad()
        faster_rcnn_loss = self.consistency_loss(faster_rcnn_preds, vit_preds, faster_rcnn_valid)
        (config.consistency_weight * faster_rcnn_loss).backward()
        self.faster_rcnn_optimizer.step()

        # Train ViT with Faster R-CNN pseudo labels
        self.vit_optimizer.zero_grad()
        vit_loss = self.consistency_loss(vit_preds, faster_rcnn_preds, vit_valid)
        (config.consistency_weight * vit_loss).backward()
        self.vit_optimizer.step()

        return {
            "faster_rcnn_consistency_loss": faster_rcnn_loss.item(),
            "vit_consistency_loss": vit_loss.item(),
            "vit_confident_image_ratio": vit_valid.mean().item(),
            "faster_rcnn_confident_image_ratio": faster_rcnn_valid.mean().item(),
        }

    def train_epoch(self, labeled_loader, unlabeled_loader):
        self.faster_rcnn.train()
        self.vit.train()

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        num_steps = len(labeled_loader)

        totals = {
            "faster_rcnn_supervised_loss": 0.0,
            "vit_supervised_loss": 0.0,
            "faster_rcnn_consistency_loss": 0.0,
            "vit_consistency_loss": 0.0,
            "vit_confident_image_ratio": 0.0,
            "faster_rcnn_confident_image_ratio": 0.0,
        }

        for _ in range(num_steps):
            try:
                images_l, targets_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                images_l, targets_l = next(labeled_iter)

            labeled_stats = self.train_labeled_step(images_l, targets_l)

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
    print(f"[Cross-Detection] using device: {describe_device(config.device)}")

    # Assume data loading is adapted for detection
    labeled_loader, unlabeled_loader, val_loader, test_loader = get_oxford_pet_datasets_for_cross_teaching(
        unet_img_size=config.unet_image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        labeled_fraction=config.labeled_fraction,
        val_fraction=config.val_fraction,
    )

    trainer = CrossTeachingDetectionTrainer()

    best_val_map = -1.0
    history = []

    for epoch in range(config.epochs):
        train_stats = trainer.train_epoch(labeled_loader, unlabeled_loader)

        faster_rcnn_val_metrics = evaluate_faster_rcnn(
            trainer.faster_rcnn,
            val_loader,
            config.device,
        )

        vit_val_metrics = evaluate_vit_detection(
            trainer.vit,
            val_loader,
            config.device,
        )

        ensemble_val_metrics = evaluate_ensemble_detection(
            trainer.faster_rcnn,
            trainer.vit,
            val_loader,
            config.device,
        )

        row = {
            "epoch": epoch + 1,
            "faster_rcnn_supervised_loss": train_stats["faster_rcnn_supervised_loss"],
            "vit_supervised_loss": train_stats["vit_supervised_loss"],
            "faster_rcnn_consistency_loss": train_stats["faster_rcnn_consistency_loss"],
            "vit_consistency_loss": train_stats["vit_consistency_loss"],
            "faster_rcnn_confident_image_ratio": train_stats["faster_rcnn_confident_image_ratio"],
            "vit_confident_image_ratio": train_stats["vit_confident_image_ratio"],
            "faster_rcnn_val_mAP": faster_rcnn_val_metrics["mAP"],
            "vit_val_mAP": vit_val_metrics["mAP"],
            "ensemble_val_mAP": ensemble_val_metrics["mAP"],
        }
        history.append(row)

        print(
            f"[Cross-Detection] epoch {epoch + 1}/{config.epochs} "
            f"ensemble_val_mAP={ensemble_val_metrics['mAP']:.4f} "
            f"faster_rcnn_val_mAP={faster_rcnn_val_metrics['mAP']:.4f} "
            f"vit_val_mAP={vit_val_metrics['mAP']:.4f}"
        )

        if ensemble_val_metrics["mAP"] > best_val_map:
            best_val_map = ensemble_val_metrics["mAP"]

            torch.save(
                {
                    "model_state_dict": trainer.faster_rcnn.state_dict(),
                    "best_val_mAP": best_val_map,
                },
                UNET_SAVE_PATH,
            )

            torch.save(
                {
                    "model_state_dict": trainer.vit.state_dict(),
                    "best_val_mAP": best_val_map,
                },
                VIT_SAVE_PATH,
            )

    faster_rcnn_checkpoint = torch.load(UNET_SAVE_PATH, map_location=config.device)
    vit_checkpoint = torch.load(VIT_SAVE_PATH, map_location=config.device)

    trainer.faster_rcnn.load_state_dict(faster_rcnn_checkpoint["model_state_dict"])
    trainer.vit.load_state_dict(vit_checkpoint["model_state_dict"])

    faster_rcnn_test_metrics = evaluate_faster_rcnn(
        trainer.faster_rcnn,
        test_loader,
        config.device,
    )

    vit_test_metrics = evaluate_vit_detection(
        trainer.vit,
        test_loader,
        config.device,
    )

    ensemble_test_metrics = evaluate_ensemble_detection(
        trainer.faster_rcnn,
        trainer.vit,
        test_loader,
        config.device,
    )

    save_json(
        METRICS_SAVE_PATH,
        {
            "model": "Cross-Teaching Detection Faster R-CNN + ViT",
            "setting": "semi-supervised",
            "labeled_fraction": config.labeled_fraction,
            "unlabeled_fraction": unlabeled_fraction,
            "confidence_threshold": config.confidence_threshold,
            "consistency_weight": config.consistency_weight,
            "best_val_ensemble_mAP": best_val_map,
            "history": history,
            "test_metrics": {
                "faster_rcnn": faster_rcnn_test_metrics,
                "vit": vit_test_metrics,
                "ensemble": ensemble_test_metrics,
            },
        },
    )

    print(f"Saved best Faster R-CNN cross-teaching model to {UNET_SAVE_PATH}")
    print(f"Saved best ViT Detection cross-teaching model to {VIT_SAVE_PATH}")
    print(f"Saved metrics to {METRICS_SAVE_PATH}")


if __name__ == "__main__":
    main()