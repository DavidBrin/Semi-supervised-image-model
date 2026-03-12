import json
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_single_training_curve(metrics_json, model_name="Model"):
    history = metrics_json["history"]

    epochs = [row["epoch"] for row in history]

    train_loss = [row.get("train_loss", None) for row in history]
    val_loss = [row.get("val_loss", row.get("unet_val_loss", None)) for row in history]
    val_dice = [row.get("val_dice", row.get("ensemble_val_dice", None)) for row in history]
    val_iou = [row.get("val_iou", row.get("ensemble_val_iou", None)) for row in history]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label="train")
    if val_loss[0] is not None:
        plt.plot(epochs, val_loss, label="val")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

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


def plot_model_comparison(unet_metrics, vit_metrics, cross_metrics):
    labels = ["Dice", "IoU", "Pixel Acc"]

    unet_vals = [
        unet_metrics["test_metrics"]["dice"],
        unet_metrics["test_metrics"]["iou"],
        unet_metrics["test_metrics"]["pixel_accuracy"],
    ]

    vit_vals = [
        vit_metrics["test_metrics"]["dice"],
        vit_metrics["test_metrics"]["iou"],
        vit_metrics["test_metrics"]["pixel_accuracy"],
    ]

    ensemble_vals = [
        cross_metrics["test_metrics"]["ensemble"]["dice"],
        cross_metrics["test_metrics"]["ensemble"]["iou"],
        cross_metrics["test_metrics"]["ensemble"]["pixel_accuracy"],
    ]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, unet_vals, width, label="U-Net")
    plt.bar(x, vit_vals, width, label="ViT")
    plt.bar(x + width, ensemble_vals, width, label="Cross-Teaching Ensemble")

    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Final Test Metrics Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confidence_ratios(cross_metrics):
    history = cross_metrics["history"]
    epochs = [row["epoch"] for row in history]

    unet_ratio = [row["unet_confident_pixel_ratio"] for row in history]
    vit_ratio = [row["vit_confident_pixel_ratio"] for row in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, unet_ratio, label="U-Net confident ratio")
    plt.plot(epochs, vit_ratio, label="ViT confident ratio")
    plt.xlabel("Epoch")
    plt.ylabel("Confident pixel ratio")
    plt.title("Cross-Teaching Confidence Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_final_summary(unet_metrics, vit_metrics, cross_metrics):
    print("Final Test Results")
    print("-" * 50)

    print("U-Net")
    print(f"Dice: {unet_metrics['test_metrics']['dice']:.4f}")
    print(f"IoU: {unet_metrics['test_metrics']['iou']:.4f}")
    print(f"Pixel Accuracy: {unet_metrics['test_metrics']['pixel_accuracy']:.4f}")
    print()

    print("ViT")
    print(f"Dice: {vit_metrics['test_metrics']['dice']:.4f}")
    print(f"IoU: {vit_metrics['test_metrics']['iou']:.4f}")
    print(f"Pixel Accuracy: {vit_metrics['test_metrics']['pixel_accuracy']:.4f}")
    print()

    print("Cross-Teaching Ensemble")
    print(f"Dice: {cross_metrics['test_metrics']['ensemble']['dice']:.4f}")
    print(f"IoU: {cross_metrics['test_metrics']['ensemble']['iou']:.4f}")
    print(f"Pixel Accuracy: {cross_metrics['test_metrics']['ensemble']['pixel_accuracy']:.4f}")