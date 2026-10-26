#!/usr/bin/env python3
"""Q4: reproducible ImageNet inference and Grad-CAM, without model training.

The GradCAM API usage adapts the assignment's usage guide and the upstream
example: https://github.com/jacobgil/pytorch-grad-cam#usage-examples .
Transforms, ImageNet labels, and pretrained parameters come from torchvision.
Image sources and descriptions, recorded before inference, are in images/sources.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import urllib.request

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ee4745_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ee4745_xdg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models import (
    ResNet18_Weights, MobileNet_V3_Small_Weights,
    resnet18, mobilenet_v3_small,
)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def judgment(image_id: str, label: str) -> tuple[str, str]:
    """Qualitative image/label judgments; exact breeds need independent evidence."""
    if image_id == "dog":
        if label in {"Samoyed", "Pomeranian", "Great Pyrenees", "kuvasz", "Eskimo dog",
                     "Japanese spaniel", "Maltese dog", "white wolf", "Arctic fox"}:
            if label in {"white wolf", "Arctic fox"}:
                return "no", "The main subject is a domestic dog, not the predicted wild animal."
            return "partly", ("The dog category and visible appearance are plausible, but "
                              f"the exact {label} breed is not independently established.")
    if image_id == "cat":
        if label in {"tabby", "tiger cat"}:
            return "yes", "The visible domestic cat has a striped tabby coat; this does not verify a breed."
        if label in {"Egyptian cat", "Persian cat", "Siamese cat"}:
            return "partly", "A domestic cat is present, but the predicted breed is not independently established."
    if image_id == "bus":
        if label == "minibus":
            return "partly", "It is a bus, but the image alone does not establish the more specific minibus subtype."
        if label == "trolleybus":
            return "partly", "A city bus is present, but trolley poles or overhead electrical connections are not visible."
        if label == "school bus":
            return "partly", "The bus category matches, but this appears to be a city bus rather than a school bus."
    return "no", "The top label does not match the recorded main object; inspect the saved image for context."


def prepare_images(root: Path) -> list[dict]:
    provenance_file = root / "images" / "sources.json"
    provenance = json.loads(provenance_file.read_text())
    assert provenance["recorded_before_model_predictions"]
    records = provenance["images"]
    # Descriptions exist before model construction or any prediction. Downloading
    # only restores an absent bundled image; hashes prevent silently changing it.
    for record in records:
        path = root / "images" / record["file"]
        if not path.exists():
            with urllib.request.urlopen(record["source_url"], timeout=60) as response:
                path.write_bytes(response.read())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Image hash mismatch: {path}")
        with Image.open(path) as im:
            assert im.size == tuple(record["size_pixels"])
            im.verify()
    return records


def image_overview(root: Path, records: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), constrained_layout=True)
    for ax, record in zip(axes, records):
        with Image.open(root / "images" / record["file"]) as im:
            ax.imshow(im.convert("RGB"))
        title = f"{record['id'].capitalize()} ({record['difficulty']})"
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(root / "figures" / "q4_images.png", dpi=180)
    plt.close(fig)



def add_interpretation(result: dict) -> None:
    """Record qualitative observations after viewing these saved figures."""
    rows = {(row["model"], row["image_id"]): row for row in result["predictions"]}
    def pct(model, image):
        return rows[(model, image)]["top3"][0]["percent"]
    result["comparison"] = (
        "Both models rank Samoyed first for the dog and minibus first for the hard street scene, "
        f"with ResNet-18 assigning higher top probabilities ({pct('ResNet-18', 'dog'):.2f}% versus "
        f"{pct('MobileNetV3-Small', 'dog'):.2f}% for the dog; {pct('ResNet-18', 'bus'):.2f}% versus "
        f"{pct('MobileNetV3-Small', 'bus'):.2f}% for the bus). "
        "They disagree on the cat: ResNet-18 ranks Egyptian cat first, narrowly above tiger cat, "
        "whereas MobileNetV3-Small ranks tiger cat first. "
        "The visible stripes support a domestic-cat/coat-pattern interpretation, but exact breeds "
        "are not independently established and the bus subtype remains uncertain. "
        "A higher softmax value does not guarantee correctness, and these three images do not "
        "establish a general model ranking."
    )
    descriptions = {
        "dog": (
            "The warmest region is around the dog's face, ears, and upper neck, with much cooler "
            "grass, foliage, and lower body. This aligns with salient dog features and helps interpret "
            "the animal prediction. It does not independently establish the Samoyed breed."
        ),
        "bus": (
            "The strongest red region covers the bus's front/window area, with yellow support "
            "extending along its upper side windows; the pavement and lower pedestrians are relatively blue. "
            "The coarse map also overlaps part of a pedestrian, so it suggests vehicle-related cues "
            "without isolating the bus precisely. It helps explain the broad bus prediction but does not "
            "verify the specific minibus subtype."
        ),
    }
    for record in result["grad_cam"]:
        record["interpretation"] = descriptions[record["image_id"]]
    result["cam_limitations"] = (
        "Grad-CAM is a coarse visual clue, not a segmentation mask or proof of causation. "
        "Each heatmap is independently normalized, so its colors cannot compare confidence across images."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(os.environ.get("EE4745_CACHE", "/workspace/ee4745_cache")))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    for dirname in ("images", "figures", "results"):
        (root / dirname).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.hub.set_dir(str(args.cache_dir / "torch_hub"))
    records = prepare_images(root)
    image_overview(root, records)

    configurations = [
        ("ResNet-18", resnet18, ResNet18_Weights.IMAGENET1K_V1),
        ("MobileNetV3-Small", mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    ]
    result = {
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "grad-cam": importlib.metadata.version("grad-cam"),
            "numpy": np.__version__, "Pillow": importlib.metadata.version("Pillow"),
        },
        "device": "CPU", "torch_threads": args.threads,
        "images": records, "models": {}, "predictions": [], "grad_cam": [],
        "procedure": {
            "trained_or_finetuned": False,
            "mode": "eval", "input_color": "RGB", "probability": "softmax(dim=1)",
            "transforms": "Each weight object's transforms() used without modification",
            "cam_layer": "ResNet-18 layer4[-1]",
            "cam_target": "Each image's own ResNet-18 top-predicted class logit",
            "cam_context": "Outside torch.no_grad() and torch.inference_mode()",
        },
    }
    for model_name, constructor, weights in configurations:
        print(f"Loading {model_name}: {weights}", flush=True)
        model = constructor(weights=weights).eval()
        transform = weights.transforms()
        categories = weights.meta["categories"]
        result["models"][model_name] = {
            "weights": str(weights), "weights_url": weights.url,
            "transforms": repr(transform),
            "mean": list(transform.mean), "std": list(transform.std),
            "input_size": list(transform.crop_size),
            "parameter_count": sum(p.numel() for p in model.parameters()),
        }
        inputs = {}
        rows = {}
        for record in records:
            with Image.open(root / "images" / record["file"]) as im:
                input_tensor = transform(im.convert("RGB")).unsqueeze(0)
            assert input_tensor.shape == (1, 3, 224, 224)
            assert input_tensor.dtype == torch.float32
            inputs[record["id"]] = input_tensor
            with torch.no_grad():
                probabilities = torch.softmax(model(input_tensor), dim=1)
                values, indices = probabilities.topk(3, dim=1)
            top = [{"class_index": int(k), "label": categories[int(k)],
                    "probability": float(p), "percent": float(p * 100)}
                   for p, k in zip(values[0].tolist(), indices[0].tolist())]
            correct, reason = judgment(record["id"], top[0]["label"])
            row = {"image_id": record["id"], "model": model_name,
                   "weights": str(weights), "top3": top,
                   "correctness": correct, "reason": reason}
            result["predictions"].append(row)
            rows[record["id"]] = row
            print(json.dumps(row), flush=True)

        if model_name == "ResNet-18":
            for image_id in ("dog", "bus"):
                input_tensor = inputs[image_id]
                top1 = rows[image_id]["top3"][0]
                target = ClassifierOutputTarget(top1["class_index"])
                # Package computes the gradients; do not wrap this in no_grad.
                with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
                    heatmap = cam(input_tensor=input_tensor, targets=[target])[0]
                assert heatmap.shape == (224, 224)
                assert np.isfinite(heatmap).all()
                mean = torch.tensor(transform.mean).view(3, 1, 1)
                std = torch.tensor(transform.std).view(3, 1, 1)
                crop = (input_tensor[0] * std + mean).permute(1, 2, 0).numpy()
                crop = np.clip(crop, 0.0, 1.0).astype(np.float32)
                overlay = show_cam_on_image(crop, heatmap, use_rgb=True)
                fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.5), constrained_layout=True)
                axes[0].imshow(crop)
                axes[0].set_title("Exact model input crop")
                axes[1].imshow(overlay)
                axes[1].set_title("Grad-CAM overlay")
                for ax in axes:
                    ax.axis("off")
                fig.suptitle(f"ResNet-18 | {image_id} | {top1['label']} "
                             f"({top1['percent']:.2f}%)", fontsize=12)
                figure_name = f"cam_{image_id}.png"
                fig.savefig(root / "figures" / figure_name, dpi=180)
                plt.close(fig)
                result["grad_cam"].append({
                    "image_id": image_id, "target": top1,
                    "figure": f"figures/{figure_name}",
                    "heatmap_min": float(heatmap.min()),
                    "heatmap_max": float(heatmap.max()),
                    "heatmap_mean": float(heatmap.mean()),
                    "has_spatial_contrast": bool(heatmap.max() > heatmap.min()),
                })
                print(f"Saved {figure_name}; heatmap range {heatmap.min():.4f}–{heatmap.max():.4f}", flush=True)
        del model
    add_interpretation(result)
    out = root / "results" / "pretrained_results.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
