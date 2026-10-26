"""EE4745 HW2. Adapted dataset/Grad-CAM APIs: assignment pp. 5-6 and
https://github.com/jacobgil/pytorch-grad-cam#usage-examples .
Run: python solve_hw2.py --part all --output-dir hw2_solution
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import urllib.request

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ee4745_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision.datasets import FashionMNIST


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def mlp(width):
    return nn.Sequential(nn.Flatten(), nn.Linear(784, width), nn.ReLU(), nn.Linear(width, 10))


def cnn():
    return nn.Sequential(nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
                         nn.Conv2d(6, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                         nn.Flatten(), nn.Linear(432, 10))


def architecture_check():
    counts = {"MLP-32": sum(p.numel() for p in mlp(32).parameters()),
              "MLP-128": sum(p.numel() for p in mlp(128).parameters())}
    model, x = cnn(), torch.zeros(2, 1, 28, 28)
    shapes = [list(x.shape)]
    with torch.no_grad():
        for layer in model:
            x = layer(x)
            shapes.append(list(x.shape))
    counts["CNN"] = sum(p.numel() for p in model.parameters())
    assert counts == {"MLP-32": 25450, "MLP-128": 101770, "CNN": 5146}
    assert shapes == [[2, 1, 28, 28], [2, 6, 24, 24], [2, 6, 24, 24],
                      [2, 6, 12, 12], [2, 12, 12, 12], [2, 12, 12, 12],
                      [2, 12, 6, 6], [2, 432], [2, 10]]
    return {"parameter_counts": counts, "cnn_dummy_shapes": shapes}


def q1(root):
    x, w1, b1 = np.array([1., -2.]), np.eye(2), np.array([0., 1.])
    w2, b2 = np.array([[1., -1.], [-1., 1.]]), np.zeros(2)
    a = w1 @ x + b1
    h = np.maximum(a, 0.)
    z = w2 @ h + b2
    p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    dz = p - np.array([0., 1.])
    da = (w2.T @ dz) * (a > 0)
    manual = {"W1": np.outer(da, x), "b1": da, "W2": np.outer(dz, h), "b2": dz}
    params = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True)
              for k, v in dict(W1=w1, b1=b1, W2=w2, b2=b2).items()}
    tx = torch.tensor(x, dtype=torch.float64)
    logits = params["W2"] @ F.relu(params["W1"] @ tx + params["b1"]) + params["b2"]
    loss = F.cross_entropy(logits[None], torch.tensor([1]))
    loss.backward()
    differences = {k: float(np.abs(v - params[k].grad.numpy()).max()) for k, v in manual.items()}
    assert max(differences.values()) < 1e-12
    result = {"a": a.tolist(), "h": h.tolist(), "z": z.tolist(), "p": p.tolist(),
              "loss": loss.item(), "manual_gradients": {k: v.tolist() for k, v in manual.items()},
              "autograd_gradients": {k: v.grad.tolist() for k, v in params.items()},
              "max_absolute_difference": max(differences.values()), **architecture_check()}
    save_json(root / "results/gradient_results.json", result)
    print("Q1 float64 maximum gradient difference:", result["max_absolute_difference"], flush=True)


def load_data(data_dir):
    # Assignment page-6 split/preprocessing; official HTTPS mirror keeps MD5 checks.
    FashionMNIST.mirrors = ["https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"]
    full = FashionMNIST(str(data_dir), train=True, download=True)
    test = FashionMNIST(str(data_dir), train=False, download=True)
    order = np.random.default_rng(2026).permutation(len(full))
    def subset(source, indices):
        images = source.data[indices].unsqueeze(1).float() / 255.0
        labels = source.targets[indices].long()
        return TensorDataset(images, labels)
    return (subset(full, order[:10000]), subset(full, order[10000:12000]),
            subset(test, np.arange(len(test))))


def evaluate(model, loader):
    model.eval()
    total = correct = 0
    loss_sum = 0.
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    assert total == len(loader.dataset)
    return {"cross_entropy": loss_sum / total, "accuracy": correct / total, "examples": total}


def fashion(root, cache):
    train, val, test = load_data(cache / "data")
    assert tuple(map(len, [train, val, test])) == (10000, 2000, 10000)
    train_eval, val_eval, test_eval = [DataLoader(d, batch_size=128, shuffle=False,
                                                num_workers=0) for d in (train, val, test)]
    models, results, events = {}, {}, []
    for name, build in [("MLP-32", lambda: mlp(32)), ("MLP-128", lambda: mlp(128)), ("CNN", cnn)]:
        torch.manual_seed(2026)
        loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(2026), drop_last=False)
        model = build()  # Default FP32 initialization and biases; no added layers.
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0)
        history, best_loss, best_weights = [], float("inf"), None
        for epoch in range(1, 6):
            model.train()
            for images, labels in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(model(images), labels)
                loss.backward()
                optimizer.step()
            row = {"epoch": epoch, "train": evaluate(model, train_eval),
                   "validation": evaluate(model, val_eval)}
            history.append(row)
            if row["validation"]["cross_entropy"] < best_loss:
                best_loss, selected_epoch = row["validation"]["cross_entropy"], epoch
                best_weights = copy.deepcopy(model.state_dict())
            print(name, row, flush=True)
        model.load_state_dict(best_weights)
        torch.save(best_weights, cache / f"{name}.pt")
        models[name] = model
        results[name] = {"selected_epoch": selected_epoch, "history": history,
                         "selected_train": history[selected_epoch-1]["train"],
                         "selected_validation": history[selected_epoch-1]["validation"],
                         "parameter_count": sum(p.numel() for p in model.parameters())}
        events.append(["checkpoint_selected", name])
    # All runs/checkpoint selections are complete before the FIRST test iteration.
    for name, model in models.items():
        results[name]["test"] = evaluate(model, test_eval)
        events.append(["test_evaluated_once", name])
        print(name, "selected epoch", results[name]["selected_epoch"], results[name]["test"], flush=True)
    save_json(root / "results/fashion_results.json", {"models": results, "protocol_events": events})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for name, ax, color in [("MLP-32", axes[0], "C0"), ("MLP-128", axes[0], "C1"), ("CNN", axes[1], "C2")]:
        for split, style in [("train", "o-"), ("validation", "s--")]:
            ax.plot(range(1, 6), [r[split]["cross_entropy"] for r in results[name]["history"]],
                    style, color=color, label=f"{name} {split}")
    for ax in axes:
        ax.set(xlabel="Epoch", xticks=range(1, 6), ylabel="Mean cross-entropy")
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(root / "figures/fashion_curves.png", dpi=180)
    plt.close(fig)


SOURCES = [
    ("dog", "dog.jpg", "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
     "f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a",
     "Clear: a fluffy white pointed-ear dog on grass; exact breed is not independently known."),
    ("cat", "cat.png", "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/chelsea.png",
     "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb",
     "Clear: a close-up brown-orange striped domestic cat with green eyes and whiskers."),
    ("bus", "bus.jpg", "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg",
     "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63",
     "Hard: a blue-white city bus, partially occluded by pedestrians, amid buildings/signs/street clutter.")]


def q4(root, cache):
    from torchvision.models import (resnet18, mobilenet_v3_small, ResNet18_Weights,
                                    MobileNet_V3_Small_Weights)
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    records = [dict(id=i, file=f, source_url=u, sha256=h, description_before_predictions=d)
               for i, f, u, h, d in SOURCES]
    # Persist descriptions and URLs BEFORE constructing a model or predicting.
    save_json(root / "images/sources.json", {"recorded_before_model_predictions": True, "images": records})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for record, ax in zip(records, axes):
        path = root / "images" / record["file"]
        if not path.exists():
            with urllib.request.urlopen(record["source_url"], timeout=60) as response:
                path.write_bytes(response.read())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        with Image.open(path) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(record["id"])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(root / "figures/q4_images.png", dpi=180)
    plt.close(fig)
    torch.hub.set_dir(str(cache / "torch_hub"))
    results = []
    configurations = [("ResNet-18", resnet18, ResNet18_Weights.IMAGENET1K_V1),
                      ("MobileNetV3-Small", mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1)]
    for name, build, weights in configurations:
        model, transform = build(weights=weights).eval(), weights.transforms()
        for record in records:
            with Image.open(root / "images" / record["file"]) as im:
                tensor = transform(im.convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                probs, indices = model(tensor).softmax(dim=1).topk(3, dim=1)
            top3 = [{"index": int(i), "label": weights.meta["categories"][int(i)],
                     "probability": float(p)} for p, i in zip(probs[0], indices[0])]
            results.append({"image": record["id"], "model": name, "weights": str(weights), "top3": top3})
            print(results[-1], flush=True)
            if name == "ResNet-18" and record["id"] in ("dog", "bus"):
                target = ClassifierOutputTarget(top3[0]["index"])
                # Grad-CAM must remain outside no_grad/inference_mode.
                with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
                    heatmap = cam(input_tensor=tensor, targets=[target])[0]
                mean = torch.tensor(transform.mean).view(3, 1, 1)
                std = torch.tensor(transform.std).view(3, 1, 1)
                crop = (tensor[0] * std + mean).permute(1, 2, 0).numpy().clip(0, 1)
                overlay = show_cam_on_image(crop, heatmap, use_rgb=True)
                fig, axes = plt.subplots(1, 2, figsize=(8, 4.4))
                for ax, picture, title in zip(axes, [crop, overlay], ["Exact input crop", "Grad-CAM"]):
                    ax.imshow(picture)
                    ax.set_title(title)
                    ax.axis("off")
                fig.suptitle(f"{record['id']} | {top3[0]['label']} ({100*top3[0]['probability']:.2f}%)")
                fig.tight_layout()
                fig.savefig(root / f"figures/cam_{record['id']}.png", dpi=180)
                plt.close(fig)
        del model
    save_json(root / "results/pretrained_results.json", {"predictions": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["all", "q1", "fashion", "q4"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("hw2_solution"))
    parser.add_argument("--cache-dir", type=Path, default=Path("ee4745_cache"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    root, cache = args.output_dir.resolve(), args.cache_dir.resolve()
    assert cache != root and root not in cache.parents, "Keep data/weights outside the solution folder."
    for folder in [root / "results", root / "figures", root / "images", cache]:
        folder.mkdir(parents=True, exist_ok=True)
    save_json(root / "results/environment.json", {"python": platform.python_version(),
              "torch": torch.__version__, "torchvision": torchvision.__version__,
              "numpy": np.__version__, "device": "CPU", "torch_threads": 2,
              "grad-cam": importlib.metadata.version("grad-cam")})
    if args.part in ("all", "q1"):
        q1(root)
    if args.part in ("all", "fashion"):
        fashion(root, cache)
    if args.part in ("all", "q4"):
        q4(root, cache)


if __name__ == "__main__":
    main()
