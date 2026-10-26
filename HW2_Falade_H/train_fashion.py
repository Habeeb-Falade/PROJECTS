"""Q2/Q3: fixed-protocol Fashion-MNIST experiments.

Data preparation and loader settings are adapted from page 6 of the supplied
EE4745 HW2 PDF. The models, training loop, and evaluation code are original.
Datasets and checkpoints are retained outside this solution directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ee4745_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST


def load_data(data_dir="data"):
    """The assignment's page-6 function, with the same data and split."""
    full = FashionMNIST(data_dir, train=True, download=True)
    test = FashionMNIST(data_dir, train=False, download=True)
    order = np.random.default_rng(2026).permutation(len(full))

    def subset(source, indices):
        # Raw images: uint8 (N, 28, 28). Add the channel dimension.
        images = source.data[indices].unsqueeze(1).float() / 255.0
        labels = source.targets[indices].long()
        return TensorDataset(images, labels)

    train_set = subset(full, order[:10000])
    val_set = subset(full, order[10000:12000])
    test_set = subset(test, np.arange(len(test)))
    return train_set, val_set, test_set


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), nn.Linear(784, width), nn.ReLU(), nn.Linear(width, 10)
        )

    def forward(self, x):
        return self.layers(x)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(12 * 6 * 6, 10),
        )

    def forward(self, x):
        return self.layers(x)


def evaluate(model, loader, device):
    """Evaluate every example with no gradients and example-weighted loss."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    assert total == len(loader.dataset)
    return {"cross_entropy": loss_sum / total, "accuracy": correct / total,
            "examples": total}


def verify_architecture():
    cnn = SmallCNN()
    x = torch.zeros(2, 1, 28, 28)
    shapes = [{"layer": "Input", "shape": list(x.shape)}]
    with torch.no_grad():
        for layer in cnn.layers:
            x = layer(x)
            shapes.append({"layer": type(layer).__name__, "shape": list(x.shape)})
    assert list(x.shape) == [2, 10]
    counts = {"MLP-32": sum(p.numel() for p in MLP(32).parameters()),
              "MLP-128": sum(p.numel() for p in MLP(128).parameters()),
              "CNN": sum(p.numel() for p in cnn.parameters())}
    assert counts == {"MLP-32": 25450, "MLP-128": 101770, "CNN": 5146}
    per_layer = {name: sum(p.numel() for p in layer.parameters())
                 for name, layer in cnn.named_modules()
                 if isinstance(layer, (nn.Conv2d, nn.Linear))}
    return {"parameter_counts": counts, "cnn_dummy_shapes": shapes,
            "cnn_learned_layer_counts": per_layer}


def train_one(name, model, train_loader, train_eval_loader, val_loader,
              device, checkpoint_dir):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state = None
    history = []
    start = time.perf_counter()
    for epoch in range(1, 6):
        model.train()
        examples_seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            examples_seen += labels.numel()
        assert examples_seen == 10000
        train_metrics = evaluate(model, train_eval_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        entry = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(entry)
        if val_metrics["cross_entropy"] < best_loss:
            best_loss = val_metrics["cross_entropy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f"{name} epoch {epoch}/5: "
              f"train CE={train_metrics['cross_entropy']:.6f}, "
              f"acc={100*train_metrics['accuracy']:.2f}%; "
              f"val CE={val_metrics['cross_entropy']:.6f}, "
              f"acc={100*val_metrics['accuracy']:.2f}%", flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    checkpoint_path = checkpoint_dir / f"{name.lower()}.pt"
    torch.save(best_state, checkpoint_path)
    result = {"selected_epoch": best_epoch,
              "selected_train": history[best_epoch - 1]["train"],
              "selected_validation": history[best_epoch - 1]["validation"],
              "history": history,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "training_seconds": time.perf_counter() - start,
              "checkpoint_path": str(checkpoint_path)}
    print(f"Selected {name} epoch {best_epoch} by validation cross-entropy.", flush=True)
    return model, result


def make_plots(results, figure_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.5), sharex=True, sharey=True)
    for name, color in [("MLP-32", "tab:blue"), ("MLP-128", "tab:orange")]:
        history = results[name]["history"]
        axes[0].plot(range(1, 6), [h["train"]["cross_entropy"] for h in history],
                     marker="o", color=color, label=f"{name} training")
        axes[0].plot(range(1, 6), [h["validation"]["cross_entropy"] for h in history],
                     marker="s", linestyle="--", color=color,
                     label=f"{name} validation")
    history = results["CNN"]["history"]
    axes[1].plot(range(1, 6), [h["train"]["cross_entropy"] for h in history],
                 "o-", color="tab:green", label="CNN training")
    axes[1].plot(range(1, 6), [h["validation"]["cross_entropy"] for h in history],
                 "s--", color="tab:red", label="CNN validation")
    axes[0].set_title("Q2: MLP widths on common axes")
    axes[1].set_title("Q3: small CNN")
    axes[0].set_ylabel("Mean cross-entropy (nats per example)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(1, 6))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "fashion_curves.png", dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path.home() / ".cache" / "ee4745_hw2")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    result_dir = args.output_dir / "results"
    figure_dir = args.output_dir / "figures"
    checkpoint_dir = args.cache_dir / "checkpoints"
    for directory in [result_dir, figure_dir, checkpoint_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    run_start = time.perf_counter()
    architecture = verify_architecture()
    mirror_note = "torchvision default mirror"
    try:
        train_set, val_set, test_set = load_data(str(args.cache_dir / "data"))
    except RuntimeError as exc:
        print(f"Default Fashion-MNIST download failed: {exc}", flush=True)
        # The official dataset's GitHub mirror; torchvision still verifies its
        # unchanged per-resource MD5 checksums before extraction.
        FashionMNIST.mirrors = [
            "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
        ]
        mirror_note = FashionMNIST.mirrors[0]
        print(f"Retrying via official HTTPS mirror: {mirror_note}", flush=True)
        train_set, val_set, test_set = load_data(str(args.cache_dir / "data"))
    assert tuple(map(len, [train_set, val_set, test_set])) == (10000, 2000, 10000)

    def make_train_loader():
        return DataLoader(train_set, batch_size=128, shuffle=True,
                          generator=torch.Generator().manual_seed(2026),
                          num_workers=0, drop_last=False)

    # Non-shuffled evaluation loaders from the assignment's page-6 example.
    train_eval_loader = DataLoader(train_set, batch_size=128,
                                   shuffle=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=128,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=128,
                             shuffle=False, num_workers=0)
    images, labels = next(iter(make_train_loader()))
    batch_demo = {"image_shape": list(images.shape), "label_shape": list(labels.shape),
                  "image_dtype": str(images.dtype), "label_dtype": str(labels.dtype),
                  "pixel_min": images.min().item(), "pixel_max": images.max().item()}
    print("DataLoader demo:", batch_demo, flush=True)
    assert images.dtype == torch.float32 and labels.dtype == torch.int64

    models = {}
    results = {}
    selection_events = []
    for name, constructor in [("MLP-32", lambda: MLP(32)),
                              ("MLP-128", lambda: MLP(128)),
                              ("CNN", SmallCNN)]:
        torch.manual_seed(2026)
        train_loader = make_train_loader()
        model = constructor()  # Default initialization, all biases enabled.
        models[name], results[name] = train_one(
            name, model, train_loader, train_eval_loader, val_loader,
            device, checkpoint_dir)
        selection_events.append({"event": "checkpoint_selected", "model": name,
                                 "seconds_from_run_start": time.perf_counter() - run_start})

    # No test-loader iteration occurs until all three runs and selections finish.
    for name in ["MLP-32", "MLP-128", "CNN"]:
        results[name]["test"] = evaluate(models[name], test_loader, device)
        selection_events.append({"event": "test_evaluated_once", "model": name,
                                 "seconds_from_run_start": time.perf_counter() - run_start})
        print(f"{name} selected checkpoint test: {results[name]['test']}", flush=True)

    order = np.random.default_rng(2026).permutation(60000)
    document = {
        "environment": {"python": platform.python_version(),
                        "pytorch": torch.__version__, "torchvision": torchvision.__version__,
                        "numpy": np.__version__, "platform": platform.platform(),
                        "processor": platform.processor(), "device": str(device),
                        "torch_threads": torch.get_num_threads()},
        "protocol": {"epochs": 5, "batch_size": 128, "seed": 2026,
                     "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0,
                     "initialization": "PyTorch defaults; all learned layers use bias",
                     "dtype": "float32", "num_workers": 0, "drop_last": False,
                     "checkpoint_selection": "minimum validation mean cross-entropy; earliest tie",
                     "train_size": 10000, "validation_size": 2000, "test_size": 10000,
                     "split_sha256": hashlib.sha256(order[:12000].tobytes()).hexdigest(),
                     "dataset_mirror": mirror_note,
                     "test_evaluations_per_model": 1},
        "batch_demo": batch_demo, "architecture": architecture,
        "models": results, "protocol_events": selection_events,
        "total_seconds": time.perf_counter() - run_start,
    }
    with (result_dir / "fashion_results.json").open("w") as handle:
        json.dump(document, handle, indent=2)
    make_plots(results, figure_dir)
    print("Saved results and curves.", flush=True)


if __name__ == "__main__":
    main()
