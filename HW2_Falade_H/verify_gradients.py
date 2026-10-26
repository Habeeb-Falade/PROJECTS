"""Q1: independent NumPy chain rule versus PyTorch float64 autograd.

Also verifies the Q2 parameter formula and Q3 layer shapes/counts.
Run: python verify_gradients.py --output results/gradient_results.json
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def verify(output: Path) -> dict:
    # All analytic quantities are computed independently in NumPy.
    x = np.array([1.0, -2.0], dtype=np.float64)
    w1 = np.eye(2, dtype=np.float64)
    b1 = np.array([0.0, 1.0], dtype=np.float64)
    w2 = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    b2 = np.zeros(2, dtype=np.float64)
    target = 1
    a = w1 @ x + b1
    h = np.maximum(a, 0.0)
    z = w2 @ h + b2
    shifted_exp = np.exp(z - z.max())
    probabilities = shifted_exp / shifted_exp.sum()
    one_hot = np.zeros(2, dtype=np.float64)
    one_hot[target] = 1.0
    dz = probabilities - one_hot
    relu_mask = (a > 0.0).astype(np.float64)  # ReLU'(0) = 0.
    dh = w2.T @ dz
    da = dh * relu_mask
    manual = {
        "W2": np.outer(dz, h),
        "b2": dz,
        "W1": np.outer(da, x),
        "b1": da,
    }

    # Autograd is used only here, not to obtain the manual gradient.
    tx = torch.tensor(x, dtype=torch.float64)
    parameters = {
        "W1": torch.tensor(w1, dtype=torch.float64, requires_grad=True),
        "b1": torch.tensor(b1, dtype=torch.float64, requires_grad=True),
        "W2": torch.tensor(w2, dtype=torch.float64, requires_grad=True),
        "b2": torch.tensor(b2, dtype=torch.float64, requires_grad=True),
    }
    logits = parameters["W2"] @ F.relu(parameters["W1"] @ tx + parameters["b1"]) + parameters["b2"]
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], dtype=torch.int64))
    loss.backward()
    autograd = {name: parameter.grad.detach().numpy() for name, parameter in parameters.items()}
    differences = {name: float(np.max(np.abs(manual[name] - autograd[name]))) for name in manual}
    maximum_difference = max(differences.values())
    for name in manual:
        np.testing.assert_allclose(manual[name], autograd[name], atol=1e-14, rtol=0.0)

    # Independent dummy-batch check of the exact Q3 architecture.
    layers = [
        ("conv1", nn.Conv2d(1, 6, kernel_size=5, padding=0, bias=True)),
        ("relu1", nn.ReLU()),
        ("pool1", nn.MaxPool2d(kernel_size=2, stride=2)),
        ("conv2", nn.Conv2d(6, 12, kernel_size=3, padding=1, bias=True)),
        ("relu2", nn.ReLU()),
        ("pool2", nn.MaxPool2d(kernel_size=2, stride=2)),
        ("flatten", nn.Flatten()),
        ("classifier", nn.Linear(12 * 6 * 6, 10, bias=True)),
    ]
    batch = torch.zeros(2, 1, 28, 28, dtype=torch.float32)
    shapes = {"input": list(batch.shape)}
    parameter_counts = {}
    with torch.no_grad():
        for name, layer in layers:
            batch = layer(batch)
            shapes[name] = list(batch.shape)
            count = sum(parameter.numel() for parameter in layer.parameters())
            if count:
                parameter_counts[name] = count
    expected_shapes = {
        "input": [2, 1, 28, 28], "conv1": [2, 6, 24, 24],
        "relu1": [2, 6, 24, 24], "pool1": [2, 6, 12, 12],
        "conv2": [2, 12, 12, 12], "relu2": [2, 12, 12, 12],
        "pool2": [2, 12, 6, 6], "flatten": [2, 432], "classifier": [2, 10],
    }
    assert shapes == expected_shapes
    assert parameter_counts == {"conv1": 156, "conv2": 660, "classifier": 4330}
    total_cnn = sum(parameter_counts.values())
    assert total_cnn == 5146
    mlp_counts = {}
    for width in (32, 128):
        model = nn.Sequential(nn.Flatten(), nn.Linear(784, width), nn.ReLU(), nn.Linear(width, 10))
        actual = sum(parameter.numel() for parameter in model.parameters())
        analytic = 795 * width + 10
        assert actual == analytic
        mlp_counts[str(width)] = {"formula": analytic, "pytorch": actual}

    result = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "dtype": "float64",
        "target_zero_based": target,
        "a": a.tolist(), "h": h.tolist(), "z": z.tolist(),
        "probabilities": probabilities.tolist(),
        "loss": float(loss.item()),
        "relu_mask": relu_mask.tolist(),
        "dz": dz.tolist(), "dh": dh.tolist(), "da": da.tolist(),
        "manual_gradients": {name: value.tolist() for name, value in manual.items()},
        "autograd_gradients": {name: value.tolist() for name, value in autograd.items()},
        "max_absolute_difference_by_parameter": differences,
        "max_absolute_difference": maximum_difference,
        "mlp_parameter_counts": mlp_counts,
        "cnn_dummy_batch_shapes": shapes,
        "cnn_parameter_counts": parameter_counts,
        "cnn_total_parameters": total_cnn,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "gradient_results.json")
    args = parser.parse_args()
    verify(args.output)
