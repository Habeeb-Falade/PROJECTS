# EE4745 Homework 2 — Habeeb Falade

Submission archive: `HW2_Falade_H.zip`. The archive contains `report.pdf` and
the supporting files listed below.

`report.pdf` answers Q1(a–c), Q2(a–b), Q3(a–b), and Q4(a–c), with the
derivations, measured results, plots, images, and Grad-CAM interpretations.
All reported numerical experiments were executed on a CPU. The report includes
the required two-line AI-use disclosure and citations for adapted examples.

## Contents

- `verify_gradients.py`: independent NumPy chain-rule calculation, PyTorch
  float64 autograd check, and architecture/count checks.
- `derivations.md`: readable mathematical derivations for Q1 and Q3(a), plus
  the Q2 parameter formula.
- `train_fashion.py`: the exact fixed-split, five-epoch experiments for both
  MLP widths and the CNN. The three validation-based checkpoint selections
  all finish before any test-set evaluation.
- `pretrained_cam.py`: RGB ImageNet inference with the two specified V1
  pretrained weight sets, followed by ResNet-18 Grad-CAM for its own top class.
- `images/`: the three actual source images and `sources.json`, containing
  source URLs, pre-prediction descriptions, image dimensions, and SHA-256s.
- `results/`: complete numerical outputs, histories, protocol events, and
  runtime metadata. These are actual recorded results, not targets.
- `figures/`: loss curves, the image overview, and input-crop/CAM comparisons.
- `make_report.py`: rebuilds the report from the recorded results and figures.
- `reproduce.sh`: reruns all experiments and rebuilds the report.
- `solve_hw2.py`: an alternative standalone script for all numerical experiments.
  To keep its output separate from the recorded report inputs, run it with
  `--output-dir ../hw2_standalone_output --cache-dir ../ee4745_cache`.

Downloaded Fashion-MNIST data and model weights/checkpoints are deliberately
excluded from the deliverable. The Python scripts keep them in a separate
cache. The included report and figures can be read without downloading data.

## Reproduce

Use Python 3.12. The recorded environment used Python 3.12.14,
PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu, NumPy 2.3.5,
matplotlib 3.10.8, Pillow 12.3.0, and grad-cam 1.5.7.
The CPU was an Intel Xeon Platinum 8573C; each experiment used two PyTorch
compute threads, with no GPU.

Run these commands from this directory. A virtual environment outside the
solution directory keeps dependencies out of the submission archive:

```bash
python3 -m venv ../ee4745_env
source ../ee4745_env/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0
python -m pip install -r requirements.txt
bash reproduce.sh
```

The default cache for `reproduce.sh` is `../ee4745_cache`, outside the solution
directory. Set `EE4745_CACHE` to another writable directory if desired:

```bash
EE4745_CACHE=/tmp/ee4745_cache bash reproduce.sh
```

In the prepared cloud machine, the environment is already installed at
`/workspace/ee4745_env`; activate it with
`source /workspace/ee4745_env/bin/activate`.
The recorded run's cache is `/workspace/ee4745_cache`.

To rebuild only the PDF from the supplied results (no training or download):

```bash
python make_report.py
```

The PDF renderer uses the standard DejaVu fonts provided by this Linux
environment. For another Linux machine, install the distribution's
`fonts-dejavu-core` package if these fonts are missing.

The PDF generator retains Habeeb Falade on the first page, in page footers,
and in the PDF metadata when the report is rebuilt.

The first full run requires Internet access to the Python package registries,
`download.pytorch.org` (and its wheel CDN), and
`raw.githubusercontent.com`. torchvision's default Fashion-MNIST HTTP
download returned 403 here, so the code uses the dataset publisher's official
HTTPS GitHub mirror on failure, retaining torchvision's expected MD5 checks.
Pretrained model downloads retain torchvision's hash checks. The three
bundled Q4 images are checked against their recorded SHA-256 hashes.

Minor numerical differences are possible on other PyTorch builds or hardware.
The report's table describes the included recorded run; no additional seeds
or longer-training experiments were run or used to select it.

## Attribution and AI use

The dataset split and DataLoader settings adapt page 6 of the supplied
`EE4745_HW2_Fall2026.pdf`; Grad-CAM API usage adapts its page 5 guide and
the package's official usage example. The model definitions, training and
evaluation loops, prediction code, and independent gradient verification
were written for this solution. Image sources are listed in
`images/sources.json` and the report. Codex generated the solution and ran
the checks; this is an AI-assisted worked solution.

References:

- https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
- https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.resnet18.html
- https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.mobilenet_v3_small.html
- https://github.com/jacobgil/pytorch-grad-cam#usage-examples
- https://github.com/zalandoresearch/fashion-mnist
