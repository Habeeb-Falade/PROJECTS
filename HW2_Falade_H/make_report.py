"""Build the worked-solution PDF from the actual saved experiment results."""
from __future__ import annotations

import io
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, Preformatted, KeepTogether,
)

ROOT = Path(__file__).resolve().parent
FONT = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("DejaVu", str(FONT / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONT / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(FONT / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono", str(FONT / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                            italic="DejaVu-Oblique", boldItalic="DejaVu-Bold")

INK = colors.HexColor("#172B45")
BLUE = colors.HexColor("#165D9B")
PALE = colors.HexColor("#EEF4FA")
GRAY = colors.HexColor("#526273")
WIDTH = A4[0] - 92
STYLES = {
    "title": ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=20,
                            leading=25, textColor=INK, spaceAfter=13),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=16,
                         leading=21, textColor=INK, spaceAfter=12),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=11.8,
                         leading=16, textColor=BLUE, spaceBefore=10, spaceAfter=7),
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=10,
                           leading=14.8, textColor=INK, spaceAfter=8),
    "small": ParagraphStyle("small", fontName="DejaVu", fontSize=8.4,
                            leading=11.5, textColor=GRAY, spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=8.2,
                           leading=11.4, textColor=INK),
    "headcell": ParagraphStyle("headcell", fontName="DejaVu-Bold", fontSize=8.2,
                               leading=11.4, textColor=colors.white),
    "eq": ParagraphStyle("eq", fontName="DejaVu", fontSize=10.5,
                         leading=17, textColor=INK, leftIndent=10,
                         spaceBefore=4, spaceAfter=8),
    "mono": ParagraphStyle("mono", fontName="DejaVuMono", fontSize=9.1,
                           leading=13, textColor=INK, leftIndent=9,
                           spaceAfter=10),
}


def p(text, style="body"):
    return Paragraph(text, STYLES[style])


def table(rows, widths=None, small=False):
    converted = []
    for i, row in enumerate(rows):
        converted.append([p(str(cell), "headcell" if i == 0 else "cell")
                          for cell in row])
    t = Table(converted, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, BLUE),
    ]))
    return t


def image(path, width=WIDTH, max_height=500):
    data = path if isinstance(path, io.BytesIO) else str(path)
    with PILImage.open(data) as im:
        w, h = im.size
    if isinstance(data, io.BytesIO):
        data.seek(0)
    scale = min(width / w, max_height / h)
    return Image(data, width=w * scale, height=h * scale, hAlign="CENTER")


def chart(models, names, width=7.0, height=3.7):
    fig, ax = plt.subplots(figsize=(width, height))
    for name, color in zip(names, ["#165D9B", "#D56A25", "#278F68"]):
        h = models[name]["history"]
        ax.plot(range(1, 6), [r["train"]["cross_entropy"] for r in h],
                "o-", color=color, label=f"{name}: training")
        ax.plot(range(1, 6), [r["validation"]["cross_entropy"] for r in h],
                "s--", color=color, label=f"{name}: validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean cross-entropy (nats per example)")
    ax.set_xticks(range(1, 6))
    ax.grid(alpha=.22)
    ax.legend(fontsize=9)
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=180)
    plt.close(fig)
    b.seek(0)
    return image(b, max_height=285)


def summary_rows(models):
    rows = [["Model", "Parameters", "Selected<br/>epoch", "Train<br/>accuracy", "Validation<br/>accuracy", "Test<br/>accuracy"]]
    for name in ["MLP-32", "MLP-128", "CNN"]:
        r = models[name]
        rows.append([name, f"{r['parameter_count']:,}", str(r["selected_epoch"]),
                     f"{100*r['selected_train']['accuracy']:.2f}%",
                     f"{100*r['selected_validation']['accuracy']:.2f}%",
                     f"{100*r['test']['accuracy']:.2f}%"])
    return rows


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E3ED"))
    canvas.line(46, 37, A4[0] - 46, 37)
    canvas.setFont("DejaVu", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(46, 23, "Habeeb Falade · EE4745 · Homework 2")
    canvas.drawRightString(A4[0] - 46, 23, str(doc.page))
    canvas.restoreState()


def main():
    g = json.loads((ROOT / "results/gradient_results.json").read_text())
    f = json.loads((ROOT / "results/fashion_results.json").read_text())
    q = json.loads((ROOT / "results/pretrained_results.json").read_text())
    models = f["models"]
    narrow, wide, cnn = [models[k] for k in ["MLP-32", "MLP-128", "CNN"]]
    story = []
    add = story.extend

    # PAGE 1: symbolic derivation.
    add([p("Neural Networks", "title"),
         p("EE4745 · Homework 2 · Fall 2026", "small"),
         p("<b>Habeeb Falade</b>"),
         p("AI-use disclosure: OpenAI Codex generated the derivations, code, and report.", "small"),
         p("Codex executed the experiments and numerical checks reported here.", "small"),
         Spacer(1, 8), p("Q1(a) · What two affine layers can express", "h2"),
         p("Replacing ReLU with the identity gives h = W<sub>1</sub>x + b<sub>1</sub>. Substitution into the output layer gives"),
         p("z = W<sub>2</sub>(W<sub>1</sub>x + b<sub>1</sub>) + b<sub>2</sub><br/>"
           "&nbsp;&nbsp;= (W<sub>2</sub>W<sub>1</sub>)x + (W<sub>2</sub>b<sub>1</sub> + b<sub>2</sub>).", "eq"),
         p("Thus W<sub>eff</sub> = W<sub>2</sub>W<sub>1</sub> ∈ ℝ<super>C×d</super> and "
           "b<sub>eff</sub> = W<sub>2</sub>b<sub>1</sub> + b<sub>2</sub> ∈ ℝ<super>C</super>. "
           "A pairwise class boundary satisfies (w<sub>i</sub> − w<sub>j</sub>)<super>T</super>x + "
           "b<sub>i</sub> − b<sub>j</sub> = 0, which is a hyperplane. Affine composition remains affine, "
           "and softmax preserves the ordering of logits, so stacking these layers cannot introduce nonlinear decision boundaries."),
         p("Q1(b) · Cross-entropy and the chain rule", "h2"),
         p("Starting with ℓ = −z<sub>y</sub> + log ∑<sub>j</sub> exp(z<sub>j</sub>), the kth derivative is"),
         p("∂ℓ/∂z<sub>k</sub> = −1{k = y} + exp(z<sub>k</sub>)/∑<sub>j</sub> exp(z<sub>j</sub>)"
           " = p<sub>k</sub> − 1{k = y}.", "eq"),
         p("Write δ<sub>z</sub> = p − e<sub>y</sub> and r = 1{a &gt; 0}. The mask uses zero at a = 0, "
           "as required. The chain rule first propagates through z = W<sub>2</sub>h + b<sub>2</sub>, "
           "then through ReLU and a = W<sub>1</sub>x + b<sub>1</sub>:"),
         p("∇<sub>h</sub>ℓ = W<sub>2</sub><super>T</super>δ<sub>z</sub>, &nbsp; "
           "δ<sub>a</sub> = (W<sub>2</sub><super>T</super>δ<sub>z</sub>) ⊙ r.", "eq"),
         table([["Parameter", "Gradient", "Shape (column-vector convention)"],
                ["W<sub>2</sub>", "δ<sub>z</sub>h<super>T</super>", "C × m"],
                ["b<sub>2</sub>", "δ<sub>z</sub>", "C × 1"],
                ["W<sub>1</sub>", "δ<sub>a</sub>x<super>T</super>", "m × d"],
                ["b<sub>1</sub>", "δ<sub>a</sub>", "m × 1"]], [90, 172, WIDTH-262]),
         Spacer(1, 9),
         p("For example, dz = dW<sub>2</sub>h + W<sub>2</sub>dh + db<sub>2</sub>; taking the inner product "
           "with δ<sub>z</sub> produces the outer product δ<sub>z</sub>h<super>T</super>. "
           "The same argument applied to da gives δ<sub>a</sub>x<super>T</super>.", "small"),
         PageBreak()])

    # PAGE 2: numerical example and verification.
    add([p("Q1(b–c) · Numerical gradients and verification", "h1"),
         p("Using the second class (PyTorch label 1), the supplied parameters give"),
         Preformatted("a = [1, -1]^T     h = [1, 0]^T     z = [1, -1]^T\n"
                      "p = [0.880797077978, 0.119202922022]^T\n"
                      "loss = log(1 + exp(2)) = 2.126928011043", STYLES["mono"]),
         p("Let q = exp(2)/(exp(2) + 1) = 0.880797077977882. Then δ<sub>z</sub> = (q, −q)<super>T</super>, "
           "W<sub>2</sub><super>T</super>δ<sub>z</sub> = (2q, −2q)<super>T</super>, r = (1, 0)<super>T</super>, "
           "and δ<sub>a</sub> = (2q, 0)<super>T</super>. Therefore all four gradients are"),
         Preformatted("grad W2 = [[ q, 0],       grad b2 = [ q, -q]^T\n"
                      "           [-q, 0]]\n\n"
                      "grad W1 = [[2q, -4q],    grad b1 = [2q, 0]^T\n"
                      "           [ 0,   0]]", STYLES["mono"]),
         table([["Gradient", "Numerical entries"],
                ["∇<sub>W₂</sub>ℓ", "[[0.880797077978, 0], [−0.880797077978, 0]]"],
                ["∇<sub>b₂</sub>ℓ", "[0.880797077978, −0.880797077978]<super>T</super>"],
                ["∇<sub>W₁</sub>ℓ", "[[1.761594155956, −3.523188311912], [0, 0]]"],
                ["∇<sub>b₁</sub>ℓ", "[1.761594155956, 0]<super>T</super>"]], [88, WIDTH-88]),
         Spacer(1, 8), p("Independent float64 check", "h2"),
         p("verify_gradients.py evaluates the forward pass and analytic chain-rule gradients with NumPy float64. "
           "It separately creates PyTorch float64 leaf parameters, calls cross_entropy on the logits, "
           "and obtains each gradient with backward(). Every entry of all four parameter gradients is compared."),
         p(f"<b>Measured maximum absolute difference: {g['max_absolute_difference']:.1f}</b> "
           "(also 0.0 separately for W₁, b₁, W₂, and b₂). Rank-one arrays store the column-vector entries in the code; "
           "the outer-product formulas are unchanged."),
         p("Software and hardware used throughout", "h2"),
         p(f"Python {escape(f['environment']['python'])}; PyTorch {escape(f['environment']['pytorch'])}; "
           f"torchvision {escape(f['environment']['torchvision'])}; NumPy {escape(f['environment']['numpy'])}. "
           "Linux x86-64 CPU, Intel Xeon Platinum 8573C; five vCPUs available, two PyTorch threads per experiment. "
           "Q1 uses float64; training and image inference use float32. No GPU was used.", "small"),
         PageBreak()])

    # PAGE 3: protocol and MLP results.
    add([p("Q2(a) · Training the two MLPs", "h1"),
         p("Architecture: Flatten → Linear(784, H) → ReLU → Linear(H, 10), independently for H = 32 and H = 128. "
           "The implementation explicitly clears gradients, computes raw-logit cross-entropy, backpropagates, and takes an Adam step."),
         p("The assignment’s load_data function fixes 10,000 training and 2,000 validation images with NumPy seed 2026; "
           "the complete official 10,000-image test split is retained. Pixels are float32 in [0, 1], labels are int64. "
           "Each run uses five epochs, batch size 128, Adam(lr=0.001, weight_decay=0), default initialization and biases. "
           "Before constructing each model, torch.manual_seed(2026) is reset and a fresh shuffled DataLoader uses its own "
           "generator seeded 2026, num_workers=0 and drop_last=False."),
         p("At each epoch’s end, non-shuffled loaders evaluate the full training and validation splits under eval() and no_grad(), "
           "with losses weighted by example count. The minimum validation cross-entropy selects a copied checkpoint "
           "(earliest epoch breaks ties). All three model runs and checkpoint selections finish before the test set is evaluated once per model. "
           "There is no augmentation, extra normalization, dropout, or batch normalization.", "small"),
         chart(models, ["MLP-32", "MLP-128"]),
         p("End-of-epoch losses, evaluated after training each epoch; both widths share the same axes.", "small"),
         table(summary_rows(models), [72, 76, 65, 88, 104, WIDTH-405]),
         Spacer(1, 7),
         p("The CNN row is included here for the combined Q2/Q3 comparison; its architecture and curves are discussed below. "
           "All selected checkpoints are from epoch 5. Complete per-epoch measurements and protocol event order are saved in results/fashion_results.json.", "small"),
         PageBreak()])

    # PAGE 4: width analysis and all-epoch numeric evidence.
    gap_n = 100*(narrow["selected_train"]["accuracy"]-narrow["selected_validation"]["accuracy"])
    gap_w = 100*(wide["selected_train"]["accuracy"]-wide["selected_validation"]["accuracy"])
    add([p("Q2(b) · Parameters and the effect of width", "h1"),
         p("Including both layers’ biases, the input-to-hidden layer contributes 784H + H parameters and the hidden-to-output "
           "layer contributes 10H + 10. Hence"),
         p("P(H) = 784H + H + 10H + 10 = <b>795H + 10</b>.", "eq"),
         table([["Width", "Formula", "PyTorch parameter sum"],
                ["H = 32", "795 × 32 + 10 = 25,450", "25,450"],
                ["H = 128", "795 × 128 + 10 = 101,770", "101,770"]], [85, 218, WIDTH-303]),
         Spacer(1, 8),
         p("sum(p.numel() for p in model.parameters()) verifies both counts. Widening increases the parameter count by "
           f"{wide['parameter_count']/narrow['parameter_count']:.3f}×. It lowers selected training cross-entropy from "
           f"{narrow['selected_train']['cross_entropy']:.4f} to {wide['selected_train']['cross_entropy']:.4f}, and validation cross-entropy "
           f"from {narrow['selected_validation']['cross_entropy']:.4f} to {wide['selected_validation']['cross_entropy']:.4f}. "
           "Training accuracy improves by 1.87 percentage points and validation accuracy by 1.05 points; test accuracy rises by 1.54 points."),
         p(f"The selected training-minus-validation accuracy gaps are {gap_n:.2f} percentage points for H = 32 and "
           f"{gap_w:.2f} for H = 128. Both training and validation loss decrease throughout the five epochs. "
           "The wider model’s slightly larger gap is small, and its validation loss does not turn upward, so this run does not show clear overfitting. "
           "The narrower model’s higher training and validation losses are consistent with greater underfitting relative to the wider model, "
           "although five still-improving epochs cannot separate limited capacity from incomplete optimization or establish either model’s asymptotic behavior."),
         p("One next experiment — proposed, not run", "h2"),
         p("Train both widths for 20 epochs with all other settings unchanged, selecting only by validation cross-entropy and leaving "
           "the test set untouched. This would show whether the narrow model catches up with more optimization and whether the wider "
           "model eventually develops a widening training–validation loss gap."),
         p("Measured losses behind the MLP curves", "h2"),
         table([["Epoch", "H=32 train CE", "H=32 val CE", "H=128 train CE", "H=128 val CE"]] +
               [[str(i+1), f"{narrow['history'][i]['train']['cross_entropy']:.6f}",
                 f"{narrow['history'][i]['validation']['cross_entropy']:.6f}",
                 f"{wide['history'][i]['train']['cross_entropy']:.6f}",
                 f"{wide['history'][i]['validation']['cross_entropy']:.6f}"] for i in range(5)],
               [48, 113, 113, 113, WIDTH-387]),
         PageBreak()])

    # PAGE 5: CNN dimensions and count.
    add([p("Q3(a) · CNN shapes and parameter counts", "h1"),
         p("For a spatial input dimension n, kernel k, padding p, dilation d and stride s, convolution or pooling produces"),
         p("n<sub>out</sub> = floor((n + 2p − d(k − 1) − 1)/s + 1).", "eq"),
         p("Convolutions use stride 1 and dilation 1. ReLU leaves the shape unchanged; both max pools use a 2 × 2 kernel and stride 2."),
         table([["Stage", "Spatial calculation", "Output shape"],
                ["Input", "—", "B × 1 × 28 × 28"],
                ["Conv1 + ReLU", "(28 − 5)/1 + 1 = 24", "B × 6 × 24 × 24"],
                ["Max pool 1", "floor((24 − 2)/2 + 1) = 12", "B × 6 × 12 × 12"],
                ["Conv2 + ReLU", "(12 + 2 × 1 − 3)/1 + 1 = 12", "B × 12 × 12 × 12"],
                ["Max pool 2", "floor((12 − 2)/2 + 1) = 6", "B × 12 × 6 × 6"],
                ["Flatten", "12 × 6 × 6 = 432", "B × 432"],
                ["Linear classifier", "432 → 10 logits", "B × 10"]], [113, 226, WIDTH-339]),
         Spacer(1, 12),
         table([["Learned layer", "Weights plus biases", "Parameters"],
                ["Conv1", "6(1 × 5 × 5 + 1)", "156"],
                ["Conv2", "12(6 × 3 × 3 + 1)", "660"],
                ["Classifier", "10(432 + 1)", "4,330"],
                ["Total", "156 + 660 + 4,330", "5,146"]], [115, 250, WIDTH-365]),
         Spacer(1, 10),
         p("A dummy batch with B = 2 verifies every convolution, ReLU, pooling, flatten and classifier output shape. "
           "The total from sum(p.numel() for p in model.parameters()) is also 5,146, with the same layer-by-layer counts."),
         p("Each output-channel convolution filter and its bias are reused at every spatial position. "
           "Moving the filter produces distinct activations from local image patches, but does not create new learned parameters. "
           "Multiplying the filter parameter count by the output height and width would therefore overcount the shared weights."),
         PageBreak()])

    # PAGE 6: CNN training and required measured comparison (five sentences).
    add([p("Q3(b) · CNN training and comparison", "h1"),
         p("The CNN was trained from scratch for five epochs with exactly the Q2 data splits, preprocessing, optimizer, seeds, "
           "loader settings, and checkpoint rule. Its selected checkpoint is epoch 5, with validation cross-entropy 0.567707."),
         chart(models, ["CNN"]),
         table(summary_rows(models), [72, 76, 65, 88, 104, WIDTH-405]),
         Spacer(1, 10), p("Comparison with the validation-selected MLP", "h2"),
         p("The comparison model is MLP-128 because its selected validation cross-entropy (0.458639) is lower than "
           "MLP-32’s (0.520183), independently of their test scores. "
           "The CNN reaches 79.04% training, 79.25% validation and 77.80% test accuracy, compared with MLP-128’s "
           "85.11%, 84.25% and 82.82%, so it trails by 5.00 validation and 5.02 test percentage points in this run. "
           "It uses only 5,146 parameters versus 101,770 for MLP-128, about 19.78 times fewer. "
           "Local receptive fields and shared convolutional filters provide an efficient way to reuse edge and texture features "
           "across positions, whereas the MLP learns separate connections for each input pixel. "
           "Those useful inductive biases do not guarantee an advantage for this very small two-convolution architecture after "
           "only five epochs: both CNN losses are still falling and its tiny training–validation gap suggests incomplete fitting "
           "rather than clear overfitting."),
         p("The code records three checkpoint-selection events before the first test-evaluation event, and exactly one test "
           "evaluation for each selected checkpoint. No training setting was changed after seeing the test results.", "small"),
         PageBreak()])

    # PAGE 7: image descriptions and provenance.
    add([p("Q4(a) · Three images selected before inference", "h1"),
         p("The images, source URLs, SHA-256 hashes, and the descriptions below were saved in images/sources.json before "
           "running either classifier. The same three RGB images are used for both models; the bus scene is the harder case."),
         image(ROOT / "figures/q4_images.png", max_height=280),
         Spacer(1, 8)])
    for item in q["images"]:
        add([p(item["id"].capitalize() + " · " + item["difficulty"], "h2"),
             p(escape(item["description_before_predictions"])),
             p("Source: " + escape(item["source_url"]), "small")])
    add([p("Exact dog or cat breed is not independently known from these photographs. Correctness judgments below therefore "
           "distinguish a broad animal match from an unverified exact ImageNet label.", "small"), PageBreak()])

    # PAGE 8: six measured prediction rows.
    add([p("Q4(b) · ImageNet predictions", "h1"),
         p("Both classifiers are torchvision models in eval() mode with IMAGENET1K_V1 weights; neither is trained or fine-tuned. "
           "Each RGB image is processed with its own weight object’s transforms(), including resizing, center cropping and normalization. "
           "Probabilities are softmax(logits, dim=1), and labels are read from weights.meta[\"categories\"]."),
         p("ResNet-18: ResNet18_Weights.IMAGENET1K_V1. MobileNetV3-Small: MobileNet_V3_Small_Weights.IMAGENET1K_V1. "
           "Both input crops are 224 × 224; ImageNet normalization uses mean (0.485, 0.456, 0.406) and "
           "standard deviation (0.229, 0.224, 0.225).", "small")])
    prediction_rows = [["Image / model", "Top three labels and softmax probabilities", "Top-1 match and reason"]]
    for r in q["predictions"]:
        prediction_rows.append([
            escape(r["image_id"].capitalize()) + "<br/>" + escape(r["model"]),
            "<br/>".join(f"{i+1}. {escape(t['label'])}: {100*t['probability']:.2f}%"
                         for i, t in enumerate(r["top3"])),
            "<b>" + escape(r["correctness"]) + "</b> — " + escape(r["reason"]),
        ])
    add([table(prediction_rows, [105, 203, WIDTH-308]), Spacer(1, 10),
         p("Comparison", "h2"),
         p(escape(q["comparison"])),
         PageBreak()])

    # PAGE 9: both actual crop/CAM figure pairs.
    add([p("Q4(c) · ResNet-18 Grad-CAM", "h1"),
         p("Grad-CAM targets each image’s own ResNet-18 top predicted class logit, using target_layers = [model.layer4[-1]] "
           "and grad-cam 1.5.7. The CAM call runs with gradients enabled. Each overlay uses the exact model input crop, "
           "recovered by undoing its normalization.", "small")])
    for r in q["grad_cam"]:
        path = Path(r["figure"])
        if not path.is_absolute():
            path = ROOT / path
        add([image(path, max_height=217), Spacer(1, 4),
             p(escape(r["interpretation"])), Spacer(1, 8)])
    add([p("Red indicates stronger and blue weaker support within an individual map. These coarse, separately normalized maps "
           "are visual clues, not segmentation masks or proof of causation; their colors do not compare confidence across images.", "small"),
         PageBreak()])

    # PAGE 10: citations and reproducibility.
    add([p("Run instructions and references", "h1"),
         p("Reproducing the results", "h2"),
         p("The supplied Python files implement the derivation check, both MLPs, CNN, custom training and evaluation loops, "
           "pretrained prediction, and Grad-CAM. README.md contains environment creation and the full commands. "
           "Run scripts from this solution directory, keeping the external data/weight cache separate from the submitted files."),
         Preformatted("python verify_gradients.py\n"
                      "python train_fashion.py --cache-dir /path/to/cache\n"
                      "python pretrained_cam.py --cache-dir /path/to/cache\n"
                      "python make_report.py", STYLES["mono"]),
         p("The result JSON files retain the measured per-epoch metrics, checkpoint-selection order, complete prediction probabilities, "
           "and numerical-gradient comparisons. Downloaded Fashion-MNIST data and model checkpoints are intentionally excluded from the solution package. "
           "A new run can differ slightly across platforms and PyTorch versions despite the fixed seeds."),
         p("Dataset download note", "h2"),
         p("The default Fashion-MNIST HTTP mirror was unavailable in this environment. The loader retried the official dataset’s HTTPS "
           "GitHub mirror at raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/. "
           "It continued to use torchvision.datasets.FashionMNIST and its unchanged file MD5 checks; the dataset, split, and preprocessing were not changed."),
         p("Sources and adapted examples", "h2"),
         p("[1] Supplied EE4745_HW2_Fall2026.pdf, pages 1–6. The page-6 load_data function and DataLoader settings are reproduced/adapted, "
           "and the page-5 Grad-CAM guide is followed. The models, training and evaluation loops, prediction code, and verification code were written for this solution.", "small"),
         p("[2] PyTorch, Datasets &amp; DataLoaders.<br/>https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html", "small"),
         p("[3] PyTorch, Optimizing Model Parameters.<br/>https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html", "small"),
         p("[4] torchvision model and weight APIs.<br/>https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html<br/>"
           "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html", "small"),
         p("[5] Jacob Gildenblat et al., pytorch-grad-cam, Usage examples and Choosing the layer(s). "
           "The GradCAM context-manager example and ClassifierOutputTarget/show_cam_on_image interface are adapted.<br/>"
           "https://github.com/jacobgil/pytorch-grad-cam#usage-examples", "small"),
         p("[6] Image sources are given in Q4(a) and images/sources.json: pytorch/hub dog.jpg, scikit-image chelsea.png, "
           "and ultralytics/assets bus.jpg. SHA-256 values identify the exact downloaded files.", "small"),
         p("[7] Official Fashion-MNIST repository.<br/>https://github.com/zalandoresearch/fashion-mnist", "small")])

    out = ROOT / "report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=46, rightMargin=46,
                            topMargin=43, bottomMargin=49,
                            title="EE4745 Homework 2 — Habeeb Falade",
                            author="Habeeb Falade",
                            creator="OpenAI Codex; AI assistance disclosed on page 1")
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
