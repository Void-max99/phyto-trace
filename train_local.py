"""PhytoTrace local CPU training.

Trains a 6-class leaf classifier without a GPU and without TensorFlow:

  1. Freeze a pretrained MobileNetV2 (ONNX, ImageNet weights) as a feature extractor.
  2. Run every training image through it once and cache the 1280-d pooled features.
  3. Fit a weighted multinomial logistic-regression head on those features (NumPy).
  4. Graft the head back onto the backbone as a single ONNX graph that honours the
     contract src/inference.py validates: NHWC RGB 0-255 in, softmax probabilities out.

Splits come from the official PlantVillage colour train/test files, not a random
80/20 of the same pool, so the reported numbers are not inflated by near-duplicates.

Usage:
    python train_local.py                     # full run
    python train_local.py --limit 200         # smoke test on a subset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(os.environ.get("PHYTOTRACE_CACHE", Path.home() / ".cache" / "phytotrace"))
ZIP_PATH = CACHE_DIR / "data.zip"
BACKBONE_PATH = CACHE_DIR / "mobilenetv2-12.onnx"
RAW_DIR = CACHE_DIR
FEATURE_CACHE = CACHE_DIR / "features.npz"

SPLIT_FILES = {
    "train": CACHE_DIR / "color_train.txt",
    "test": CACHE_DIR / "color_test.txt",
}
SPLIT_URLS = {
    "train": "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits/color_train.txt",
    "test": "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits/color_test.txt",
}

# Folder name in PlantVillage -> label shown in the UI (order defines output index).
CLASSES = {
    "Potato___Early_blight": "Potato early blight",
    "Potato___Late_blight": "Potato late blight",
    "Potato___healthy": "Potato healthy",
    "Tomato___Early_blight": "Tomato early blight",
    "Tomato___Late_blight": "Tomato late blight",
    "Tomato___healthy": "Tomato healthy",
}
CLASS_NAMES = list(CLASSES)
DISPLAY_NAMES = [CLASSES[name] for name in CLASS_NAMES]

IMG_SIZE = 224
FEATURE_DIM = 1280
# Where the backbone's global-average-pool input lives in mobilenetv2-12.
BACKBONE_CUT = "463"
SCALE = 1.0 / 127.5
SHIFT = 1.0

EPOCHS = 300
LEARNING_RATE = 0.01
L2 = 1e-4
VALIDATION_FRACTION = 0.1
TARGET_COVERAGE = 0.80
SEED = 42


# --------------------------------------------------------------------------- data


def extract_classes(zip_path: Path = ZIP_PATH, raw_dir: Path = RAW_DIR) -> None:
    """Extract only the six configured class folders from the PlantVillage zip."""
    wanted = tuple(f"raw/color/{name}/" for name in CLASS_NAMES)
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            m
            for m in archive.infolist()
            if m.filename.startswith(wanted) and not m.is_dir()
        ]
        if not members:
            raise SystemExit(f"No images for {CLASS_NAMES} inside {zip_path}")
        print(f"extracting {len(members)} images -> {raw_dir}")
        for index, member in enumerate(members, 1):
            target = raw_dir / member.filename
            if target.exists() and target.stat().st_size == member.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(target, "wb") as sink:
                while chunk := source.read(1 << 20):
                    sink.write(chunk)
            if index % 1000 == 0:
                print(f"  {index}/{len(members)}")


def read_split(name: str, limit: int | None = None) -> list[tuple[Path, int]]:
    """Map a split file to (local image path, class index), keeping our six classes."""
    cached = CACHE_DIR / f"{name}_index.json"
    if cached.exists() and limit is None:
        payload = json.loads(cached.read_text())
        return [(RAW_DIR / rel, label) for rel, label in payload]

    split_file = SPLIT_FILES[name]
    if not split_file.exists():
        import urllib.request

        print(f"fetching {SPLIT_URLS[name]}")
        urllib.request.urlretrieve(SPLIT_URLS[name], split_file)
    lines = split_file.read_text(encoding="utf-8").splitlines()

    prefix = "raw/color/"
    wanted = {f"{prefix}{name}/": index for index, name in enumerate(CLASS_NAMES)}
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith(prefix):
            continue
        folder = line[: line.index("/", len(prefix)) + 1]
        if folder not in wanted:
            continue
        rows.append((line, wanted[folder]))
    if limit is not None:
        # keep the subset class-balanced so a smoke test is not one class
        per_class = max(1, limit // len(CLASS_NAMES))
        counts = dict.fromkeys(range(len(CLASS_NAMES)), 0)
        balanced = []
        for row in rows:
            if counts[row[1]] < per_class:
                counts[row[1]] += 1
                balanced.append(row)
        rows = balanced
    else:
        cached.write_text(json.dumps(rows))

    return [(RAW_DIR / rel, label) for rel, label in rows]


# ---------------------------------------------------------------------- features


def load_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        image = opened.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)


def backbone_session() -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = os.cpu_count() or 1
    options.log_severity_level = 3
    return ort.InferenceSession(
        str(BACKBONE_PATH), options, providers=["CPUExecutionProvider"]
    )


def build_extractor() -> str:
    """Cut mobilenetv2-12 at the global average pool and prepend NHWC + scaling."""
    onnx.utils.extract_model(
        str(BACKBONE_PATH), str(CACHE_DIR / "backbone_features.onnx"), ["input"], [BACKBONE_CUT]
    )
    return str(CACHE_DIR / "backbone_features.onnx")


def extract_features(paths: list[Path], session: ort.InferenceSession, batch: int = 32) -> np.ndarray:
    features = np.empty((len(paths), FEATURE_DIM), dtype=np.float32)
    started = time.time()
    for start in range(0, len(paths), batch):
        chunk = paths[start : start + batch]
        pixels = np.stack([load_pixels(path) for path in chunk]).transpose(0, 3, 1, 2)
        pixels = pixels / 127.5 - 1.0
        pooled = session.run(None, {"input": pixels})[0].mean(axis=(2, 3))
        features[start : start + len(chunk)] = pooled.reshape(len(chunk), FEATURE_DIM)
        done = start + len(chunk)
        if done % 640 < batch or done == len(paths):
            elapsed = time.time() - started
            rate = done / elapsed if elapsed else 0.0
            print(f"  {done}/{len(paths)} features  {rate:.1f} img/s  {elapsed:.0f}s", flush=True)
    return features


def load_features(limit: int | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if FEATURE_CACHE.exists() and limit is None:
        cached = np.load(FEATURE_CACHE)
        print(f"using cached features {FEATURE_CACHE}")
        return {
            "train": (cached["train_x"], cached["train_y"]),
            "test": (cached["test_x"], cached["test_y"]),
        }

    extract_classes()
    session = ort.InferenceSession(build_extractor(), providers=["CPUExecutionProvider"])
    result = {}
    for split in ("train", "test"):
        rows = read_split(split, limit)
        labels = np.asarray([label for _, label in rows], dtype=np.int64)
        print(f"{split}: {len(rows)} images", flush=True)
        result[split] = (extract_features([path for path, _ in rows], session), labels)

    if limit is None:
        np.savez_compressed(
            FEATURE_CACHE,
            train_x=result["train"][0],
            train_y=result["train"][1],
            test_x=result["test"][0],
            test_y=result["test"][1],
        )
        print(f"cached features -> {FEATURE_CACHE}")
    return result


# ---------------------------------------------------------------- linear head


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.sum(probabilities * np.log(probabilities + 1e-9), axis=1) / np.log(probabilities.shape[1])


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Scalar temperature that minimises negative log-likelihood on held-out logits."""
    grid = np.concatenate([np.linspace(0.25, 5.0, 96), np.linspace(5.0, 60.0, 111)])

    def nll(temperature: float) -> float:
        probabilities = softmax(logits / temperature)
        return float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, None)).mean())

    losses = [nll(t) for t in grid]
    best = float(grid[int(np.argmin(losses))])
    # refine on a fine grid around the coarse minimum
    fine = np.linspace(max(0.1, best - 0.5), best + 0.5, 201)
    return float(fine[int(np.argmin([nll(t) for t in fine]))])


def choose_thresholds(
    probabilities: np.ndarray,
    target_coverage: float = 0.80,
) -> tuple[float, float]:
    """Entropy and confidence gates whose OR keeps `target_coverage` of calibration data.

    Both thresholds are quantiles of the same distribution, so one scalar bisection
    is enough: tightening the pair raises abstention monotonically.
    """
    entropy = normalized_entropy(probabilities)
    confidence = probabilities.max(axis=1)

    def coverage_at(quantile: float) -> float:
        entropy_gate = float(np.quantile(entropy, quantile))
        confidence_gate = float(np.quantile(confidence, 1 - quantile))
        kept = (entropy <= entropy_gate) & (confidence >= confidence_gate)
        return float(kept.mean())

    low, high = 0.0, 0.999
    for _ in range(60):
        middle = (low + high) / 2
        if coverage_at(middle) < target_coverage:
            low = middle
        else:
            high = middle
    quantile = (low + high) / 2
    return float(np.quantile(entropy, quantile)), float(np.quantile(confidence, 1 - quantile))


def fit_head(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    l2: float = L2,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted multinomial logistic regression, full-batch Adam. NumPy only."""
    rng = np.random.default_rng(seed)
    samples, dimensions = x.shape
    classes = len(np.unique(y))
    weights = rng.normal(0.0, 0.01, size=(dimensions, classes)).astype(np.float64)
    bias = np.zeros(classes, dtype=np.float64)
    first = (np.zeros_like(weights), np.zeros_like(bias))
    second = (np.zeros_like(weights), np.zeros_like(bias))
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    targets = np.zeros((samples, classes), dtype=np.float64)
    targets[np.arange(samples), y] = 1.0
    weight_sum = sample_weight.sum()

    for epoch in range(1, epochs + 1):
        probabilities = softmax(x @ weights + bias)
        error = (probabilities - targets) * sample_weight[:, None] / weight_sum
        grad_weights = x.T @ error + l2 * weights
        grad_bias = error.sum(axis=0)

        for param, grad, moment, velocity in (
            (weights, grad_weights, first[0], second[0]),
            (bias, grad_bias, first[1], second[1]),
        ):
            moment *= beta1
            moment += (1 - beta1) * grad
            velocity *= beta2
            velocity += (1 - beta2) * grad * grad
            corrected_moment = moment / (1 - beta1**epoch)
            corrected_velocity = velocity / (1 - beta2**epoch)
            param -= lr * corrected_moment / (np.sqrt(corrected_velocity) + epsilon)

        if epoch % 50 == 0:
            loss = -np.log(np.clip(probabilities[np.arange(samples), y], 1e-12, None))
            print(f"  epoch {epoch:>3}  loss {float((loss * sample_weight).sum() / weight_sum):.4f}")

    return weights, bias


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    counts[counts == 0] = 1.0
    balanced = counts.sum() / (len(CLASS_NAMES) * counts)
    return balanced[labels]


# ------------------------------------------------------------------- reporting


def report(
    logits: np.ndarray,
    labels: np.ndarray,
    name: str,
    temperature: float = 1.0,
    thresholds: tuple[float, float] | None = None,
    output_path: Path | None = None,
) -> dict:
    probabilities = softmax(logits / temperature)
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)

    per_class = {}
    f1_values = []
    for index, display in enumerate(DISPLAY_NAMES):
        true_positive = int(confusion[index, index])
        precision_den = int(confusion[:, index].sum())
        recall_den = int(confusion[index, :].sum())
        precision = true_positive / precision_den if precision_den else 0.0
        recall = true_positive / recall_den if recall_den else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[display] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": recall_den,
        }

    total = int(confusion.sum())
    correct = predicted == labels
    top2 = int((np.argsort(probabilities, axis=1)[:, -2:] == labels[:, None]).any(axis=1).sum())
    confidence = probabilities.max(axis=1)
    entropy = normalized_entropy(probabilities)

    # Risk-coverage curve: what accuracy you keep if you drop the least certain images.
    curve = []
    order = np.argsort(entropy)
    for coverage in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00):
        kept = order[: max(1, int(round(coverage * total)))]
        curve.append(
            {
                "coverage": coverage,
                "accuracy": round(float(correct[kept].mean()), 4),
            }
        )

    metrics = {
        "split": name,
        "samples": total,
        "accuracy": round(float(correct.mean()) if total else 0.0, 4),
        "top2_accuracy": round(top2 / total if total else 0.0, 4),
        "macro_f1": round(float(np.mean(f1_values)), 4),
        "temperature": round(float(temperature), 4),
        "class_names": DISPLAY_NAMES,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "confidence": {
            "mean": round(float(confidence.mean()), 4),
            "mean_entropy": round(float(entropy.mean()), 4),
        },
        "risk_coverage": curve,
    }
    if thresholds is not None:
        entropy_gate, confidence_gate = thresholds
        abstained = (entropy > entropy_gate) | (confidence < confidence_gate)
        metrics["abstention"] = {
            "entropy_threshold": round(entropy_gate, 4),
            "confidence_threshold": round(confidence_gate, 4),
            "coverage": round(float((~abstained).mean()), 4),
            "accuracy_when_accepted": round(
                float(correct[~abstained].mean()) if (~abstained).any() else 0.0, 4
            ),
        }
    if output_path:
        output_path.write_text(json.dumps(metrics, indent=2))
        print(f"metrics -> {output_path}")
    return metrics


# ---------------------------------------------------------------------- export


def export_onnx(weights: np.ndarray, bias: np.ndarray, temperature: float, output_path: Path) -> None:
    """Chain NHWC transpose -> scaling -> frozen backbone -> head -> softmax.

    The calibration temperature is folded into the head weights, so the exported
    softmax already produces calibrated probabilities.
    """
    backbone = onnx.load(str(CACHE_DIR / "backbone_features.onnx"))
    weights = weights / temperature
    bias = bias / temperature

    nodes = [
        helper.make_node("Transpose", ["pixels"], ["nchw"], perm=[0, 3, 1, 2], name="to_nchw"),
        helper.make_node("Mul", ["nchw", "scale"], ["scaled"], name="scale"),
        helper.make_node("Sub", ["scaled", "shift"], ["input"], name="shift"),
    ]
    nodes.extend(backbone.graph.node)
    nodes.extend(
        [
            helper.make_node("GlobalAveragePool", [BACKBONE_CUT], ["pooled"], name="pool"),
            helper.make_node("Flatten", ["pooled"], ["features"], name="flatten"),
            helper.make_node("Gemm", ["features", "head_weight", "head_bias"], ["logits"], name="head"),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1, name="softmax"),
        ]
    )

    initializers = [
        numpy_helper.from_array(np.asarray(SCALE, dtype=np.float32), "scale"),
        numpy_helper.from_array(np.asarray(SHIFT, dtype=np.float32), "shift"),
        numpy_helper.from_array(weights.astype(np.float32), "head_weight"),
        numpy_helper.from_array(bias.astype(np.float32), "head_bias"),
    ]
    initializers.extend(backbone.graph.initializer)

    graph = helper.make_graph(
        nodes,
        "phytotrace_mobilenetv2",
        [helper.make_tensor_value_info("pixels", TensorProto.FLOAT, ["batch", IMG_SIZE, IMG_SIZE, 3])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", len(CLASS_NAMES)])],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="phytotrace-local",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    print(f"onnx -> {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")


def write_metadata(
    metrics: dict,
    temperature: float,
    thresholds: tuple[float, float],
    output_path: Path,
) -> None:
    entropy_gate, confidence_gate = thresholds
    metadata = {
        "class_names": DISPLAY_NAMES,
        "source_class_names": CLASS_NAMES,
        "input_size": IMG_SIZE,
        "input_layout": "NHWC",
        "color_space": "RGB",
        "preprocessing": "rgb_0_to_255",
        "output": "softmax_probabilities",
        "entropy_threshold": round(entropy_gate, 4),
        "confidence_threshold": round(confidence_gate, 4),
        "temperature": round(float(temperature), 4),
        "thresholds_calibrated": True,
        "calibration": {
            "temperature_fit": "minimise NLL on a 10% holdout of the training split",
            "threshold_rule": "entropy > threshold OR confidence < threshold",
            "threshold_target_coverage": 0.80,
            "note": "thresholds and temperature were chosen on training-split holdout data only",
        },
        "training": "frozen MobileNetV2 features + weighted logistic regression, local CPU",
        "dataset": "PlantVillage colour images, official train/test split (mohanty/PlantVillage)",
        "test_metrics": {
            key: metrics[key]
            for key in ("samples", "accuracy", "top2_accuracy", "macro_f1")
        },
        "test_abstention": metrics.get("abstention"),
    }
    output_path.write_text(json.dumps(metadata, indent=2))
    print(f"metadata -> {output_path}")


def verify_export(nn_path: Path, image_paths: list[Path], expected: np.ndarray) -> None:
    """The exported graph must reproduce the NumPy head's decisions from raw pixels."""
    session = ort.InferenceSession(str(nn_path), providers=["CPUExecutionProvider"])
    pixels = np.stack([load_pixels(path) for path in image_paths])
    probabilities = session.run(None, {"pixels": pixels})[0]
    if probabilities.shape != (len(image_paths), len(CLASS_NAMES)):
        raise SystemExit(f"exported model returned {probabilities.shape}")
    mismatch = int((probabilities.argmax(axis=1) != expected).sum())
    print(f"export check: {len(image_paths) - mismatch}/{len(image_paths)} predictions identical to NumPy head")
    if mismatch:
        raise SystemExit("exported ONNX does not match the trained head")


# ------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="images per split for a smoke test")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    if not ZIP_PATH.exists():
        raise SystemExit(f"PlantVillage zip missing at {ZIP_PATH}")
    if not BACKBONE_PATH.exists():
        raise SystemExit(f"backbone ONNX missing at {BACKBONE_PATH}")

    features = load_features(args.limit)
    train_x, train_y = features["train"]
    test_x, test_y = features["test"]

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(train_x))
    split_at = int(len(order) * (1 - VALIDATION_FRACTION))
    fit_idx, val_idx = order[:split_at], order[split_at:]

    # Standardise features; the transform is folded into the exported Gemm below.
    mean = train_x[fit_idx].mean(axis=0)
    std = train_x[fit_idx].std(axis=0) + 1e-6

    def scale(x: np.ndarray) -> np.ndarray:
        return ((x - mean) / std).astype(np.float64)

    print(f"\nfitting head: {len(fit_idx)} train / {len(val_idx)} holdout")
    weights, bias = fit_head(
        scale(train_x[fit_idx]),
        train_y[fit_idx],
        class_weights(train_y[fit_idx]),
        epochs=args.epochs,
    )
    val_logits = scale(train_x[val_idx]) @ weights + bias
    temperature = fit_temperature(val_logits, train_y[val_idx])
    thresholds = choose_thresholds(softmax(val_logits / temperature), TARGET_COVERAGE)
    print(f"calibration: T={temperature:.3f}  entropy>{thresholds[0]:.3f}  confidence<{thresholds[1]:.3f}")
    val_metrics = report(
        val_logits, train_y[val_idx], "train holdout", temperature, thresholds
    )
    print(
        f"holdout accuracy {val_metrics['accuracy']:.4f}  coverage {val_metrics['abstention']['coverage']:.3f}"
        f"  accepted accuracy {val_metrics['abstention']['accuracy_when_accepted']:.4f}"
    )

    print(f"\nrefitting on all {len(train_x)} training images")
    weights, bias = fit_head(
        scale(train_x), train_y, class_weights(train_y), epochs=args.epochs
    )

    model_dir = PROJECT_ROOT / "model"
    model_dir.mkdir(exist_ok=True)
    test_metrics = report(
        scale(test_x) @ weights + bias,
        test_y,
        "official PlantVillage colour test split",
        temperature,
        thresholds,
        output_path=model_dir / "metrics.json",
    )
    print(
        f"test accuracy {test_metrics['accuracy']:.4f}  macro F1 {test_metrics['macro_f1']:.4f}"
        f"  top2 {test_metrics['top2_accuracy']:.4f}"
    )
    for name, values in test_metrics["per_class"].items():
        print(f"  {name:<22} P {values['precision']:.3f}  R {values['recall']:.3f}  F1 {values['f1']:.3f}  n={values['support']}")
    print("\nrisk-coverage (drop least certain first):")
    for point in test_metrics["risk_coverage"]:
        print(f"  coverage {point['coverage']:.2f} -> accuracy {point['accuracy']:.4f}")
    print(
        f"\nabstention: keeps {test_metrics['abstention']['coverage']:.1%} of images,"
        f" {test_metrics['abstention']['accuracy_when_accepted']:.4f} accurate on those"
    )

    # Fold standardisation and temperature into the head so the ONNX Gemm needs raw features.
    folded_weight = weights / std[:, None]
    folded_bias = bias - (mean / std) @ weights
    nn_path = model_dir / "mobilenetv2.onnx"
    export_onnx(folded_weight, folded_bias, temperature, nn_path)
    write_metadata(test_metrics, temperature, thresholds, model_dir / "mobilenetv2.json")

    verify_export(
        nn_path,
        [path for path, _ in read_split("test", args.limit)][:8],
        (scale(test_x[:8]) @ weights + bias).argmax(axis=1),
    )


if __name__ == "__main__":
    sys.exit(main())
