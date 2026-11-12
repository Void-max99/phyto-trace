# PhytoTrace — Research Discovery Surface

Research tooling for surfacing unusual plant specimens, organizing them for expert review, and exploring recurring patterns.

PhytoTrace keeps its existing classifier, evaluation, and local inference workflow, and adds a discovery surface on top. The frozen MobileNetV2 model (38 classes, 14 crops) defines the known-class reference; the triage policy is the decision gate, while anomaly scoring and the review queue support researcher-led discovery. Anomaly scores indicate visual distance from cached training features, not proof of a new disease or a guarantee that an unfamiliar specimen will be caught.

The discovery workflow supports single-image review, bulk anomaly scanning, clustering, and JSON/CSV export without removing the existing inference and analysis tools.

## Why this project feels stronger than a basic model repo

- Calibrated uncertainty, not raw accuracy alone
- Explicit abstention rule with coverage-aware thresholds
- Shipped ONNX inference path and model metadata
- Clean local Flask web app with sample images, live inference, webcam capture, occlusion heatmaps, and model variant switching
- Test coverage
- Honest evaluation against unseen species and robustness stress tests, including a selective-classification audit

## Current status

| Attribute | Value |
|---|---|
| Project focus | Research discovery surface on top of existing inference |
| Version | 2.1.0-s |
| Tests | 26 / 26 passing |
| Training | Local CPU training scripts; frozen backbone shipped |
| Inference | Local CPU, no GPU required |
| Model | `model/mobilenetv2_full.onnx` (38 classes / 14 crops, fused image-in → softmax-out) |
| Variants | `mobilenetv2_full.onnx` (default 38-class model), `mobilenetv2.onnx` (6-class LR), `mobilenetv2_mlp.onnx` (6-class MLP) |

## Quick start

```bash
cd /home/void/phytotrace
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Project health audit
phytotrace status

# Inspect cached data manifest
phytotrace inspect

# Print effective config
phytotrace config

# List available model registry entries and deployment variants
phytotrace registry

# Batch-predict a folder of images
phytotrace batch ./data/custom_images --output ./predictions.csv

# Benchmark a labeled dataset with class subfolders
phytotrace benchmark ./data/test_set

# Sweep entropy/confidence thresholds and surface the best operating points
phytotrace sweep ./data/test_set --output-json ./reports/sweep.json --output-html ./reports/sweep.html

# Start the local inference and discovery server
python -m phytotrace.api
# open http://127.0.0.1:5000
```

If you are running directly from source without installing:

```bash
python -m phytotrace status
python -m phytotrace.api
```

The package is intentionally dry-run by default: no training, no downloads, and no runtime mutation on import.

## Discovery-first workflow

PhytoTrace is a researcher-facing discovery surface layered over the existing plant-disease model. The triage policy is the gate for whether an image can be accepted or must be reviewed:

- **Triage policy** — high-risk classifications escalate; anomaly and calibrated uncertainty route cases to review rather than silently accepting them.
- **Anomaly gate** — compares specimen features with cached training features; it is a useful novelty signal, not a validated detector of new diseases.
- **Review queue** — preserves candidate images and context for researcher follow-up, clustering, and export.
- **Discovery tools** — bulk scans, cosine-distance clustering, and JSON/CSV exports help researchers inspect repeated patterns.
- **Existing capabilities remain** — 38-class inference, hierarchical abstention, conformal prediction, calibrated evaluation, robustness checks, CLI, and local deployment.

Clusters are hypotheses for expert investigation, not automatic diagnoses or newly established disease classes. Evaluation remains limited to the documented datasets; field performance is unverified.

## Class surface (demonstrable capability)

| Attribute | Value |
|---|---|
| Total classes | 38 |
| Total crops | 14 |
| Crops | Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato |
| Largest crop | Tomato — 10 classes |
| Smallest crops | Blueberry, Raspberry, Soybean, Squash — 1 class each |

Every class is exposed through `GET /classes` and `phytotrace classes`. Examiners can click through all 38 classes in the web UI, not just a cherry-picked subset.

## Key results

| Metric | Value |
|---|---|
| Test accuracy (38-class) | 96.26% |
| Top-2 accuracy (38-class) | 99.18% |
| Macro F1 (38-class) | 0.9487 |
| Accuracy when accepted (calibrated gate) | 99.80% |
| Coverage at shipped thresholds | 79.9% |
| Unseen-species abstention (calibrated) | 77.77% |

The important point is not just that the model is accurate — it is that it behaves responsibly under uncertainty, across 14 crops and 38 classes, which is the part that gives the project a stronger real-world story.

## New-disease discovery workflow

The model knows 38 classes across 14 crops. When it meets a specimen it
doesn't recognise — high anomaly score *and* abstention — it routes to a
five-question questionnaire instead of forcing a wrong label:

1. What crop or plant species is this?
2. What symptoms do you see?
3. Where on the plant does it appear?
4. When did symptoms start?
5. Any treatments applied?

Answers are saved to the review queue (`$PHYTOTRACE_CACHE/review_queue.json`,
defaulting to `~/.cache/phytotrace/review_queue.json`) with the image,
prediction, and anomaly score. Bulk scans enqueue anomalous cases that the
triage policy sends to review. An offline clustering script
(`report/discover_clusters.py`) groups similar flagged specimens using DBSCAN
in backbone feature space. Clusters of ≥2 similar unknowns become candidates
for a new disease class in the next model version.

Endpoints:
- `POST /review` — save questionnaire answers
- `GET /review-queue` — list pending reviews
- `POST /review/<id>/status` — update review status (pending/reviewed/confirmed-new/rejected)

## Honest limitations

- **Dataset is lab-collected.** PlantVillage images are single leaves on uniform backgrounds, photographed outdoors with one camera. The model is not validated on field photos, blurry inputs, or non-leaf images. If you point it at a hand, soil, or a screenshot, behavior is untested.
- **Unseen-species abstention is imperfect.** On 9,381 images of 32 species the model never trained on, it abstains 77.77%. However, a selective-classification audit found no tested score (MSP, entropy, feature proximity, Mahalanobis) reliably separates unseen species — OOD AUROC hovers near 0.47. The gate measures unusualness, not ignorance.
- **Abstention is dominated by threshold choice.** Temperature scaling improves calibration metrics (ECE 0.0262→0.0177) but contributes only ~2 points to abstention; threshold selection does ~63 points. This matches current selective-classification literature.
- **Fine-tuning was aborted.** End-to-end fine-tuning of MobileNetV2 was started but stopped because it did not improve on the frozen-backbone model (96.26%). Fine-tune checkpoints exist under `model/` but are not shipped.
- **Discovery cache is local data.** The anomaly gate requires `backbone_features.onnx` and `features_full.npz` under `$PHYTOTRACE_CACHE` (default `~/.cache/phytotrace`). These large cache files are not shipped. Without them, inference remains available, `/health` reports `anomaly_ready: false`, and `/scan` returns an explicit unavailable response.

## Project structure

```text
phytotrace/               # package core
  __init__.py             # version and status metadata
  __main__.py             # `python -m phytotrace`
  api/                    # Flask inference and discovery API
  cli/                    # CLI for status/config/inspection
  config.py               # validated TOML-backed config
  core/                   # preprocessing, inference, calibration
  data/                   # dataset manifest and loading
  discovery/              # clustering and export
  models/                 # anomaly detection and classifier heads
  policy/                 # triage and conformal decisions
  eval/                   # metrics + OOD + robustness analysis
  training/               # cloud-training stub
report/                   # figures, model card, writeup
model/                    # shipped ONNX model and calibration metadata
tests/                    # 26 passing checks
Dockerfile                # container build
docker-compose.yml        # local deployment
pyproject.toml            # packaging metadata and CLI entry point
requirements.txt          # dependency pinning
```

## API and interface

The app exposes a production-style inference endpoint:

| Route | Purpose |
|---|---|
| `GET /` | UI dashboard |
| `GET /health` | engine and anomaly-detector readiness + metadata |
| `GET /samples` | sample image catalog |
| `GET /classes` | all 38 classes grouped by 14 crops — the known world |
| `POST /predict` | single-image diagnosis + triage decision |
| `POST /scan` | bulk anomaly scan — flag unknowns across many images |
| `GET /clusters` | DBSCAN clustering of flagged unknowns by visual similarity |
| `POST /review` | save questionnaire answers and specimen context |
| `GET /review-queue` | list saved researcher reviews |
| `POST /review/<id>/status` | update a review status |
| `GET /export` | export review queue as JSON / CSV |
| `GET /export/clusters` | export clustering results as JSON / CSV |

This adds immediate demo value and shows the project is more than a model artifact sitting in a folder.

## Deployment and execution

```bash
# Docker build
docker build -t phytotrace .

docker run -p 5000:5000 phytotrace

# Or with Compose
docker compose up
```

The Compose cache volume starts empty. Populate it with the two discovery cache
files above before expecting anomaly scans or clustering to work in Docker.

## Safety and constraints

- No training on import
- No network calls at inference time
- Missing model files fail gracefully with explicit messages
- Heavy model work is intentionally triggered by explicit commands, not side effects

## Best angle for presentations

Say this:

> I extended a plant-disease inference project with a researcher-facing discovery workflow. A triage policy gates acceptance, and anomaly scoring, a review queue, clustering, and exports help experts inspect unusual specimens. The anomaly score is only a novelty signal; field performance and new-disease detection are not yet validated.

That framing makes the project sound substantially more mature and differentiated than a generic image classifier.

## References

See `report/REPORT.md` and `report/MODEL_CARD.md` for the deeper analysis, figures, and evaluation details.
