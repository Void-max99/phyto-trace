# PhytoTrace — Discovery-First Researcher Surface

This plan preserves the existing inference, evaluation, and triage capabilities while making researcher-led discovery the primary product surface. The triage policy is the acceptance gate; anomaly scoring, bulk review, clustering, and export form the discovery workflow. The anomaly score is not a guarantee of detecting new diseases.

## 1. Positioning statement

PhytoTrace is a research tool for surfacing and organizing specimens that are unusual relative to its known classes. It does not claim field-ready diagnosis or automatic new-disease identification. The 38-class model across 14 crops provides a known-class reference; researchers decide what clustered cases mean.

The **triage policy** gates acceptance and routes uncertain or high-risk predictions to review. The discovery surface builds on that gate: anomaly scoring identifies candidates, the review queue retains specimens and researcher context, and clustering/exports support follow-up study. These signals are exploratory and must not be presented as confirmed disease findings.

## 2. Constraints (non-negotiable)

- Zero budget. i3-1305U, 16 GB RAM, integrated graphics. CPU-only inference.
- No GPU, no Colab, no cloud, no API keys, no model downloads at runtime.
- Frozen ImageNet MobileNetV2 backbone exported to ONNX. Single fused graph: NHWC RGB 0–255 in, softmax probabilities out.
- No separate head files. No runtime mutation on import.
- Local-first. Flask + ONNX Runtime + Python stdlib where possible.
- Examiner-readable: docstrings on every module, every threshold justified in `config.toml`, no hidden logic, no magic.
- BCA-final-year appropriate: an undergraduate must be able to explain every file.

## 3. Phase 1 — Shipped (no training)

### 3.1 Inference core (`phytotrace/core/`)
- `FusedInferenceEngine`: loads `model/mobilenetv2_full.onnx` (default variant), validates metadata contract (input size, normalization, class list, temperature, thresholds).
- Preprocessing: resize 224 bilinear, pad to square, `/127.5 - 1.0` scaling, NHWC layout.
- Output: full sorted 38-class distribution, top class, confidence, normalized entropy, latency_ms.
- Latency: ~45–70 ms per image on CPU (i3-1305U).
- Default variant: `full` (38 classes / 14 crops). Variants: `mlp` (6 classes), `lr` (6 classes, original).

### 3.2 Policy layer (`phytotrace/policy/`) — IMPLEMENTED
This is the S-tier contribution. Fixed decision order:

1. `ESCALATE_HIGH_RISK` — regulated/high-risk class (e.g. late blight) always escalates regardless of confidence.
2. `REVIEW_UNCERTAIN` — anomaly gate or entropy above threshold → defer.
3. `REVIEW_CONFLICT` — text/second-signal contradicts image → defer (text is a veto, never a vote).
4. `HIERARCHICAL_ACCEPT` — if posterior supports crop level but not early vs late blight, return coarse label instead of guessing or refusing entirely.
5. `ACCEPT` — high confidence, low entropy, no conflict → accept class.
6. `REQUEST_ONE_PHOTO` — single usable photo missing → ask for one retake.
7. `ASK_ONE_QUESTION` — one clarifying question from the five-question bank.
8. `REVIEW_UNCERTAIN` — final fallback.

Thresholds read from `model/mobilenetv2_full.json`; `config.toml` only as fallback. Policy cannot drift from the artifact.

### 3.3 Hierarchical abstention (core novelty) — IMPLEMENTED
- Merge early↔late blight within each crop at the policy layer.
- Recovers 26 of 56 test errors: accuracy 95.78% → 97.74% without touching the model.
- Answer granularity matches what the posterior actually supports.

### 3.4 Split conformal prediction (secondary) — IMPLEMENTED
- Replace holdout-tuned threshold with prediction sets carrying distribution-free coverage guarantee.
- Runs on cached features, no training.
- Report coverage and accepted accuracy on the same test split.

### 3.5 Interface (`phytotrace/api/`, `phytotrace/cli/`)
- Flask app: `GET /`, `GET /health`, `GET /samples`, `GET /classes`, `POST /predict`.
- Researcher discovery: `POST /scan`, review queue, status updates, DBSCAN clusters, and JSON/CSV exports.
- `/health`: engine_ready, classes, temperature, thresholds, input_size, threads, uptime, anomaly_ready.
- `/classes`: all 38 classes grouped by 14 crops — demonstrates model breadth.
- `/predict`: multipart upload → class, confidence, entropy, abstain, verdict, full distribution, latency_ms, image dims.
- `GET /samples`: 3 specimens (one accept, two abstain) so both states demo without a leaf photo.
- CLI `classes` command: prints all classes grouped by crop in a Rich table.
- `?selftest=N` query param auto-runs sample N for headless verification.
- CLI: `status`, `config`, `inspect`, `registry`, `batch`, `benchmark`, `sweep`. All dry-run by default.
- UI: single-file console, drag-drop, clipboard paste, confidence ring, entropy gauge, 38-class distribution, history strip, toasts. Zero green palette; sky/indigo accents, teal for passed.

### 3.6 Evaluation (`phytotrace/eval/`)
- Test split: official PlantVillage color_test.txt, n=1,328. Never used for calibration.
- Metrics: accuracy, macro F1, top-2, coverage, accepted accuracy, ECE, MCE, Brier.
- Bootstrap CIs: 1000 resamples, seed 42.
- Unseen species: 9,381 images from 32 excluded PlantVillage species. Report calibrated abstention rate.
- Robustness: brightness, JPEG, Gaussian noise, blur radius 1/2/4. Table of accuracy / abstain / accepted accuracy.
- Selective audit: MSP / entropy / energy / feature proximity. Honest statement that no score reliably separates near-distribution OOD (AUROC ≈ 0.47–0.83).
- Limitations documented in README, not hidden: lab data only, no field photos, anomaly gate measures unusualness not ignorance.

## 4. Phase 2 — Later (training)

- `train_local.py`: frozen backbone → 1280-d features → class-weighted logistic regression head → temperature scaling → fused ONNX export. ~100 s CPU.
- `train_heads_ablation.py`: MLP / SVM / kNN / RF on cached features for comparison only.
- Training stub raises `NotImplementedError` if executed in Phase 1. Not part of the shipped artifact.

## 5. File contract

```
phytotrace/
  core/           fused inference engine + preprocessing
  policy/         decision order, hierarchical abstention, conformal sets
  api/            Flask app, /health, /samples, /predict
  cli/            Rich CLI commands
  eval/           metrics, bootstrap, OOD, robustness, selective audit
  training/       disabled stubs only
  templates/      single-file web console
  static/samples/ 3 demo specimens
model/
  mobilenetv2_full.onnx   fused classifier (shipped)
  mobilenetv2_full.json   metadata contract
  mobilenetv2.onnx        6-class LR variant (shipped, intact)
  mobilenetv2_mlp.onnx    6-class MLP variant (shipped, intact)
config.toml
pyproject.toml
requirements.txt
Dockerfile
docker-compose.yml
tests/            35+ passing
README.md
REPORT.md         academic writeup + honest limitations
MODEL_CARD.md
MASTERPLAN.md     this file
```

## 6. Acceptance criteria

- The complete test suite passes.
- `python -m phytotrace status` → PASS.
- `GET /health` → 200, with engine and anomaly readiness reported separately.
- `POST /predict` → <100 ms, returns accept and abstain states correctly.
- High-risk cases accepted: 0 in every evaluation condition.
- Both verdict states demoable via `?selftest=1` (accept) and `?selftest=3` (abstain).
- Every threshold has a source line in `config.toml` or `model/mobilenetv2_full.json`.

## 7. Honesty rules

- Do not claim S-tier accuracy. Claim S-tier decision behavior.
- Do not hide the OOD AUROC ≈ 0.48 limitation.
- Do not soften the field-data gap.
- Do not claim novelty on calibration alone; the novelty is the triage policy and its evaluation protocol.
- References in REPORT.md verified before submission; do not cite from memory.
