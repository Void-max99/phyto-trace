# PhytoTrace — full handoff brief

Written 01 Nov 2026 for an external reviewer (human or AI). Everything below is
the current state of the project, including the parts that do not flatter it.
Numbers are from files in `model/*.json`, not from memory.

> Historical baseline: this brief describes the earlier six-class model and
> triage-focused project framing. The current 38-class discovery-first
> direction and working-tree status are documented in `README.md`; use this
> file only for the historical measurements and constraints below.

---

## 1. What it is

A 6-class plant leaf disease classifier for potato and tomato, shipped as a
fused ONNX graph, served by a local Flask app, with a calibrated abstention
rule: the model returns a class only when its own uncertainty is below a
threshold, and otherwise flags the image for human review.

Classes: potato early blight, potato late blight, potato healthy, tomato early
blight, tomato late blight, tomato healthy.

## 2. Origin and constraints

- Final-year BCA (Bachelor of Computer Applications) project, Indian university.
- Zero budget. Hardware is an i3-1305U, 16 GB RAM, integrated graphics.
- Hard constraint from the owner: **no local training** at this stage. Training
  is treated as a cloud step that has not been executed since the shipped model
  was produced. All analysis since then runs on cached features.
- The owner explicitly does not want over-engineering, and does not want the
  project to read as AI-generated boilerplate.

## 3. Dataset

- PlantVillage colour images, `mohanty/PlantVillage` on HuggingFace.
- Official split files `color_train.txt` / `color_test.txt`, not a random 80/20.
  Rationale: PlantVillage contains near-duplicate leaves; a random split can
  place near-duplicates on both sides.
- 6 classes only, chosen because early vs late blight is visually confusable on
  both crops — the abstention rule has a real job.
- Train 5,324 images, test 1,328 images.
- `Potato healthy` is tiny: 120 train / 32 test.
- An additional 9,381 images from the 32 PlantVillage species excluded from
  training are used as an out-of-distribution set.
- Cache lives at `~/.cache/phytotrace/` (data.zip 2.2 GB, features.npz 29 MB,
  unseen_features.npz 41 MB).

## 4. Method

1. Frozen ImageNet-pretrained MobileNetV2 as a feature extractor (1280-d pooled
   features). No fine-tuning. No GPU.
2. Class-weighted multinomial logistic regression on those features.
3. Temperature scaling: T fitted by minimising NLL on a 10% holdout of the
   *training* split (533 images). Test split never used for this.
4. Abstention thresholds (entropy and confidence) chosen on the same holdout,
   targeting 80% coverage.
5. The transpose, `/127.5 - 1` scaling, frozen backbone, head, temperature and
   softmax are exported as **one fused ONNX graph**: NHWC RGB 0–255 in,
   softmax probabilities out. 8.87 MB. No separate head file exists or is
   needed.

Full local retrain takes ~100 s on CPU. That path (`train_local.py`) exists and
works, but has not been run since the shipped artifact was produced.

## 5. Results (all measured, test split n=1,328)

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 95.78% | 94.73–96.91 |
| Macro F1 | 0.9507 | 0.9333–0.9666 |
| Top-2 accuracy | 99.47% | — |
| Coverage at shipped thresholds | 74.92% | 72.67–77.33 |
| Accuracy when accepted | 99.60% | 99.20–99.90 |
| Unseen-species abstention (calibrated) | 77.77% | 76.94–78.61 |

Calibration: ECE 0.0262 → 0.0177, MCE 0.4081 → 0.3546, Brier 0.0664 → lower
after temperature scaling.

OOD scores (MSP / entropy / energy) all reach AUROC ≈ 0.83 against the 32 unseen
species. Energy does **not** beat MSP here, contrary to Liu et al. 2020 — the OOD
set is near-distribution, not far.

**Head ablation (frozen features, gate fixed at 80% ID coverage):**

| Head | Accuracy | Unseen answered | Fit time |
|---|---|---|---|
| MLP(128) | 96.69% | 20.95% | ~4 s |
| Linear SVM | 96.23% | 19.50% | — |
| LR refit | 96.23% | 24.26% | 2.7 s |
| Logistic (shipped) | 95.78% | 27.55% | — |
| kNN k=5 | 91.72% | 50.36% | — |
| Random forest | 90.51% | 14.28% | — |

Conclusion drawn: the head is not the binding constraint; in-distribution
accuracy does not predict refusal behaviour. The MLP beats the shipped head on
both axes and is deliberately **not** shipped, because swapping it would
invalidate every number in the report.

**Robustness (100 test images, synthetic corruption):**

| Corruption | Accuracy | Abstain rate | Acc. when accepted |
|---|---|---|---|
| none | 97% | 19% | 100% |
| gaussian noise σ=32 | 57% | 45% | 78.2% |
| blur radius 1 | 86% | 57% | 95.4% |
| blur radius 2 | 56% | 80% | 80.0% |
| blur radius 4 | 35% | 65% | 40.0% |

The blur result is the important one: abstention *falls* from 80% to 65% as
blur worsens while accuracy collapses to 35%. Direct evidence that the rule
measures unusualness, not ignorance.

**Error structure (56 test errors):**

- 26 are tomato early ↔ late (12 one way, 14 the other). 46% of all errors.
- 17 cross crops, 39 stay within a crop.
- 13 are healthy ↔ diseased confusions.

## 6. What is actually shipped

```
phytotrace/
  config.py         validated dataclass config, TOML-backed
  data/             loader + preprocessing (resize, pad, normalise)
  models/
    backbone.py     MobileNetV2 feature extractor wrapper
    heads/          logistic, mlp, svm, knn, rf + registry
    calibration.py  temperature scaling, ECE/MCE/Brier
    inference.py    FusedInferenceEngine (shipped) + InferenceEngine (ablation)
  evaluation/       metrics, OOD scores, robustness, report generation
  training/         cloud stub — raises if asked to execute locally
  app.py            Flask: /, /health, /samples, /predict
  cli.py            Rich CLI: status, config, inspect, train, evaluate
  templates/        single-file web console
  static/samples/   3 sample specimens
train_local.py      the only training path, ~100 s CPU
train_heads_ablation.py
evaluate.py         ID + unseen-species evaluation
evaluate_robustness.py
report/REPORT.md    full academic writeup, figures 1–6
report/MODEL_CARD.md
model/              onnx + metadata + 7 result JSONs
tests/              32 passing
```

Roughly 2,800 lines of Python in package + tests.

The Flask app serves a single-page console with drag-drop, clipboard paste,
sample specimens, a confidence ring, entropy gauge, full 6-class probability
distribution, and a clickable history strip. Verified end to end: prediction
26–70 ms on CPU, both accept and abstain states render correctly.

## 7. Honest weaknesses

These are the things a reviewer should press on. None are hidden in the repo.

1. **It is not better than Google Lens at recognition.** Six classes, lab
   images, uniform backgrounds. A generalist model with orders of magnitude more
   data wins on raw classification. Any claim of "better plant disease
   detection" is false.

2. **PlantVillage is not field data.** Every image is a single leaf on a plain
   background, studio-lit. The model has never been evaluated on a phone photo
   of a plant in soil. This is the single biggest credibility gap and it is
   documented, not closed.

3. **The headline abstention result was initially overstated and has been
   corrected.** The original claim compared calibrated thresholds against
   placeholder thresholds, changing two things at once. At matched 80% ID
   coverage the raw softmax already abstains on 70.56% of unseen species;
   calibrated reaches 72.45%. **Threshold selection does ~63 points of the work,
   temperature ~2.** Temperature still improves ECE and every OOD score by
   +0.8–1.3 AUROC, but it is not the source of the abstention result. The
   README, REPORT.md, MODEL_CARD.md and implementation.md all carry the
   corrected framing.

4. **No git repository.** Files are the history. This is a real risk for a
   submission.

5. **Reference list in REPORT.md was written from memory** and has not been
   verified against publisher records.

6. **Legacy cruft.** `src/app.py`, `src/inference.py`, `static/templates/` are
   the pre-refactor application and are now dead code. Root-level
   `evaluate.py` / `evaluate_robustness.py` predate the package and duplicate
   some of it.

7. **No cross-dataset evaluation.** PlantDoc was never acquired. There is no
   number for field-image performance, good or bad.

8. **No Grad-CAM or any interpretability output.** ONNX Runtime has no
   autograd; adding it would require PyTorch or TensorFlow, which the project
   deliberately avoids.

## 8. The novelty problem, stated plainly

The owner asked the correct question: *"this gives the same output as Google
Lens, so what is the novelty?"*

Current answer, which is weak: calibrated abstention. A threshold is not a
contribution on its own.

Proposed reframe, not yet implemented: **the errors are concentrated on one
boundary, so answer at the granularity the model can actually support.**

Derived from the existing confusion matrix, before any new code: merging early
and late blight within each crop recovers 26 of the 56 errors. That alone takes
accuracy from 95.78% to **97.74%** without touching the model. The system would
answer *"tomato blight — early or late, cannot separate"* instead of either
guessing or refusing entirely.

Two additions under consideration:

- **Partial / hierarchical abstention** — accept at the coarsest level the
  posterior supports, rather than accepting or rejecting the whole answer.
  Cheap, runs on cached features, no training.
- **Split conformal prediction** — replace the holdout-tuned threshold with
  prediction sets that carry a distribution-free coverage guarantee. Also cheap,
  also no training.

Not under consideration: a disease encyclopedia. The web already does that
better, and competing on content is a losing move.

## 9. Questions worth putting to a reviewer

1. Is hierarchical abstention a legitimate contribution at undergraduate level,
   or is it a UX wrapper around a confusion matrix?
2. Does split conformal add real rigour here, or is it decoration on top of an
   already-calibrated model?
3. Is the corrected calibration framing (threshold selection dominates,
   temperature is a small secondary gain) a strength — honest science — or does
   it deflate the project's only distinctive claim?
4. Given that the model is 6-class and lab-only, is there any framing under
   which this is more than a competent course project?
5. What is the highest-value single addition, given ~2 hours of work and no
   training?
