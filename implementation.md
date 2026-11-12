# PhytoTrace Implementation and Evaluation Plan

> Historical evaluation record for the earlier six-class model. The current
> discovery-first product direction and 38-class runtime are documented in
> `README.md`; figures below describe the evaluation run that produced them.

## Objective

Build a local, uncertainty-aware plant-leaf image screening prototype. The app must abstain when the model is missing, malformed, or not sufficiently confident. This is not a substitute for expert diagnosis.

## Current state

- Flask upload UI and server-side image validation are implemented.
- Uploads are size-limited and stored under generated names.
- ONNX Runtime inference validates accompanying metadata and expects RGB, 224 x 224, NHWC float input in the 0-255 range. MobileNetV2 scaling and the calibration temperature are embedded in the exported graph; output is softmax probabilities.
- A trained 6-class model ships in `model/`, produced by `train_local.py` on CPU.
- Temperature scaling and abstention thresholds were selected on a held-out slice of the training split. They are no longer provisional guesses.

## Method

1. Use the PlantVillage colour images for six classes: potato and tomato, each with early blight, late blight and healthy. Early and late blight are visually similar on both crops, so part of the class boundary is genuinely ambiguous. That is what the abstention rule is for.
2. Split with the dataset's official `color_train.txt` / `color_test.txt` files rather than a random 80/20 of one pool. Image-level random splits of PlantVillage can place near-duplicate leaves on both sides. The official split is still PlantVillage, but it is the dataset authors' split and it is the same for every run.
3. Extract a 1280-dimensional feature per training image from a frozen, ImageNet-pretrained MobileNetV2. Fit a class-weighted multinomial logistic regression head on those features. No GPU, no TensorFlow, no backbone fine-tuning.
4. Hold out 10% of the training split for calibration. Fit the temperature by minimising negative log-likelihood on that holdout. Choose the entropy and confidence thresholds on the same holdout, targeting 80% coverage. The test split is not used for either step.
5. Export the transpose, scaling, frozen backbone, head, temperature and softmax as a single ONNX graph, then assert that the exported graph reproduces the NumPy head's predictions.

## Results

Test split, 1,328 images: accuracy 95.78% (95% CI 94.73–96.91), macro F1 0.9507 (CI 0.9333–0.9666), top-2 accuracy 99.47%.

At the shipped thresholds the model accepts 74.9% of images and is 99.60% accurate on them (CI 99.20–99.90). Without abstention it is 95.78% accurate on everything. Risk-coverage at 80% coverage is 99.44%, at 90% it is 98.83%. AURC is 0.0034.

On 9,381 images from the 32 plant species excluded from training, the calibrated model abstains on 77.77% (CI 76.94–78.61). The 22.23% of unseen images that are still accepted are wrong by construction; mean confidence on them is 0.9845.

**The abstention figure is mostly a threshold-selection effect, not a calibration effect.** The original comparison put the calibrated model at its tuned thresholds against the raw model at placeholder thresholds (0.85/0.60), which changed two things at once. Re-tuning the raw model's threshold to the same 80% in-distribution coverage already abstains on 70.56% of unseen images; the fitted temperature adds 1.89 points, to 72.45%. Temperature scaling still earns its place — ECE 0.0262 → 0.0177, MCE 0.4081 → 0.3546, and +0.8 to +1.3 AUROC on every out-of-distribution score — but it is responsible for roughly 3% of the headline gap, not all of it.

Other measured findings:

- **Head ablation (frozen features, 80% coverage protocol):** MLP(128) 96.69% accuracy / 20.95% of unseen images answered; logistic regression (shipped) 95.78% / 27.55%; linear SVM 96.23% / 19.50%; random forest 90.51% / 14.28%; k-NN k=5 91.72% / 50.36%. Accuracy spans 6 points across heads; unseen-species abstention spans 36. The head is not the binding constraint, and in-distribution accuracy does not predict refusal behaviour.
- **OOD scores:** MSP, predictive entropy and energy all reach AUROC ≈ 0.83 against unseen species. Energy does *not* beat MSP here (AURC 0.0071 vs 0.0034), contrary to Liu et al. (2020) — the OOD set is near-distribution rather than far.
- **Robustness (100 test images, synthetic corruption):** brightness and JPEG compression barely move accuracy or abstention. Blur is not handled gracefully: abstention rises to 80% at radius 2 but *falls* to 65% at radius 4 while accuracy drops to 35% and accuracy-on-accepted falls to 40%. The rule measures unusualness, not ignorance.

`Potato healthy` has only 32 test images. Its per-class recall should be read as indicative, not precise.

## Evaluation still required before claiming real-world readiness

- Evaluate on an independent, real-world dataset such as PlantDoc, with class mapping documented. Specified in the upgrade plan, not executed; the images were never acquired.
- Test out-of-scope subjects that are not leaves at all. The unseen-species test covers a different *class*, not a different *kind of image*; a photograph of a hand or a phone screen remains untested and is the most likely failure mode. The blur result above is evidence that this gap is real.
- Record dataset versions, split procedure, model artifact hashes, runtime versions, and reproducible seeds. Hashes, runtime versions and all seeds now appear in `report/REPORT.md` §9; dataset version pinning is still manual.
- Recalibrate if the class set, backbone, or training data changes.

## Operational constraints

- Local Flask server binds to `127.0.0.1`; debug mode is off unless explicitly enabled with `FLASK_DEBUG=1`.
- Uploaded files are limited to 5 MB and validated as decodable images.
- Keep dataset archives, uploads, and model binaries out of version control.
- Never describe PlantVillage validation numbers as field performance.
