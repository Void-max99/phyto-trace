# MASTER ENGINEERING IMPROVEMENT PROMPT — PHYTOTRACE

## Project
**PhytoTrace v2.1.0-s**

**Research direction:** Uncertainty-Aware Plant Disease Recognition and Researcher-Assisted Anomaly Triage.

## Your role

Act as a senior Machine Learning Engineer, Computer Vision Researcher, MLOps Engineer, and Software Architect.

Your task is to inspect, improve, test, and strengthen the EXISTING PhytoTrace project.

You are working on an existing implementation, not creating a new project from scratch.

Your priority is to improve actual functionality, scientific reliability, model evaluation, deployment readiness, and research credibility.

**IMPORTANT: Work on the source code. Do not treat this as a documentation-only task.**

---

# 1. NON-NEGOTIABLE RULES

1. Inspect the entire existing repository before changing anything.
2. Understand the existing architecture, dependencies, inference pipeline, models, APIs, evaluation scripts, and tests.
3. Preserve working functionality.
4. Do not replace the existing MobileNetV2 architecture without measurable justification.
5. Do not remove existing features simply to simplify development.
6. Do not invent experimental results, datasets, citations, model capabilities, or performance improvements.
7. Never report a metric unless it has actually been calculated from valid experimental data.
8. Do not fabricate missing cache files, model weights, training data, or evaluation results.
9. Do not silently change model labels, preprocessing, thresholds, or calibration parameters.
10. Avoid unnecessary dependencies and excessive computational requirements.
11. Design for local CPU execution without requiring a GPU.
12. Keep every significant modification testable and reproducible.
13. Do not introduce network calls into the normal inference workflow.
14. Never present anomaly detection as proof of a new disease.
15. Do not merely explain what needs improvement. Implement improvements wherever the available repository and data permit.

When a requirement cannot be completed because data, dependencies, or experimental resources are missing, clearly identify the blocker and implement the necessary infrastructure where possible.

---

# 2. CURRENT PROJECT BASELINE

Treat these as README-reported values, not independently verified facts.

- 38 known disease classes.
- 14 crops.
- MobileNetV2 frozen-backbone architecture.
- ONNX inference.
- Local Flask application.
- Hierarchical abstention.
- Conformal prediction.
- Temperature scaling and calibration.
- Confidence and entropy-based triage.
- Feature-based anomaly scoring.
- Researcher review queue.
- DBSCAN clustering.
- JSON/CSV export.
- 26 reported passing tests.

Reported evaluation:

| Metric | Existing result |
|---|---:|
| Test accuracy | 96.26% |
| Top-2 accuracy | 99.18% |
| Macro F1 | 0.9487 |
| Accepted prediction accuracy | 99.80% |
| Coverage | 79.9% |
| Unseen-species abstention | 77.77% |
| OOD AUROC | Approximately 0.47 |

The most important existing weakness is that the tested OOD scoring methods do not reliably discriminate unseen species.

Another important limitation is that the discovery functionality depends on two cache files that are not shipped:

- `backbone_features.onnx`
- `features_full.npz`

Field-image generalisation is also unverified.

Do not conceal these problems.

---

# 3. PRIORITY ONE — SCIENTIFICALLY RELIABLE OOD DETECTION

This is the most important research improvement.

The current OOD AUROC of approximately 0.47 indicates that the existing tested scores are not reliable discriminators in the evaluated setting.

Do not attempt to hide this through threshold manipulation.

## Required work

### A. Audit existing OOD implementation

Inspect:

- Feature extraction.
- Backbone representations.
- Feature normalisation.
- Cosine-distance calculations.
- Mahalanobis scoring.
- Maximum softmax probability.
- Predictive entropy.
- Existing anomaly thresholds.
- OOD evaluation scripts.
- Data partitioning.
- Calibration methodology.

Identify implementation errors, methodological weaknesses, and limitations separately.

### B. Investigate alternative scoring methods

Implement and evaluate suitable candidates where technically justified:

- Maximum Softmax Probability.
- Predictive entropy.
- Energy-based scoring.
- Cosine similarity to class prototypes.
- Nearest-neighbour feature distance.
- Mahalanobis distance.
- Feature-space density estimation.
- Ensemble disagreement, only if computationally practical.
- Input or representation-level distance measures.

Do not assume that a newer technique is automatically better.

Keep the existing methods as baselines.

### C. Evaluate properly

For every tested method, calculate applicable metrics:

- AUROC.
- AUPR-OOD.
- AUPR-ID.
- FPR at 95% TPR.
- Detection accuracy at documented operating thresholds.
- Per-class OOD behaviour where possible.
- Score distributions.
- ID/OOD separation visualisations.

Report results independently for each evaluated OOD dataset.

Do not choose thresholds on the final test set.

Use separate development/calibration and test partitions.

### D. Improve the scoring architecture

Design the scoring system so that:

1. Each method can be evaluated independently.
2. Scores and thresholds are configurable.
3. Missing methods fail gracefully.
4. Thresholds are derived from development data.
5. The system records which detector generated a decision.
6. Every decision can be reproduced.
7. Existing inference functionality remains available.

If the available evidence does not support reliable OOD detection, preserve that limitation instead of forcing a positive result.

**Success criterion:** A reproducible OOD evaluation framework with honest comparisons and no fabricated improvement claims.

---

# 4. PRIORITY TWO — SELECTIVE CLASSIFICATION AND UNCERTAINTY

Strengthen the existing abstention and triage framework.

The goal is not merely to maximise accepted accuracy. It is to make the relationship between prediction coverage and reliability measurable.

Implement or improve:

- Coverage-risk curves.
- Risk-coverage analysis.
- Selective accuracy.
- Selective risk.
- Per-class coverage.
- Per-class accepted accuracy.
- Calibration assessment.
- Reliability diagrams.
- Expected Calibration Error.
- Maximum Calibration Error, where appropriate.
- Brier score.
- Negative log-likelihood.
- Threshold sensitivity analysis.

Review the existing temperature scaling and conformal prediction implementations.

Ensure that:

- Calibration data is independent of final test data.
- Conformal prediction follows its documented assumptions.
- Prediction sets are not confused with calibrated probabilities.
- Abstention reasons are explicit.
- High confidence is not automatically interpreted as correctness.
- Threshold changes are traceable.

Build an evaluation report that compares the existing system with every modified configuration.

Do not sacrifice coverage without explaining the corresponding reliability improvement.

---

# 5. PRIORITY THREE — REAL-WORLD ROBUSTNESS

The current dataset limitations must be addressed.

The system has not established reliable performance on real-world field photographs.

Build a proper robustness evaluation framework.

## Required experiments

Evaluate the model under controlled variations:

- Brightness changes.
- Contrast changes.
- Gaussian noise.
- Blur.
- Image resizing.
- Compression artefacts.
- Partial occlusion.
- Rotation.
- Changes in background.
- Different image resolutions.
- Distracting objects.

Where possible, include real field photographs from appropriately licensed and documented datasets.

Do not mix augmented versions of the same source image across training and test partitions.

## Required reporting

For every experiment:

- Baseline performance.
- Modified-condition performance.
- Accuracy degradation.
- Macro F1 degradation.
- Calibration degradation.
- Abstention behaviour.
- Coverage change.

Separate synthetic robustness testing from actual field-domain validation.

**Important:** Synthetic transformations are not substitutes for real field validation.

If appropriate field datasets are unavailable, create the evaluation infrastructure and clearly identify the missing data.

---

# 6. PRIORITY FOUR — DATASET AND EXPERIMENTAL VALIDITY

Audit the entire experimental pipeline.

Check:

- Train/validation/test separation.
- Duplicate and near-duplicate images.
- Data leakage.
- Class distribution.
- Dataset provenance.
- Label consistency.
- Preprocessing consistency.
- Random seeds.
- Reproducibility.
- Evaluation scripts.

Where the dataset permits, introduce a group-aware split so related images cannot leak across partitions.

Generate:

- Per-class precision.
- Per-class recall.
- Per-class F1.
- Macro and weighted F1.
- Confusion matrix.
- Class distribution.
- Misclassification examples.

Create a structured experiment configuration.

Every experiment should record:

- Model version.
- Dataset version.
- Dataset split.
- Random seed.
- Preprocessing configuration.
- Threshold configuration.
- Calibration configuration.
- Evaluation date.
- Relevant metrics.

Avoid silently overwriting previous experiment results.

---

# 7. PRIORITY FIVE — DISCOVERY WORKFLOW

Improve the existing researcher-assisted discovery functionality.

Current functionality includes:

- Anomaly scanning.
- Feature-based clustering.
- Review queue.
- Five-question specimen questionnaire.
- Researcher review status.
- JSON/CSV export.

Strengthen it without overstating its scientific capability.

## Required improvements

### A. Review queue

Ensure each candidate stores:

- Unique specimen ID.
- Original image or reliable local image reference.
- Predicted known class.
- Prediction confidence.
- Uncertainty score.
- Anomaly score.
- Detector name and version.
- Crop information.
- Questionnaire answers.
- Submission timestamp.
- Review status.
- Researcher notes.

Validate input fields.

Handle duplicate submissions safely.

Prevent malformed requests from corrupting stored review data.

### B. Clustering

Audit the existing DBSCAN implementation.

Investigate:

- Feature normalisation.
- Distance metric.
- Epsilon selection.
- Minimum sample selection.
- Noise handling.
- Cluster stability.
- Sensitivity to hyperparameters.

Provide useful cluster summaries.

Include specimen IDs, cluster size, representative images, and available questionnaire information.

Do not automatically convert a cluster into a disease class.

A cluster must remain an investigative hypothesis.

### C. Researcher feedback

Improve the review status workflow:

- Pending.
- Reviewed.
- Confirmed observation.
- Rejected.
- Requires further investigation.

Keep human-confirmed observations separate from automated predictions.

Never interpret a researcher's preliminary review as scientifically established disease confirmation.

---

# 8. PRIORITY SIX — DISCOVERY CACHE AND DEPLOYMENT

Fix the existing cache dependency problem.

The README identifies two files required by the discovery system:

- `backbone_features.onnx`
- `features_full.npz`

A fresh installation currently cannot be assumed to have these files.

## Required work

1. Audit how these files are generated.
2. Identify their exact schema and compatibility requirements.
3. Build a reliable cache-generation command if one does not already exist.
4. Validate cache integrity before use.
5. Verify compatibility with the loaded model.
6. Detect missing or corrupted files.
7. Display actionable error messages.
8. Avoid crashing normal inference when discovery caches are unavailable.
9. Document cache generation separately from inference installation.
10. Ensure Docker and local execution use consistent cache paths.

Where practical, add:

- Cache version metadata.
- Model hash.
- Dataset hash.
- Feature dimensions.
- Creation timestamp.
- Validation checks.

Do not invent cache contents.

A clean installation must provide a clear explanation of which features are available and which require additional assets.

---

# 9. PRIORITY SEVEN — SOFTWARE ENGINEERING

Perform a full engineering audit.

Inspect:

- Python package structure.
- Imports.
- API handlers.
- Configuration validation.
- Error handling.
- Logging.
- File handling.
- Temporary files.
- Cache management.
- Model loading.
- Resource management.
- Dependency versions.
- CLI commands.
- Docker configuration.
- Security-sensitive operations.

Improve code quality without unnecessary architectural rewrites.

## Requirements

- Explicit input validation.
- Predictable error responses.
- Consistent logging.
- No silent exception swallowing.
- No unsafe file-path handling.
- No arbitrary file overwrites.
- No unnecessary global mutable state.
- Clean API response schemas.
- Clear separation between inference, evaluation, and discovery.
- Graceful handling of missing optional components.

Keep local inference independent of external services.

---

# 10. PRIORITY EIGHT — TESTING

Expand the existing test suite.

The reported baseline is 26 passing checks.

Do not remove tests simply to achieve a green test result.

Add appropriate unit and integration tests for:

- Model loading.
- ONNX inference.
- Preprocessing.
- Calibration.
- Abstention.
- Conformal prediction.
- OOD scores.
- Threshold validation.
- Review queue.
- Questionnaire validation.
- Clustering.
- Cache generation and loading.
- Export functions.
- API routes.
- Missing dependencies.
- Corrupted files.
- Invalid inputs.
- Docker configuration where practical.

Use small synthetic fixtures where possible.

Do not require a large dataset for ordinary unit tests.

Separate slow experimental tests from fast routine tests.

Run the entire relevant test suite after modifications.

Report the exact test count and actual results.

---

# 11. PRIORITY NINE — USER INTERFACE

Audit the existing Flask interface.

Improve the actual working application.

Keep the interface research-oriented, clear, and technically professional.

Prioritise:

- Prediction result clarity.
- Explicit uncertainty presentation.
- Clear accepted/review decisions.
- Anomaly-score explanation.
- Review queue visibility.
- Cluster exploration.
- Specimen details.
- Evaluation dashboards.
- Model information.
- System health.
- Missing-cache warnings.

Make sure that every visible control performs its advertised action.

Do not create decorative buttons or mock functionality.

A visualisation must display actual system data or clearly identified sample data.

Do not label a specimen as a newly discovered disease based on an anomaly score.

---

# 12. PRIORITY TEN — RESEARCH PRESENTATION

Update the research framing to match the actual implementation and experimental evidence.

Use the following central project description:

**PhytoTrace: An Uncertainty-Aware Plant Disease Recognition and Researcher-Assisted Anomaly Triage System.**

The project should communicate three distinct capabilities:

1. Recognition of known disease classes.
2. Selective prediction and uncertainty-based triage.
3. Researcher-assisted investigation of unusual specimens.

Do not claim automated discovery or confirmation of new plant diseases.

Update relevant project documentation only after implementation and verification.

Required documents:

- README.
- Technical architecture.
- Model card.
- Evaluation methodology.
- Experimental results.
- Limitations.
- Reproducibility instructions.
- API documentation.
- Deployment instructions.
- Research report.

Use only real experimental results.

Explicitly distinguish:

- Implemented functionality.
- Experimentally demonstrated capability.
- Unverified hypotheses.
- Known limitations.
- Future research directions.

---

# 13. PERFORMANCE CONSTRAINTS

The project must remain practical on a CPU-only machine.

Prioritise:

- Efficient ONNX inference.
- Lazy model loading where appropriate.
- Avoiding unnecessary model duplication.
- Reusable feature caches.
- Memory-efficient batch processing.
- Configurable batch sizes.
- Efficient image processing.
- Avoiding unnecessary recomputation.

Do not introduce huge models or GPU-only dependencies without a compelling, measurable reason.

Do not optimise code based solely on assumptions.

Benchmark before and after significant performance changes.

---

# 14. IMPLEMENTATION ORDER

Execute the work in this order.

## Phase 1 — Repository audit

Inspect the actual repository.

Produce an internal implementation map and identify critical dependencies.

Record the existing test results before modifications.

Do not start rewriting files before understanding their purpose.

## Phase 2 — Research validity

Prioritise:

1. OOD evaluation audit.
2. Selective classification audit.
3. Calibration assessment.
4. Dataset leakage checks.
5. Robustness evaluation.

Fix demonstrable implementation errors before introducing new methods.

## Phase 3 — Discovery reliability

Improve:

1. Cache generation.
2. Anomaly scoring.
3. Clustering.
4. Review queue.
5. Researcher feedback.
6. Data exports.

## Phase 4 — Engineering

Improve:

1. Error handling.
2. Configuration.
3. API reliability.
4. CLI.
5. Packaging.
6. Deployment.
7. Resource management.

## Phase 5 — Testing

Add tests and run existing and new tests.

Fix regressions.

## Phase 6 — Interface

Improve the functioning Flask application.

Connect all components to real backend functionality.

## Phase 7 — Documentation

Update documentation based on what was actually implemented and verified.

---

# 15. DEFINITION OF DONE

The task is not complete merely because files were edited.

The following conditions must be satisfied wherever the repository and available resources permit:

- Existing inference functionality is preserved.
- Existing model variants remain functional.
- OOD evaluation is reproducible.
- Selective classification is properly evaluated.
- Calibration is independently assessed.
- Data leakage checks exist.
- Robustness experiments are reproducible.
- Discovery cache generation is documented and validated.
- Review queue operations work.
- Clustering is reproducible.
- API routes work.
- CLI commands work.
- Docker configuration is validated.
- Tests pass.
- No fabricated metrics exist.
- No misleading new-disease claims exist.
- Documentation matches the implementation.

Blocked requirements must be explicitly listed rather than falsely marked complete.

---

# 16. FINAL REPORT REQUIRED

At completion, produce a factual engineering report containing:

### A. Changes implemented
List the actual source files modified and the functionality improved.

### B. Research improvements
Explain the evaluation and methodology changes.

### C. Before/after comparison
Show actual measured results.

Use `Not evaluated` wherever a comparison cannot be performed.

### D. Testing
Provide exact test results.

### E. Deployment
Provide clean-installation and execution instructions.

### F. Remaining weaknesses
List unresolved scientific, technical, and data limitations.

### G. Research credibility
Explain which claims are supported by measured evidence and which remain hypotheses.

### H. Final assessment
Provide an evidence-based assessment of the improved system.

---

# FINAL EXECUTION INSTRUCTION

Start by inspecting the actual repository.

Then implement the improvements in the specified priority order.

Do not stop after generating an audit document or improvement plan.

Do not replace source-code implementation with Markdown explanations.

Do not manufacture results to make the project appear successful.

When a task is blocked, continue with the remaining independent tasks.

**The objective is not to make PhytoTrace look like an advanced research project. The objective is to make its actual implementation, research methodology, evaluation, and reliability worthy of one.**