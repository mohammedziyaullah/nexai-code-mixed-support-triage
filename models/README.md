# Models Directory

This directory stores serialized model artifacts, vectorizers, and configuration files for intent and urgency classification.

## Model Governance and Artifact Policies

1. **Evaluation Before Committing:**
   - Any generated model artifact (e.g., joblib/pickle files, tokenizers, fine-tuned model checkpoints) must be thoroughly evaluated against defined validation benchmarks (accuracy, macro F1, calibration, and inference latency) before being considered for commit.
   - Evaluation metrics, training configuration, and performance logs must be documented alongside candidate models.

2. **Storage Considerations:**
   - Large binary artifacts exceeding typical Git storage limits must be tracked using Git LFS or external artifact registries rather than committed as raw binaries to Git history.
   - Lightweight pipelines and configuration files may be versioned after validation.
