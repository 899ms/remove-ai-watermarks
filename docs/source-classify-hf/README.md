---
license: apache-2.0
library_name: numpy
pipeline_tag: image-classification
tags:
  - image-forensics
  - source-attribution
  - open-set-classification
---

# RAIW source-pipeline classifier

This is a compact, abstaining pixel-forensics classifier for three labels:
`openai`, `google`, and `unknown`. It is intended for original-export-looking
images whose metadata may have been removed without changing the pixels.

This model is **not a SynthID detector**. It does not decode or verify a
watermark payload. Its labels describe complete generation and export
pipelines, so renderer, geometry, encoding, post-processing, and watermark
state may be confounded. `unknown` does not mean that SynthID is absent, and a
provider label does not prove that SynthID is present.

## Model

- Input: a 124-dimensional residual/spectral forensic feature vector plus a
  768-dimensional phase-free spectral vector.
- Head: 892 -> 128 -> 64 -> 3 MLP with GELU activations.
- Output: `openai`, `google`, or abstain as `unknown` using independent
  provider margins.
- Artifact: `source-pipeline-mlp.npz`
- SHA-256: `0df4c644d02616265d083dbd3204e4dc617f7133b94af0eb0dc59d5362189290`

The artifact contains only learned parameters, normalization constants,
labels, and thresholds. It does not contain training images, paths, image
hashes, embeddings, or a training catalog.

## Evaluation

The operating point was selected on an internal calibration split. The locked
internal test and the public `bigdatamark/synthid-research` dataset were not
used for architecture, checkpoint, or threshold selection.

| Evaluation | Class | Recall | Precision |
|---|---|---:|---:|
| Locked internal test (5,369 images) | OpenAI | 60.4% | 84.9% |
| Locked internal test (5,369 images) | Google | 57.6% | 87.9% |
| Locked internal test (5,369 images) | Unknown | 98.6% | 93.7% |
| Public external originals (900 images) | OpenAI | 68.7% | 100.0% |
| Public external originals (900 images) | Google | 81.3% | 98.8% |
| Public external originals (900 images) | Unknown/Flux | 99.0% | 66.4% |

On the public external set, a 95% resize reduced Google recall to 1.0%, and a
JPEG quality-75 round trip caused the model to abstain on every image. These
are expected abstentions, not evidence that a watermark was removed.

## Intended use

Use this model as one probabilistic signal in a broader forensic report or to
suggest an appropriate processing path. Do not use it to claim watermark
presence, authorship, policy compliance, fraud, or provenance. Applications
should expose the abstention and the model's narrow source-pipeline claim.

Feature extraction and inference are versioned in the
`remove-ai-watermarks` Python project rather than in this model repository.
