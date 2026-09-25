---
license: apache-2.0
library_name: numpy
pipeline_tag: image-classification
tags:
  - image-forensics
  - source-attribution
  - openai
  - google
  - open-set-classification
---

# OpenAI and Google image source classifier

This opt-in model classifies decoded image pixels as a likely `openai` or
`google` export pattern, or abstains as `unknown`. It is designed for images
whose metadata was removed without deliberately erasing or regenerating the
pixel pattern. It is **not a SynthID detector**: a provider label does not prove
that SynthID is present, and `unknown` does not prove that it is absent.

## Artifact and runtime

The pickle-free `openai-google-source-v1.npz` contains only derivative
spectral templates and numeric thresholds. It contains no training images,
paths, image hashes, embeddings, catalog, or per-image predictions. Its SHA-256
is `0e94e565c181416e4f63a2bfc06cd5179f4201befda7481ac510f6a168c612f4`.
The corpus used to fit the templates is private and is not redistributed.

Install `remove-ai-watermarks[source-classify]` and call the explicit Python
API `classify_image_source(path)`. The installed library pins this repository
to a verified commit and verifies the artifact hash. It can use a local NPZ
through `RAIW_IMAGE_SOURCE_WEIGHTS`. The old `classify_source` API is a separate,
earlier model and is not silently replaced by this one.

The result exposes `label`, `reason`, numeric branch `scores`, and
`watermark_truth="unknown"`. A simultaneous OpenAI and Google match is returned
as label `unknown` with reason `conflict`, not guessed as one provider.

## Evidence and limits

On 600 public [Qwen Image Bench](https://huggingface.co/datasets/Qwen/Qwen-Image-Bench)
originals from unused prompt IDs, the frozen model classified 198/200
OpenAI-labeled, 196/200 Google-labeled, and abstained on 200/200 FLUX-labeled
files. These are uploader source labels, not per-file authenticated provenance;
this public route was opened during earlier model development, so it is a
regression check, not an independent final test.

An additional post-freeze camera control selected 42 JPEG photographs with
camera EXIF and pre-2022 capture dates from a pinned prefix of
[lightcella/photo-corpus](https://huggingface.co/datasets/lightcella/photo-corpus).
The model returned `unknown` for all 42 original decoded-pixel views and their
JPEG95, JPEG75 and bilinear-1024 transformations. The photos come from one
archive shard; EXIF and the dataset's Flickr-original description are not
cryptographic proof of origin. No originals are redistributed here.

The model can be wrong on other providers, camera types, editing pipelines,
resizes, crops and compression settings. Some non-target generator and
photographic stress panels found false provider matches in related candidates.
It must not be used as evidence of watermark presence, authorship, fraud,
policy compliance, or provenance. Report the abstention and keep independent
metadata/provenance findings separate from this source-pattern prediction.

The artifact is Apache-2.0 licensed. Input photographs and their respective
licenses are not part of this repository.
