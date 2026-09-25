# Source-pipeline classification

`classify_source` is the earlier, opt-in, metadata-free pixel classifier for images that
still resemble an original provider export. It returns `openai`, `google`, or
an explicit `unknown` abstention.

This is not a SynthID detector. It does not decode or verify a watermark
payload, and it must not be used to claim that SynthID is present or absent.
The model describes the complete source/export pipeline, where renderer,
geometry, encoding, post-processing, and watermark state can be confounded.

Install the lightweight runtime:

```bash
uv add "remove-ai-watermarks[source-classify]"
```

Call the Python API explicitly:

```python
import remove_ai_watermarks as raiw

result = raiw.classify_source("metadata-stripped.png")
print(result.label)       # "openai" | "google" | "unknown"
print(result.reason)      # "classified" | "abstained" | "feature_unavailable"
print(result.margins)     # provider margins before thresholding
print(result.thresholds)  # frozen provider thresholds
```

`feature_unavailable` is returned for images below the required 256x256
geometry or for a degenerate feature vector. `abstained` means neither provider
cleared its independent calibrated margin. Both are honest `unknown` results.

The first call downloads the exact pinned revision of
[`wiltodelta/raiw-source-classify`](https://huggingface.co/wiltodelta/raiw-source-classify).
Set `RAIW_SOURCE_CLASSIFY_WEIGHTS` to the NPZ itself or a directory containing
`source-pipeline-mlp.npz` for an offline deployment.

The model is useful as a routing suggestion for a processing workflow or as one
probabilistic signal in a broader forensic report. Do not use it as evidence of
authorship, fraud, policy compliance, provenance, or watermark presence. The
published model card records the evaluation and robustness limits.

## New image-source classifier

`classify_image_source` is a separate, frozen model that incorporates selected
spectral and spatial evidence from the earlier research. It does not silently
change `classify_source` or its thresholds. Call it explicitly:

```python
result = raiw.classify_image_source("metadata-stripped.png")
print(result.label)             # "openai" | "google" | "unknown"
print(result.reason)            # "classified" | "abstained" | "conflict" | "feature_unavailable"
print(result.watermark_truth)   # always "unknown"
```

Its first call downloads a hash-verified artifact from the immutable revision
of [`wiltodelta/openai-google-image-source-classifier`](https://huggingface.co/wiltodelta/openai-google-image-source-classifier).
For offline use, set `RAIW_IMAGE_SOURCE_WEIGHTS` to the NPZ file or its
directory. It classifies likely export-source patterns in decoded pixels, not
SynthID presence. The [model card](image-source-hf/README.md) records its
aggregate tests and limitations.
