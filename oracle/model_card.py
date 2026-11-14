#!/usr/bin/env python
"""model_card.py — write the Hub model card (README.md) into a converted lane directory.

usage: python model_card.py <laneDir>
"""
import os, sys

CARD = """---
library_name: mlx
license: apache-2.0
license_link: https://huggingface.co/kopper/RelayVSR/blob/main/LICENSE
pipeline_tag: video-to-video
base_model: kopper/RelayVSR
tags:
  - mlx
  - apple-silicon
  - video-super-resolution
  - super-resolution
  - video-to-video
  - relayvsr
---

# mlx-community/RelayVSR-bf16

**RelayVSR** ([kopper/RelayVSR](https://huggingface.co/kopper/RelayVSR),
[kopperx/RelayVSR](https://github.com/kopperx/RelayVSR), [arXiv 2609.37850](https://arxiv.org/abs/2609.37850)) —
streaming ×4 video super-resolution by large–small model collaboration — converted to MLX for the Swift/MLX port
[`xocialize/mlx-relayvsr-swift`](https://github.com/xocialize/mlx-relayvsr-swift).

A large generative model (the Wan2.2-TI2V-5B DiT, distilled to one step) runs on **one frame in fifteen** and
proposes a reference latent. A 32 M-parameter transformer then super-resolves **every** frame from the
low-resolution frame and the two reference latents around it. There is no VAE and no text encoder at runtime.

## What to expect

- **Faithful.** It stays close to the source: defocus stays soft and a drawing stays a drawing, so it suits anime and
  graphics as well as live action.
- **Soft.** It removes film grain and adds little high-frequency detail, and it does not repair codec damage.
- **×4 only**; output lags input by up to 15 frames; every frame is kept.

## Files

| file | component | params | notes |
|---|---|---|---|
| `generator_bf16.safetensors` | the reference generator: Wan2.2-TI2V-5B DiT (LoRA merged, one step at t = 1000) + the LQ projector | 5,031,059,136 | released dtype (bf16), bit for bit; `patch_embedding` Conv3d (O,D,H,W,I), `lq_proj` Conv2d (O,H,W,I) |
| `decoder_fp32.safetensors` | the lightweight VSR network (upstream `flash.safetensors`, variant S, step 60000 EMA) | 31,934,208 | released dtype (fp32), bit for bit; Conv2d (O,H,W,I) |
| `prompt_bf16.safetensors` | the fixed prompt embedding (1, 512, 4096) | — | split out of upstream's `wan.safetensors` |
| `config.json` | both upstream configs, sampling constants, provenance (source sha256s) | | |

Keys are upstream's verbatim; only conv layouts change (channels-last), and the two `lq_proj` norm weights are
flattened from (C, 1, 1) to (C). Produced by the port's `oracle/convert_weights.py` from `kopper/RelayVSR @ 751954dc`.

One lane serves both precisions: the port runs the generator in bf16 and the small network in fp16 by default, and
upcasts the same files exactly for its fp32 parity lane.

## Parity with upstream

Gated against upstream's own code executed verbatim on the CPU in fp32 (a 9-frame clip chosen so that both window
masks bite and the generator's 4-keyframe cache wraps):

- key contract: 833 + 200 tensors, no missing and no unused key;
- every component within 3e-5 (max abs / reference max) of upstream;
- the whole stream at least 120 dB per frame, on the CPU and on the GPU.

The default half-precision setting stays 49–53 dB from the fp32 lane on real footage, level with upstream's own bf16.
Not yet gated against a CUDA run of upstream's Triton attention kernels; the port uses dense attention under the same
masks.

## Use

```
swift build -c release --build-system swiftbuild --product relayvsr-smoke
.build/out/Products/Release/relayvsr-smoke run <lrFramesDir> <outDir> <this directory>
```

```swift
let pipe = try RelayVSRPipeline.load(directory: laneDirectory)
try pipe.run(frames: lrFrames) { index, frame in … }        // (4H, 4W, 3) in [−1, 1], in order
```

Through [MLXEngine](https://github.com/xocialize/mlx-engine-swift) the `MLXRelayVSR` package downloads this repository
on first use.

Measured on an M5 Max (GPU shared with other apps): 0.075 s per 1280×768 output frame and 0.245 s at 1920×1080;
process peak 14.3 GB and 16.8 GB through the engine.

## Licence

Apache-2.0, as released by the RelayVSR authors; the base model Wan2.2-TI2V-5B is Apache-2.0. The training data is
not disclosed by the authors.
"""

out = sys.argv[1]
with open(os.path.join(out, "README.md"), "w") as f:
    f.write(CARD)
print("wrote", os.path.join(out, "README.md"))
