# mlx-relayvsr-swift

**RelayVSR** — streaming ×4 video super-resolution — on Apple silicon with [MLX Swift](https://github.com/ml-explore/mlx-swift).
A port of [kopperx/RelayVSR](https://github.com/kopperx/RelayVSR) ([paper](https://arxiv.org/abs/2609.37850),
Apache-2.0 code and weights).

RelayVSR splits the work between two networks. A large generative model (the Wan2.2-TI2V-5B DiT, distilled to one
step) runs on **one frame in fifteen** and proposes a reference latent. A 32 M-parameter transformer then
super-resolves **every** frame from the low-resolution frame and the two reference latents around it. There is no VAE.

## What to expect

- **Faithful.** It stays close to the source: defocus stays soft and a drawing stays a drawing, so it suits anime and
  graphics as well as live action. On ×4 full-reference clips it scores far above the other generative upscalers
  (SSIMULACRA2 35.5 on live action against −0.9 for SeedVR2-3B and −16.2 for FlashVSR; 23.3 on an anime pan against
  −50.1 and −39.4).
- **Soft.** It removes film grain and adds little high-frequency detail. For the sharpest faces, FlashVSR is the
  better pick. It does not repair codec damage.
- **Fast and light** (M5 Max, generator bf16 + decoder fp16, s per output frame, pipeline only): **0.075 at
  1280×768 and 0.245 at 1920×1080**; process peak 14.6 GB and 18.6 GB. Measured while other apps were using the GPU.
- **×4 only** natively (the engine package also offers ×2 as a box average of the ×4 result). Output lags input by up
  to 15 frames; every frame is kept.

## Layout

| Product | What |
|---|---|
| `RelayVSRMLX` | The engine-agnostic core: both networks (upstream's parameter paths), `RelayVSRPipeline` / `RelayVSRStream`, the ×4 resamplers |
| `MLXRelayVSR` | The [MLXEngine](https://github.com/xocialize/mlx-engine-swift) `videoUpscale` package: decode → stream → HEVC |
| `relayvsr-smoke` | Parity gates (S0 keys, S1 taps, E2E), the frame-directory runner, the real-engine drive |

## Weights

```
hf download kopper/RelayVSR flash.safetensors wan.safetensors
python oracle/convert_weights.py <that snapshot> <laneDir>          # 10.2 GB: generator bf16, decoder fp32
```

Keys are upstream's verbatim; only conv weights are moved to channels-last.

## Use

```
swift build -c release --build-system swiftbuild --product relayvsr-smoke
B=.build/out/Products/Release/relayvsr-smoke
$B run <lrFramesDir> <outDir> <laneDir>                              # PNG frames in, ×4 PNG frames out
$B engine <in.mov> <out.mp4> <laneDir> [--scale 2|4]                 # through MLXServeEngine
```

```swift
let pipe = try RelayVSRPipeline.load(directory: laneDir)             // generator bf16, decoder fp16
try pipe.run(frames: lrFrames) { index, frame in … }                 // (4H, 4W, 3) in [−1, 1], in order
```

Parity, precision choices, memory and the open items are in [PORTING-SPEC.md](PORTING-SPEC.md).
