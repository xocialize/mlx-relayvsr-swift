# RelayVSR → MLX Swift: porting spec and receipts (2026-11-14)

Upstream: [kopperx/RelayVSR](https://github.com/kopperx/RelayVSR) @ `e856773`, weights
[kopper/RelayVSR](https://huggingface.co/kopper/RelayVSR) @ `751954dc` (Apache-2.0 both; arXiv 2609.37850).
Why it was ported, and how it compares with SeedVR2 and FlashVSR: `mlxengine-forge/Docs/RELAYVSR-SPIKE.md`.

## Phase table

| Phase | Gate | Result |
|---|---|---|
| S0 key contract | both networks load `verify: [.all]`; tensor and parameter counts equal the released files | **PASS** — generator 833 / 5,031,059,136 (wan.safetensors minus `prompt_embeds`), decoder 200 / 31,934,208 |
| S1 components | sub-op taps against upstream's CPU-fp32 goldens, each component fed golden inputs | **PASS** — every isolated tap rel ≤ 3e-5 (CPU and GPU, fp32) |
| E2E | the whole stream from (LR frames, injected noise) | **PASS** — every frame ≥ 120.5 dB (CPU), ≥ 122.7 dB (GPU), fp32 |
| S6 production dtypes | generator bf16 + decoder fp16 against the fp32 lane | 49–53 dB on every frame of the ×4 cells — upstream's own bf16 is 49.7–52.5 dB |
| S7 engine wrap | `MLXRelayVSR` through the real `MLXServeEngine`; MAT / licence / FIT / CAN suites | **PASS** — 15 package tests; ×4 and ×2 clips decode, every frame kept |

Reports: `oracle/reports/s1_cpu_fp32.log`, `s1_gpu_fp32.log`, `s1_gpu_production_drift.log`, `dump_goldens.log`.

## What maps to what

| Port | Upstream | Notes |
|---|---|---|
| `RelayVSRGenerator` (`Generator.swift`) | `models/wan/transformer.py` `WanTransformer3DModel`, streaming path | diffusers key names; one keyframe per call |
| `LQ4xProj` | `models/wan/lq_proj.py` | pixel-unshuffle 16 → 1×1 conv → stride-2 3×3 conv → Linear; added to the patch tokens |
| `RelayVSRGeneratorState` | `WanSelfAttentionKVCache` × 30 + `WanStreamingState` | a 4-slot ring of unrotated K/V; slot = keyframe mod 4 |
| `RelayVSRGenerator.prepare(prompt:)` | `WanVSRPipeline.prepare_streaming` | t = 1000 constants, prompt projection, 30 cross-attention K/V, computed once |
| `RelayVSRDecoder` (`Decoder.swift`) | `models/flash/model.py` `FlashVSRConditionNet` (variant S) | conv encoder → 10 GQA blocks → PixelShuffle ×16 residual + bicubic |
| `prepareCondition` / `beginStream` / `decodeStep` | `prepare_condition` / `begin_stream` / `decode_step` | persistent reference K/V + rolling video K/V (window 2) |
| `RelayVSRStream`, `RelayVSRPipeline.run` | `runtime/streaming.py` `CollaborativeFrameStream` | keyframes every 15 frames, both endpoints, overlap frame decoded twice |
| `RelayVSRResample.upsample4` | `F.interpolate(…, scale_factor=4, bilinear / bicubic, align_corners=False)` | fixed 4-phase taps; matches torch to 2e-7 |

## Substitutions (everything else is isomorphic)

1. **Layout.** NCHW / NCDHW → channels-last. Tokens stay row-major.
2. **The generator's local spatial attention.** Upstream: a Triton kernel on CUDA, FlexAttention under a block mask
   elsewhere. Here: dense SDPA under the same `mask_mod` — a 22×40-token window centred on the query and clamped
   inside the grid, the same for every cached keyframe.
3. **The decoder's local attention.** Upstream: a fused Triton kernel on CUDA, SDPA under the window mask 64 queries
   at a time elsewhere. Here: one SDPA call per layer under the same mask — per key plane a query-centred,
   half-open, boundary-truncated window (24×24 for video frames, 48×48 for the two references).
4. **Grouped-query attention.** K/V of each of the 2 groups are repeated over its 4 heads.
5. **RoPE.** Generator tables in Double, applied in fp32 (upstream: float64 on CPU and CUDA). Decoder tables in fp32,
   as upstream.
6. **RMSNorm reduces in fp32** (see below). Identical in the fp32 lane.
7. **Delivery.** `RelayVSRStream.push` returns a whole interval once its far endpoint has arrived; upstream yields
   the same frames one at a time as they are decoded. Same frames, same order.

## Parity

The oracle (`oracle/dump_goldens.py`) runs upstream VERBATIM on the CPU in fp32 through its own non-CUDA paths; the
only shim is a `triton` import stub. The clip is procedural, 9 frames of 64×192 LR with window 3, chosen so that:
five keyframes run and the fifth wraps the 4-slot K/V ring; the generator's 8×24 token grid is taller than its window;
the decoder's 16×48 grid is taller than both of its windows; and every interval uses the rolling video memory.

| Gate (fp32) | CPU | GPU |
|---|---|---|
| bicubic ×4 | rel 2e-7 | 2e-7 |
| generator constants (temb, timestep_proj, prompt projection, cross K/V) | ≤ 9e-7 | ≤ 9e-7 |
| LQ projector | 1.2e-6 | 1.2e-6 |
| generator blocks 0 / 1 / 29, keyframes 0 and 4 | ≤ 3.2e-6 | ≤ 2.7e-6 |
| reference latents 0–4 | ≤ 3.2e-5 | ≤ 1.9e-5 |
| decoder reference priming (tokens, K/V of blocks 0 and 9, hidden) | ≤ 1.8e-6 | ≤ 1.9e-6 |
| decoder frames, intervals 0 and 1 (tokens, blocks 0 and 9, output) | ≤ 2.6e-5 | ≤ 1.5e-5 |
| **E2E, 9 frames** | **≥ 120.5 dB** (rel ≤ 1.1e-4) | **≥ 122.7 dB** (rel ≤ 5.8e-5) |

Not yet gated against a CUDA run of the Triton kernels (no CUDA here). Upstream states its kernels compute the same
masks; the spike measured its two non-CUDA paths against the dense stand-ins at ≤ 1/255 level in fp32.

## Precision

Released dtypes: generator bf16, decoder fp32. Upstream infers with both in bf16.

- **`parity`** — fp32 everywhere; the generator is an exact upcast of its released numbers.
- **`production`** — **generator bf16, decoder fp16.**

Why not bf16 for the decoder, as upstream does: on MLX a bf16 decoder first measured 15–18 dB further from fp32 than
torch's bf16 (block 0: 39.8 dB against 50.3 dB in torch, CPU and MPS alike). The cause was the RMSNorm reduction: a
mean of squares over 512–3072 values in 8 mantissa bits loses more in MLX than in torch. With the reduction in fp32
(substitution 6) the bf16 lanes match torch:

| against the fp32 goldens, same noise | torch bf16 (MPS) | MLX bf16 | MLX production |
|---|---|---|---|
| generator, keyframe 0 prediction | 51.1 dB | 50.9 dB | 50.9 dB |
| reference latents 0–3 | 46.1–46.6 dB | 45.4–46.5 dB | 45.4–46.5 dB |
| decoder block 0 / block 9 (golden latents) | 50.3 / 55.2 dB | 49.7 / 55.4 dB | **66.7 / 72.9 dB** |

fp16 for the decoder costs nothing (72–76 ms/frame either way) and is the more accurate, so it is the default.

**One outlier, explained.** On the procedural clip the fifth keyframe drifts more in MLX bf16 than in torch bf16
(latent 4: 34.8 dB against 45.2 dB; block 29: 55.2 against 66.6 dB) while keyframes 0–3 are level. It is exact in
fp32, so the logic is right. It does not show on real footage: with window 4 (17 keyframes, the ring wrapping
thirteen times) production stays 49–53 dB from fp32 on every frame of both the live-action and the anime clip. The
procedural clip is noise-heavy and out of distribution for a GAN-distilled one-step generator.

## Quality (×4, 320×192 → 1280×768, 49 frames; the spike's cells and metric code)

SSIMULACRA2 mean · temporal error ×Lanczos · Sobel gradient. `bench/x4_quality.csv` has every row.

| arm | Wp clean | Wp codec | Wa clean | Wa codec | Hp clean | Hp codec |
|---|---|---|---|---|---|---|
| upstream, torch-MPS bf16 | 23.4 · 1.44 · 32 | −14.2 · 1.44 · 36 | 7.0 · 3.21 · 15 | −10.9 · 1.55 · 13 | 33.8 · 1.18 · 23 | −25.2 · 1.00 · 22 |
| **this port, production** | 23.3 · 1.50 · 32 | −17.5 · 1.54 · 38 | 7.7 · 3.10 · 15 | −10.5 · 1.51 · 13 | 35.5 · 1.16 · 23 | −24.5 · 0.99 · 22 |

The two arms draw different noise (torch's RNG and MLX's), so they agree as two seeds of one model do.

## Speed (M5 Max, 49 frames, s per output frame, the pipeline call alone)

Taken 2026-11-14 with the GPU shared (AGX utilisation 67–98 % before each run, thermal `.nominal`); `bench/run_*.json`.

| | 1280×768 | 1920×1080 |
|---|---|---|
| upstream torch-MPS bf16 (dense stand-ins) | 0.25–0.28 | 0.46–0.72 |
| **this port, production** | **0.075** | **0.245** |
| this port, fp32 | 0.103 | — |

Where the time goes (production): at 1280×768 the generator takes 0.27 s per keyframe (1.36 s for 5) and the decoder
46 ms per frame (2.25 s); at 1920×1080, 0.64 s per keyframe and 175 ms per frame. The decoder is 60–70 % of the cost, and at 1080p its dense attention spends most of it on masked-out pairs (the mask is 13.5 % full).

Through the engine, decode and HEVC encode included: 132 ms/frame at 1280×768, 325 ms/frame at 1920×1080.

## Memory (through `MLXServeEngine`, one clip per process, lifetime-peak `phys_footprint`)

| input → output | lane | phys peak | MLX peak |
|---|---|---|---|
| 320×192 → 1280×768 | bf16 | 14.33 GB | 12.75 GB |
| 480×270 → 1920×1080 | bf16 | 16.80 GB | 15.25 GB |
| 320×192 → 1280×768 | fp32 | 26.09 GB | 24.43 GB |

Weights are 10.2 GB of that in the bf16 lane. Memory does not grow with clip length. The declared footprints
(`RelayVSRUpscalePackage.footprints`) are these points with 1.2× headroom; the ceiling is 1920×1088.

## Open items

Bridge: receipt [[AB-R-0866]] (this port), [[AB-R-0863]] (the spike), lesson [[AB-L-0340]] (bf16 reductions), tasks
[[AB-T-0486]] (item 3) and [[AB-T-0487]] (items 1 and 7).

1. **Publish the lane.** `RelayVSRConfiguration.repo` names `mlx-community/RelayVSR-bf16`, which does not exist yet.
   Until it does, the engine's fresh-machine download path is untested and hosts must pass `weightsDirectory` or hold
   the lane in their model store.
2. **CUDA oracle.** Gate against a real CUDA run with the Triton kernels (a Runpod GPU for an hour or two).
3. **A windowed attention kernel for the decoder.** Dense masked attention is exact and fine to ~1 Mpx; at 1080p it
   is 70 % of the run and at 4K output the mask alone would be ~4 GB and the score matrices several times that. The window is at most 2 × 2,304 +
   2 × 576 keys per query against 4 × 8,160 at 1080p — about 5.7× less work. `mlx-flashvsr-swift`'s Metal
   block-sparse kernel is the starting point.
4. **Scene cuts.** Keyframes fall every 15 frames regardless of content; an interval across a cut gets endpoints from
   two shots. Untested (all clips so far are single shots).
5. **An int8 generator.** 10 GB of the 14 GB is generator weights.
6. **Clean timing.** Every number here was taken on a shared GPU.
7. **Registration.** Not yet registered in ForgeCore or the model registry.

## Tooling

```
swift build -c release --build-system swiftbuild --product relayvsr-smoke        # .build/out/Products/Release/
swift test --build-system swiftbuild                                               # 9 structure + 15 package tests

relayvsr-smoke s0 <laneDir>
relayvsr-smoke s1 <laneDir> <golden.safetensors> [--e2e] [--gpu [--tf32]] [--rel 1e-4] [--chain-rel 1e-3]
                  [--dtype bf16] [--dec-dtype bf16|fp16]            # a half-precision lane = a drift study
relayvsr-smoke run <lrFramesDir> <outDir> <laneDir> [--dtype bf16|fp32] [--dec-dtype fp16|bf16|fp32]
                   [--window 16] [--limit N] [--seed S] [--profile] [--repeat N] [--cache-mb N]
relayvsr-smoke engine <in.mov|mp4> <out.mp4> <laneDir | --store root> [--scale 2|4] [--fp32] [--window W] [--warmup clip]
```

Oracle (`oracle/`, run with a venv holding torch, diffusers 0.37.1, accelerate, einops, safetensors):
`convert_weights.py <kopper/RelayVSR snapshot> <laneDir>` → `RELAYVSR_SRC=<checkout> RELAYVSR_WEIGHTS=<snapshot>
dump_goldens.py <out.safetensors>`. `GOLDEN_DEVICE` / `GOLDEN_DTYPE` / `GOLDEN_NOISE` exist for drift studies only.
Gate modes set `MLX_ENABLE_TF32=0` before the first MLX op unless `--tf32`.
