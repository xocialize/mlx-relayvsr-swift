#!/usr/bin/env python
"""dump_goldens.py — upstream RelayVSR (kopperx/RelayVSR @ e856773) on CPU in fp32, with taps, as the parity oracle.

Upstream runs VERBATIM through its own non-CUDA paths: the Wan generator's uncompiled FlexAttention under the
spatial block mask and the decoder net's chunked SDPA reference. The only shim is the `triton` import stub (both
attention files import it at module scope; neither kernel is called off CUDA). CPU keeps upstream's float64 RoPE
source and float64 scheduler arithmetic.

The clip is procedural (no pixels from any corpus): 9 frames, 64×192 (W×H) LR → 256×768, window_size 3, so
  • keyframes fall on 0, 2, 4, 6, 8 — five generator calls, the fifth wrapping the 4-slot K/V ring;
  • the generator's 8×24 token grid is taller than its 22-row window, so the clamped window mask bites;
  • the decoder's 16×48 token grid is taller than both its windows (24 video, 48 reference), so both masks bite;
  • each 3-frame interval exercises the decoder's rolling video memory (window 2) and its reset at an endpoint.

Recorded (fp32 unless noted; torch layouts, NCHW):
  lr                      [9, 192, 64, 3] uint8
  noise.<j>               [1, 48, 1, 48, 16]   the generator's sampling noise for keyframe j (INJECT this)
  latent.<j>              [1, 48, 48, 16]      the reference latent it returns
  gen.temb / gen.tproj / gen.ctx / gen.cross_k.0 / gen.cross_v.0     the cached constants
  gen.kf<j>.lq            lq_proj tokens          (j = 0, 4)
  gen.kf<j>.b<i>          block i output tokens   (i = 0, 1, 29)
  gen.kf<j>.pred          the model prediction [1, 48, 1, 48, 16]
  dec.cond<j>.tokens / .b<i>.key / .b<i>.value / .hidden            endpoint priming (j = 0; i = 0, 9)
  dec.w<w>f<n>.tokens / .b<i> / .bicubic                             decode taps (window w = 0, 1; n = 0, 1, 2)
  out.<t>                 [3, 768, 256]        every emitted frame, in [-1, 1] before clamping

usage: RELAYVSR_SRC=<checkout> RELAYVSR_WEIGHTS=<dir> dump_goldens.py <out.safetensors>
"""
import os, sys, types

UPSTREAM = os.environ["RELAYVSR_SRC"]
WEIGHTS = os.environ["RELAYVSR_WEIGHTS"]
sys.path.insert(0, UPSTREAM)

import numpy as np
import torch

# torch/diffusers first with no `triton` visible, then the stub only while upstream imports (see run_relayvsr.py)
import diffusers.models.modeling_utils  # noqa: F401
import torch.nn.attention.flex_attention  # noqa: F401
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: F401
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin  # noqa: F401
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward  # noqa: F401
from diffusers.models.cache_utils import CacheMixin  # noqa: F401
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps  # noqa: F401
from diffusers.models.normalization import FP32LayerNorm  # noqa: F401


class _Any:
    def __call__(self, *a, **k):
        return a[0] if len(a) == 1 and callable(a[0]) and not k else _Any()

    def __getattr__(self, name):
        return _Any()


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


_t = _Stub("triton")
_t.language = _Stub("triton.language")
sys.modules["triton"], sys.modules["triton.language"] = _t, _t.language
import refvsr_inference.models.wan.sampling as _samp
from refvsr_inference.checkpoints import load_checkpoints
from refvsr_inference.models.wan.sampling import WanVSRPipeline
from refvsr_inference.runtime.streaming import CollaborativeFrameStream, InputFrame
del sys.modules["triton"], sys.modules["triton.language"]
from safetensors.torch import save_file

T, H, W, WINDOW = 9, 192, 64, 3
G = {}


def put(name, t):
    G[name] = t.detach().to("cpu").float().contiguous().clone() if t.dtype != torch.uint8 else t.clone()


def tap(when, name):
    """A forward hook that records the output and returns None (a non-None return would REPLACE the output)."""
    def hook(module, args, output):
        if when(output):
            put(name(), output)
    return hook


def clip():
    """Smooth colour field + a moving textured box + fine noise: detail at every scale, motion every frame."""
    rs = np.random.RandomState(7)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    tex = rs.rand(H, W, 3).astype(np.float32)
    frames = []
    for t in range(T):
        base = np.stack([0.5 + 0.4 * np.sin(xx / 9.0 + t * 0.3), 0.5 + 0.4 * np.cos(yy / 13.0 - t * 0.2),
                         0.5 + 0.3 * np.sin((xx + yy) / 17.0)], -1)
        img = 0.8 * base + 0.2 * tex
        y0, x0 = 20 + 9 * t, 6 + 3 * t
        img[y0:y0 + 40, x0:x0 + 24] = 0.3 + 0.7 * np.roll(tex, 5 * t, axis=0)[y0:y0 + 40, x0:x0 + 24]
        frames.append((img.clip(0, 1) * 255).round().astype(np.uint8))
    return np.stack(frames)


@torch.no_grad()
def main(out):
    torch.manual_seed(0)
    dev, dtype = torch.device("cpu"), torch.float32
    wan, flash = load_checkpoints(f"{WEIGHTS}/wan.safetensors", f"{WEIGHTS}/flash.safetensors", device=dev, dtype=dtype)
    sampler = WanVSRPipeline(
        scheduler_config={"num_train_timesteps": 1000, "shift": 5.0},
        sampling={"execution_mode": "streaming", "num_inference_steps": 1, "timesteps": [1000],
                  "guidance_scale": 1.0, "seed": 42, "kv_cache_window_size": wan.temporal_context_frames},
    )
    sampler.prepare_streaming(model=wan, prompt_embeds=wan.prompt_embeds, device=dev, compute_dtype=dtype)
    cc = sampler._streaming_conditioning_cache
    put("gen.temb", cc.temb); put("gen.tproj", cc.timestep_proj); put("gen.ctx", cc.encoder_hidden_states)
    put("gen.cross_k.0", cc.cross_attention_caches[0].key_value[0])
    put("gen.cross_v.0", cc.cross_attention_caches[0].key_value[1])

    # --- generator taps
    kf = {"j": -1}
    orig_randn = _samp.randn_tensor

    def randn(shape, **k):
        n = orig_randn(shape, **k)
        kf["j"] += 1
        put(f"noise.{kf['j']}", n)
        return n
    _samp.randn_tensor = randn
    TAP_KF, TAP_B = (0, 4), (0, 1, 29)
    on_kf = lambda o: kf["j"] in TAP_KF
    wan.lq_proj.register_forward_hook(tap(on_kf, lambda: f"gen.kf{kf['j']}.lq"))
    for i in TAP_B:
        wan.blocks[i].register_forward_hook(tap(lambda o: kf["j"] in TAP_KF and o.shape[1] > 0,
                                                lambda i=i: f"gen.kf{kf['j']}.b{i}"))
    wan.register_forward_hook(tap(on_kf, lambda: f"gen.kf{kf['j']}.pred"))

    # --- decoder taps
    cond = {"j": -1}
    orig_prep = flash.prepare_condition

    def prep(latent):
        cond["j"] += 1
        j = cond["j"]
        put(f"latent.{j}", latent)
        if j == 0:
            hooks = []
            for i in (0, 9):
                blk = flash.backbone.blocks[i]
                op = blk.prime_condition

                def wrapped(h, *, rope, grid_size, _op=op, _i=i):
                    o, c = _op(h, rope=rope, grid_size=grid_size)
                    put(f"dec.cond0.b{_i}.key", c.key); put(f"dec.cond0.b{_i}.value", c.value)
                    if _i == 9:
                        put("dec.cond0.hidden", o)
                    return o, c
                blk.prime_condition = wrapped
                hooks.append((blk, op))
            def cond_tokens(m, a, o):
                put("dec.cond0.tokens", o + flash.condition_type_embedding)
            h = flash.condition_encoder.register_forward_hook(cond_tokens)
            r = orig_prep(latent)
            h.remove()
            for blk, op in hooks:
                blk.prime_condition = op
            return r
        return orig_prep(latent)
    flash.prepare_condition = prep

    win = {"w": -1, "n": 0}
    orig_begin = flash.begin_stream

    def begin(*a, **k):
        win["w"] += 1
        win["n"] = 0
        return orig_begin(*a, **k)
    flash.begin_stream = begin
    tag = lambda: f"dec.w{win['w']}f{win['n']}"
    live = lambda: win["w"] in (0, 1)
    def lr_tokens(m, a, o):
        if live():
            put(f"{tag()}.tokens", o + flash.video_type_embedding)
    flash.lr_encoder.register_forward_hook(lr_tokens)
    for i in (0, 9):
        blk = flash.backbone.blocks[i]
        op = blk.forward_video

        def fv(h, _op=op, _i=i, **k):
            o, c = _op(h, **k)
            if live():
                put(f"{tag()}.b{_i}", o)
            return o, c
        blk.forward_video = fv
    orig_rec = flash._reconstruct

    def rec(hidden, *, lr_frame, token_height, token_width):
        if live():
            put(f"{tag()}.bicubic", torch.nn.functional.interpolate(lr_frame, scale_factor=4, mode="bicubic",
                                                                    align_corners=False))
        o = orig_rec(hidden, lr_frame=lr_frame, token_height=token_height, token_width=token_width)
        win["n"] += 1
        return o
    flash._reconstruct = rec

    frames = clip()
    G["lr"] = torch.from_numpy(frames)
    stream = CollaborativeFrameStream(wan_model=wan, flash_model=flash, wan_pipeline=sampler,
                                      prompt_embeds=wan.prompt_embeds, device=dev, compute_dtype=dtype,
                                      window_size=WINDOW, seed=42)
    packets = (InputFrame(i, torch.from_numpy(f).permute(2, 0, 1).contiguous(), source_time=i / 24.0)
               for i, f in enumerate(frames))
    n = 0
    for r in stream.frames(packets):
        put(f"out.{r.index}", r.rgb[0])
        n += 1
        print(f"frame {r.index} window [{r.window_start}, {r.window_end}]", flush=True)
    assert n == T and list(stream.last_keyframe_indices) == [0, 2, 4, 6, 8], list(stream.last_keyframe_indices)
    meta = {"frames": str(T), "lr": f"{W}x{H}", "window_size": str(WINDOW), "seed": "42", "dtype": "fp32",
            "device": "cpu", "torch": torch.__version__, "upstream": "kopperx/RelayVSR@e856773",
            "keyframes": "0,2,4,6,8"}
    save_file(G, out, metadata=meta)
    print(f"{len(G)} tensors → {out}")
    for k in sorted(G):
        print(f"  {k:28s} {tuple(G[k].shape)}")


if __name__ == "__main__":
    main(sys.argv[1])
