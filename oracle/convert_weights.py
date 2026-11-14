#!/usr/bin/env python
"""convert_weights.py — kopper/RelayVSR (Apache-2.0) checkpoints → the MLX lane this package loads.

  <out>/generator_bf16.safetensors   wan.safetensors minus `prompt_embeds`; released dtype (bf16) kept bit for bit
  <out>/decoder_fp32.safetensors     flash.safetensors; released dtype (fp32) kept bit for bit
  <out>/prompt_bf16.safetensors      {"prompt_embeds": [1, 512, 4096]} — the fixed text conditioning
  <out>/config.json                  both upstream configs (the safetensors metadata), provenance, hashes

Keys are upstream's VERBATIM. Layout changes only (values untouched):
  conv weights → channels-last:  Conv3d [O, I, D, H, W] → [O, D, H, W, I];  Conv2d [O, I, H, W] → [O, H, W, I]
  lq_proj.norm{1,2}.weight [C, 1, 1] → [C]

usage: convert_weights.py <dir with wan.safetensors + flash.safetensors> <out>
"""
import hashlib, json, os, sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)


def meta(path):
    with safe_open(path, framework="pt") as h:
        return dict(h.metadata() or {})


def sha(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            d.update(b)
    return d.hexdigest()


def layout(sd):
    for k in list(sd):
        v = sd[k]
        if k.endswith(".weight") and v.ndim == 5:
            sd[k] = v.permute(0, 2, 3, 4, 1).contiguous()
        elif k.endswith(".weight") and v.ndim == 4:
            sd[k] = v.permute(0, 2, 3, 1).contiguous()
        elif k.endswith(".weight") and v.ndim == 3 and v.shape[1:] == (1, 1):
            sd[k] = v.reshape(-1).contiguous()
    return sd


wan_path, flash_path = f"{src}/wan.safetensors", f"{src}/flash.safetensors"
wm, fm = meta(wan_path), meta(flash_path)
assert wm["format"] == "refvsr-wan-v1" and fm["format"] == "refvsr-flash-v1"

wan = load_file(wan_path)
prompt = wan.pop("prompt_embeds")
assert {str(v.dtype) for v in wan.values()} == {"torch.bfloat16"}
save_file(layout(wan), f"{out}/generator_bf16.safetensors")
save_file({"prompt_embeds": prompt.contiguous()}, f"{out}/prompt_bf16.safetensors")
n_gen = sum(v.numel() for v in wan.values())
del wan

flash = load_file(flash_path)
assert {str(v.dtype) for v in flash.values()} == {"torch.float32"}
save_file(layout(flash), f"{out}/decoder_fp32.safetensors")
n_dec = sum(v.numel() for v in flash.values())

json.dump({
    "model": "RelayVSR",
    "source": {"repo": "kopper/RelayVSR", "code": "kopperx/RelayVSR", "license": "Apache-2.0",
               "base_model": "Wan-AI/Wan2.2-TI2V-5B (Apache-2.0)",
               "wan_sha256": sha(wan_path), "flash_sha256": sha(flash_path)},
    "generator": {"config": json.loads(wm["config"]), "training_step": wm.get("training_step"),
                  "weight_variant": wm.get("weight_variant"), "parameters": n_gen, "dtype": "bf16"},
    "decoder": {"config": json.loads(fm["config"]), "training_step": fm.get("training_step"),
                "weight_variant": fm.get("weight_variant"), "parameters": n_dec, "dtype": "fp32"},
    "prompt": {"shape": list(prompt.shape), "dtype": "bf16"},
    "sampling": {"num_train_timesteps": 1000, "shift": 5.0, "timesteps": [1000], "guidance_scale": 1.0},
    "layout": "keys verbatim; conv weights channels-last; lq_proj norm weights flattened",
}, open(f"{out}/config.json", "w"), indent=1)
print("generator", n_gen, "decoder", n_dec, "prompt", tuple(prompt.shape))
