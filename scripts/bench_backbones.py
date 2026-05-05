"""Per-backbone observe()/img_step throughput benchmark.

Measures the wall-clock cost of the per-step training-time hot path for each
backbone, with optional ``torch.compile``. Use this BEFORE deciding whether
to vectorize the per-step Python loop in ``CategoricalRSSM.observe`` —
"measure first, optimize second."

Usage::

    # CPU baseline:
    python scripts/bench_backbones.py

    # GPU with compile (run on cluster):
    python scripts/bench_backbones.py --device cuda --compile

    # Production-shape config:
    python scripts/bench_backbones.py --device cuda --compile \
        --batch-size 16 --batch-length 64 --deter 2048 --tokens 8

Output: per-backbone median observe() wall-clock (seconds) and steps/sec
(``batch_size · batch_length / wall_clock``). Lower wall-clock = better.

Notes:
- CPU numbers are not representative of cluster (no CUDA graphs, no compiled
  GPU kernels). Use for debugging the relative ordering of backbones, not
  absolute throughput.
- ``torch.compile`` warmup takes 30+ seconds on first invocation. The reported
  median excludes the warmup runs.
- The benchmark uses ``observe`` directly with random inputs; it does not run
  the full Dreamer training step (no losses, no backward, no replay buffer).
  For end-to-end throughput see ``scripts/smoke_throughput.sh``.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from omegaconf import OmegaConf

from rssm_factory import build_rssm


torch.set_float32_matmul_precision("high")

# Backbones to benchmark. We include both alias (mamba2) and base (mamba) for
# the SSD/Mamba pair so the user can verify they share an implementation.
BACKBONES = ("gru", "transformer", "mamba2", "s4", "s5")


def make_config(
    *,
    backbone: str,
    deter: int,
    tokens: int,
    transformer_cache_length: int,
    mamba_state_size: int,
    mamba_nheads: int,
    s4_state_size: int,
    s5_heads: int,
    s5_state_size: int,
    device: str,
):
    """Build a minimal Hydra-style config for a single backbone."""
    return OmegaConf.create({
        "backbone": backbone,
        "rssm": {
            "stoch": 32, "deter": deter, "hidden": 1024, "discrete": 32,
            "img_layers": 2, "obs_layers": 1, "dyn_layers": 1, "blocks": 8,
            "act": "SiLU", "norm": True, "unimix_ratio": 0.01,
            "initial": "learned", "device": device,
        },
        "transformer": {
            "tokens": tokens, "layers": 2, "heads": 4, "ff_mult": 4,
            "cache_length": transformer_cache_length,
        },
        "mamba": {
            "tokens": tokens, "layers": 2, "expand": 2, "conv_kernel": 3,
            "state_size": mamba_state_size, "nheads": mamba_nheads,
        },
        "s4": {"tokens": tokens, "layers": 2, "state_size": s4_state_size},
        "s5": {"tokens": tokens, "layers": 2, "heads": s5_heads, "state_size": s5_state_size},
    })


def bench_backbone(
    backbone: str,
    *,
    device: torch.device,
    batch_size: int,
    batch_length: int,
    deter: int,
    tokens: int,
    embed_size: int,
    act_dim: int,
    use_compile: bool,
    n_warmup: int,
    n_repeat: int,
    transformer_cache_length: int,
    mamba_state_size: int,
    mamba_nheads: int,
    s4_state_size: int,
    s5_heads: int,
    s5_state_size: int,
) -> dict:
    """Build the RSSM, run observe() repeatedly, return median wall-clock.

    Returns a dict with ``median_s``, ``mean_s``, ``min_s``, ``steps_per_sec``.
    """
    config = make_config(
        backbone=backbone, deter=deter, tokens=tokens,
        transformer_cache_length=transformer_cache_length,
        mamba_state_size=mamba_state_size, mamba_nheads=mamba_nheads,
        s4_state_size=s4_state_size,
        s5_heads=s5_heads, s5_state_size=s5_state_size,
        device=str(device),
    )
    rssm = build_rssm(config, embed_size=embed_size, act_dim=act_dim).to(device)
    rssm.eval()

    fn = rssm.observe
    if use_compile:
        # mode="reduce-overhead" matches what dreamer.py uses for _cal_grad.
        # It captures cudagraphs on CUDA; on CPU it falls back to default.
        fn = torch.compile(fn, mode="reduce-overhead")

    # Random inputs.
    embed = torch.randn(batch_size, batch_length, embed_size, device=device)
    action = torch.randn(batch_size, batch_length, act_dim, device=device)
    reset = torch.zeros(batch_size, batch_length, dtype=torch.bool, device=device)
    reset[:, 0] = True
    initial = rssm.initial(batch_size)

    # Warmup (compile traces happen here; not counted in timing).
    with torch.no_grad():
        for _ in range(n_warmup):
            out = fn(embed, action, initial, reset)
            if device.type == "cuda":
                torch.cuda.synchronize()
            del out

    # Timed runs.
    times = []
    with torch.no_grad():
        for _ in range(n_repeat):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(embed, action, initial, reset)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            del out

    median_s = statistics.median(times)
    mean_s = statistics.mean(times)
    min_s = min(times)
    steps_per_sec = (batch_size * batch_length) / median_s
    return dict(
        median_s=median_s, mean_s=mean_s, min_s=min_s,
        steps_per_sec=steps_per_sec, n_repeat=len(times),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--compile", dest="use_compile", action="store_true",
                    help="Wrap observe() with torch.compile(mode='reduce-overhead')")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--batch-length", type=int, default=64)
    ap.add_argument("--deter", type=int, default=256,
                    help="Deter dim. Production: 2048; small CPU bench: 256.")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--embed-size", type=int, default=128)
    ap.add_argument("--act-dim", type=int, default=4)
    ap.add_argument("--n-warmup", type=int, default=2)
    ap.add_argument("--n-repeat", type=int, default=5)
    ap.add_argument("--transformer-cache-length", type=int, default=128,
                    help="Paper-faithful long-memory transformer needs >= batch_length.")
    ap.add_argument("--mamba-state-size", type=int, default=64,
                    help="Mamba-2 paper-typical d_state.")
    ap.add_argument("--mamba-nheads", type=int, default=8,
                    help="Mamba-2 default head count.")
    ap.add_argument("--s4-state-size", type=int, default=64)
    ap.add_argument("--s5-heads", type=int, default=4)
    ap.add_argument("--s5-state-size", type=int, default=32)
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    print(f"# Backbone observe() benchmark", flush=True)
    print(f"#   device={device}, compile={args.use_compile}", flush=True)
    print(f"#   batch_size={args.batch_size}, batch_length={args.batch_length}", flush=True)
    print(f"#   deter={args.deter}, tokens={args.tokens}", flush=True)
    print(f"#   transformer.cache_length={args.transformer_cache_length}", flush=True)
    print(f"#   mamba.state_size={args.mamba_state_size}, mamba.nheads={args.mamba_nheads}", flush=True)
    print(f"#   s4.state_size={args.s4_state_size}", flush=True)
    print(f"#   s5.heads={args.s5_heads}, s5.state_size={args.s5_state_size}", flush=True)
    print(f"#   warmup={args.n_warmup}, repeat={args.n_repeat}", flush=True)
    print(flush=True)
    header = f"{'backbone':>12} | {'median (ms)':>12} | {'min (ms)':>10} | {'steps/sec':>11}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for backbone in args.backbones:
        try:
            r = bench_backbone(
                backbone, device=device,
                batch_size=args.batch_size, batch_length=args.batch_length,
                deter=args.deter, tokens=args.tokens,
                embed_size=args.embed_size, act_dim=args.act_dim,
                use_compile=args.use_compile,
                n_warmup=args.n_warmup, n_repeat=args.n_repeat,
                transformer_cache_length=args.transformer_cache_length,
                mamba_state_size=args.mamba_state_size,
                mamba_nheads=args.mamba_nheads,
                s4_state_size=args.s4_state_size,
                s5_heads=args.s5_heads, s5_state_size=args.s5_state_size,
            )
            print(
                f"{backbone:>12} | {r['median_s']*1000:>12.2f} | "
                f"{r['min_s']*1000:>10.2f} | {r['steps_per_sec']:>11.1f}",
                flush=True,
            )
        except Exception as exc:
            print(f"{backbone:>12} | FAILED: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
