#!/usr/bin/env python3
"""Render LEADERBOARD.md from models.yaml so the tables can never drift from the data.

    python generate_leaderboard.py > LEADERBOARD.md
"""

from __future__ import annotations

import datetime as dt

from llmram import (
    GPU_UTIL,
    Model,
    estimate,
    fmt_ctx,
    fmt_gb,
    fmt_params,
    headline_metric,
    headline_score,
    load_models,
    smallest_fit,
)

CTX_DEFAULT = 32_768
KIB = 1024

# Hardware budgets people actually have, smallest first.
BUDGETS = [
    ("Laptop / 8 GB GPU", 8),
    ("RTX 4090 / 5090 (24-32 GB)", 24),
    ("RTX 6000 Pro (96 GB)", 96),
    ("1x H100 / A100 (80 GB)", 80),
    ("1x H200 (141 GB)", 141),
    ("Mac Studio M3 Ultra (512 GB unified)", 512),
    ("8x H100 node (640 GB)", 640),
    ("8x H200 node (1,128 GB)", 1128),
    ("8x B300 node (2,304 GB)", 2304),
]


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def kv_flag(m: Model) -> str:
    src = (m.arch.get("kv") or {}).get("source", "estimated")
    return "" if src in ("config", "documented") else " \\*"


def main() -> None:
    models, doc = load_models()
    as_of = doc["meta"]["as_of"]

    print(f"""# Open-Weight LLM Leaderboard, with the RAM one call actually costs

Generated from [`models.yaml`](models.yaml) by
[`generate_leaderboard.py`](generate_leaderboard.py). Data as of **{as_of}**.
Do not hand-edit; regenerate with `python generate_leaderboard.py > LEADERBOARD.md`.

## The one thing to understand first

Total parameters and active parameters answer two different questions:

- **Total parameters** decide **how much memory you need**. Every weight has to
  be resident somewhere the accelerator can reach, because any token might route
  to any expert.
- **Active parameters** decide **how fast it runs and what it costs per token**.
  This is the slice of the network multiplied for a given token.

A Mixture-of-Experts model with 2.8T total and 104B active is as cheap to compute
as a 104B dense model and as expensive to hold in memory as a 2.8T one. That gap
is the single most misread number in the open-weight ecosystem: {fmt_params(models[0].total_params_b)} of
weights at 3.7% activation buys you frontier quality at mid-size speed, and a
memory bill that no single GPU on the market can pay.

Sparse models have gotten sparser, not smaller. The frontier tier now sits at
3-6% activation, which is why "runs fast" and "fits on my box" have completely
decoupled.

## How "RAM for one call" is defined

One call means **concurrency of 1**: a single request, holding

```
weights + KV cache(context) + fixed per-sequence state + activations + workspace
```

This is the floor to run the model at all. Serving many users concurrently adds
one KV cache and one recurrent state per in-flight request, so a production
deployment needs materially more. The estimator's assumptions:

| Component | Model used |
|---|---|
| Weights | `total_params x bytes/param`, uplifted for the parts real checkpoints leave unquantized (+5% below 8-bit, +2% at FP8) |
| MLA KV cache | `(kv_lora_rank + qk_rope_head_dim) x growing_layers` elements per token |
| GQA KV cache | `kv_heads x (k_head_dim + v_head_dim) x growing_layers` elements per token |
| Sliding-window layers | capped at `window` tokens, so flat in context |
| Linear-attention layers | `heads x d_k x d_v` per layer per sequence, flat in context |
| Activations | `prefill_chunk x hidden x 2 bytes x 24` |
| Workspace | 1 GB flat for CUDA context, graphs and collective buffers |
| GPU headroom | usable memory is {GPU_UTIL:.0%} of nameplate |

Those constants are not guesses; they are pinned by
[`test_llmram.py`](test_llmram.py) against published measurements. The estimator
reproduces Kimi K3's 13.5 KiB/token FP8 cache and its measured 1,560.9 GB MXFP4
checkpoint, GLM-5.2's 11.8 GiB cache at 262K context, DeepSeek V4-Flash's
166.9 GB repository, and NVIDIA's stated 4x B200 minimum for Nemotron 3 Ultra.

Rows marked `*` have a KV geometry inferred from the attention family rather than
read from a published config. Their weight figures are still solid; treat their
cache figures as +/- 50%.
""")

    # ---------------------------------------------------------------- main board
    ranked = sorted(models, key=lambda m: -(headline_score(m) or -1))
    scored = [m for m in ranked if headline_score(m) is not None]

    print(f"""## Leaderboard: capability against memory

Grouped by benchmark, because these scores are **not on a common scale**: a GPQA
Diamond percentage and an Artificial Analysis Intelligence Index score cannot be
ranked against each other. Within each group, higher is better. RAM is for one
call at {CTX_DEFAULT:,} tokens of context, 4-bit weights, FP8 KV cache.
""")

    by_metric: dict[str, list[Model]] = {}
    for m in scored:
        by_metric.setdefault(headline_metric(m)[0], []).append(m)

    for metric, group in by_metric.items():
        group.sort(key=lambda m: -headline_score(m))
        print(f"\n### Ranked by {metric}\n")
        rows = []
        for m in group:
            b = estimate(m, CTX_DEFAULT, "int4", "fp8")
            rows.append(
                [
                    f"**{m.name}**",
                    m.vendor,
                    f"{headline_score(m):g}",
                    fmt_params(m.total_params_b),
                    fmt_params(m.active_params_b),
                    f"{m.active_share * 100:.1f}%",
                    fmt_gb(b.total_gb) + kv_flag(m),
                    smallest_fit(b.total_gb),
                    m.license,
                ]
            )
        print(
            table(
                [
                    "Model",
                    "Vendor",
                    metric,
                    "Total params",
                    "Active params",
                    "Active %",
                    "RAM / call",
                    "Fits on",
                    "License",
                ],
                rows,
            )
        )

    # ------------------------------------------------------------- full roster
    print(f"""
## Every model, by size, at three serving precisions

RAM for one call at {CTX_DEFAULT:,} tokens. `Native` uses the measured official
checkpoint where one is published, otherwise the model's shipped precision.
""")
    rows = []
    for m in sorted(models, key=lambda x: -x.total_params_b):
        cells = [f"**{m.name}**", fmt_params(m.total_params_b), fmt_params(m.active_params_b)]
        for prec in ("bf16", "fp8", "int4"):
            cells.append(fmt_gb(estimate(m, CTX_DEFAULT, prec, "fp8").total_gb))
        nat = estimate(m, CTX_DEFAULT, "native", "fp8")
        cells.append(fmt_gb(nat.total_gb) + f" ({nat.weights_precision})")
        cells.append(fmt_ctx(m.context_native))
        cells.append(m.arch.get("attention", "-"))
        rows.append(cells)
    print(
        table(
            ["Model", "Total", "Active", "BF16", "FP8", "4-bit", "Native", "Context", "Attention"],
            rows,
        )
    )

    # -------------------------------------------------------------- KV cache cost
    print("""
## The long-context bill: KV cache per token

Weights are a fixed cost; the KV cache is what turns a 32K call into a 1M call.
Architecture matters more than size here. Hy3 is a 295B model that pays 160
KiB/token because it runs plain GQA over 80 layers. Qwen3.5-397B is a *larger*
model that pays 15 KiB/token because only 15 of its 60 layers keep a growing
cache. At 262K context that is a 43 GB cache versus a 4 GB one.
""")
    kv_rows = []
    for m in sorted(models, key=lambda x: -estimate(x, 1024, "int4", "fp8").kv_bytes_per_token):
        b = estimate(m, 1024, "int4", "fp8")
        if b.kv_bytes_per_token <= 0:
            continue
        full = min(m.context_native, 1_048_576)
        at_full = estimate(m, full, "int4", "fp8")
        at_128k = (
            fmt_gb(estimate(m, 131_072, "int4", "fp8").kv_gb)
            if m.context_native >= 131_072
            else "n/a"
        )
        kv_rows.append(
            [
                f"**{m.name}**",
                m.arch.get("attention", "-"),
                f"{b.kv_bytes_per_token / KIB:.1f} KiB",
                at_128k,
                f"{fmt_gb(at_full.kv_gb)} @ {fmt_ctx(full)}",
                fmt_gb(b.recurrent_gb) if b.recurrent_gb > 0.05 else "-",
                (m.arch.get("kv") or {}).get("source", "estimated"),
            ]
        )
    print(
        table(
            [
                "Model",
                "Attention",
                "KV / token (FP8)",
                "KV @ 128K",  # n/a where the model's native context is shorter
                "KV @ full context",
                "Fixed state",
                "Source",
            ],
            kv_rows[:24],
        )
    )
    print("\n(Truncated to the 24 most cache-hungry models; run "
          "`python llmram.py --model <id>` for any single model.)")

    # ------------------------------------------------------------ what fits where
    print("""
## What fits on what

For each hardware budget, the largest and the highest-scoring model that fits one
call at 32K context in 4-bit. "Fits" means the total lands inside the usable
fraction of nameplate memory.
""")
    rows = []
    for label, mem in BUDGETS:
        usable = mem * GPU_UTIL
        fitting = [m for m in models if estimate(m, CTX_DEFAULT, "int4", "fp8").total_gb <= usable]
        if not fitting:
            rows.append([label, f"{usable:.0f} GB", "nothing here fits", "-"])
            continue
        biggest = max(fitting, key=lambda m: m.total_params_b)
        best = max(fitting, key=lambda m: headline_score(m) or -1)
        rows.append(
            [
                label,
                f"{usable:.0f} GB",
                f"{biggest.name} ({fmt_params(biggest.total_params_b)})",
                f"{best.name}"
                + (f" ({headline_score(best):g})" if headline_score(best) is not None else ""),
            ]
        )
    print(table(["Budget", "Usable", "Largest that fits", "Best-scoring that fits"], rows))

    # ------------------------------------------------------------------- excluded
    print("""
## Not included, and why

"Open source LLM" leaderboards routinely list models whose weights you cannot
download. These are excluded here on purpose:
""")
    for e in doc.get("excluded", []):
        print(f"- **{e['name']}** — {' '.join(e['reason'].split())}")

    print(f"""
## Caveats worth reading before you spend money

1. **Quantization is not free.** 4-bit is the assumed default above because it is
   what most people actually deploy, but it costs accuracy, and how much depends
   on the model. Models trained quantization-aware (Kimi K3 in MXFP4, DeepSeek V4
   in FP4/FP8, gpt-oss in MXFP4) lose far less than models squeezed after the
   fact. For those, the `Native` column is the honest number.
2. **Fitting is not serving.** These figures are concurrency 1. Every additional
   in-flight request adds a KV cache and a recurrent state. A deployment sized to
   the numbers here will run one user at a time, slowly.
3. **Memory bandwidth, not capacity, sets your token rate.** Active parameters
   times bandwidth is the rough ceiling on decode speed. A model that barely fits
   will usually also be too slow to enjoy.
4. **Aggregate memory is not the same as usable memory.** Sharding a model across
   GPUs costs replicated attention, activations and collective buffers, and MoE
   expert parallelism needs the expert-routing communication buffers on top.
5. **Licenses vary more than the "open source" label suggests.** Apache 2.0 and
   MIT are unrestricted. Llama 4's community license caps you at 700M monthly
   active users. Cohere's Command A+ is non-commercial. Read before shipping.
6. **This is a snapshot.** The frontier moved from 671B dense-ish MoE to 2.8T in
   about eighteen months. Regenerate rather than trust the timestamp.

---
Generated {dt.date.today().isoformat()} from data as of {as_of}.
""")


if __name__ == "__main__":
    main()
