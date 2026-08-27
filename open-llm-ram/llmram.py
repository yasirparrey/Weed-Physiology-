#!/usr/bin/env python3
"""Estimate the memory one inference call needs from an open-weight LLM.

"One call" means concurrency of 1: the weights, the KV cache for a single
sequence of the requested length, any fixed per-sequence recurrent state, and
the activation/workspace memory an inference engine allocates to run it. It is
the floor for serving the model at all, not a throughput-optimised footprint.

Usage:
    python llmram.py                              # leaderboard at defaults
    python llmram.py --context 128000 --weights fp8
    python llmram.py --model kimi-k3 --detail
    python llmram.py --markdown > LEADERBOARD.md
    python llmram.py --gpu h200 --context 1000000
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: pip install -r requirements.txt")

DATA_FILE = Path(__file__).with_name("models.yaml")

GIB = 1024**3
GB = 10**9

# Effective bytes per parameter. Sub-8-bit formats carry block scales, so MXFP4
# costs 4.25 bits and NVFP4/INT4 group-quantised formats about 4.5 bits.
WEIGHT_BYTES = {
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "int4": 4.5 / 8,
    "nvfp4": 4.5 / 8,
    "mxfp4": 4.25 / 8,
    "fp4_fp8_mixed": 4.5 / 8,
}

# Real checkpoints never quantise everything: embeddings, norms, routers, the LM
# head and (for VLMs) the vision tower usually stay at higher precision. These
# uplifts are calibrated against measured checkpoints - Kimi K3's MXFP4 release
# is 1,560.9 GB against 1,487.5 GB of naive math (+4.9%), and DeepSeek V4-Flash
# is 166.9 GB against 159.8 GB (+4.5%).
QUANT_UPLIFT = {"bf16": 1.0, "fp16": 1.0, "fp8": 1.02}
QUANT_UPLIFT_DEFAULT = 1.05

KV_BYTES = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0}

# Activation/workspace model. A chunked-prefill engine holds roughly
# CHUNK x hidden x 2 bytes x ACT_BUFFERS of transient activations, plus a flat
# allowance for CUDA context, graph capture and collective buffers. The 1 GB
# flat term reproduces the published ~22 GB total for Qwen3.5-35B-A3B at 4-bit.
DEFAULT_PREFILL_CHUNK = 8192
ACT_BUFFERS = 24
FIXED_OVERHEAD_GB = 1.0
DEFAULT_HIDDEN_SIZE = 4096  # only used when a model card omits hidden_size

# key -> (display name, memory GB, kind). Unified-memory hosts have the capacity
# to hold big models but a fraction of the bandwidth, so they are excluded when
# picking the "smallest thing this fits on" and only used when asked for by name.
GPUS = {
    "rtx4090": ("RTX 4090", 24, "discrete"),
    "rtx5090": ("RTX 5090", 32, "discrete"),
    "l40s": ("L40S", 48, "discrete"),
    "a100": ("A100 80GB", 80, "discrete"),
    "h100": ("H100 80GB", 80, "discrete"),
    "rtx6000pro": ("RTX 6000 Pro Blackwell", 96, "discrete"),
    "h200": ("H200", 141, "discrete"),
    "b200": ("B200", 180, "discrete"),
    "mi300x": ("MI300X", 192, "discrete"),
    "b300": ("B300", 288, "discrete"),
    "mi355x": ("MI355X", 288, "discrete"),
    "dgxspark": ("DGX Spark (unified)", 128, "unified"),
    "m3ultra": ("Mac Studio M3 Ultra (unified)", 512, "unified"),
}
# Engines cannot address 100% of a card. vLLM defaults to 0.90 and single-model
# deployments commonly raise it; Moonshot's own K3 sizing guide uses 0.95.
GPU_UTIL = 0.95


# --------------------------------------------------------------------------- data


@dataclass
class Breakdown:
    weights_gb: float
    kv_gb: float
    recurrent_gb: float
    activations_gb: float
    fixed_gb: float
    kv_bytes_per_token: float
    kv_source: str
    weights_precision: str
    weights_from_measurement: bool
    context: int

    @property
    def total_gb(self) -> float:
        return (
            self.weights_gb
            + self.kv_gb
            + self.recurrent_gb
            + self.activations_gb
            + self.fixed_gb
        )


@dataclass
class Model:
    raw: dict
    id: str
    name: str
    vendor: str
    total_params_b: float
    active_params_b: float
    license: str
    context_native: int
    native_precision: str
    released: str
    tier: str = "mid"
    checkpoint_gb: float | None = None
    scores: dict = field(default_factory=dict)
    arch: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        return cls(
            raw=d,
            id=d["id"],
            name=d["name"],
            vendor=d["vendor"],
            total_params_b=float(d["total_params_b"]),
            active_params_b=float(d["active_params_b"]),
            license=d.get("license", "unknown"),
            context_native=int(d.get("context_native", 32768)),
            native_precision=d.get("native_precision", "bf16"),
            released=str(d.get("released", "")),
            tier=d.get("tier", "mid"),
            checkpoint_gb=d.get("checkpoint_gb"),
            scores=d.get("scores") or {},
            arch=d.get("arch") or {},
        )

    @property
    def total_params(self) -> float:
        exact = self.arch.get("exact_param_count")
        return float(exact) if exact else self.total_params_b * 1e9

    @property
    def active_share(self) -> float:
        return self.active_params_b / self.total_params_b

    @property
    def hidden_size(self) -> int:
        return int(self.arch.get("hidden_size") or DEFAULT_HIDDEN_SIZE)

    @property
    def is_dense(self) -> bool:
        return abs(self.active_params_b - self.total_params_b) < 1e-9


def load_models(path: Path = DATA_FILE) -> tuple[list[Model], dict]:
    doc = yaml.safe_load(path.read_text())
    return [Model.from_dict(m) for m in doc["models"]], doc


# ---------------------------------------------------------------------- estimation


def weight_bytes_per_param(precision: str) -> float:
    try:
        return WEIGHT_BYTES[precision]
    except KeyError:
        raise SystemExit(
            f"unknown precision {precision!r}; choose from {', '.join(WEIGHT_BYTES)}"
        )


def weights_gb(model: Model, precision: str) -> tuple[float, bool]:
    """Return (gigabytes, whether the figure comes from a measured checkpoint)."""
    if precision == "native":
        precision = model.native_precision
        if model.checkpoint_gb:
            return float(model.checkpoint_gb), True
    uplift = QUANT_UPLIFT.get(precision, QUANT_UPLIFT_DEFAULT)
    return model.total_params * weight_bytes_per_param(precision) * uplift / GB, False


def kv_geometry(model: Model) -> tuple[float, float, float, str]:
    """Per-token KV bytes at 1 byte/element, plus fixed and precision-independent parts.

    Returns (elements_per_token, fixed_bytes_windowed_or_recurrent_at_2B,
             precision_independent_bytes_per_token, source).
    """
    kv = model.arch.get("kv") or {}
    source = kv.get("source", "estimated")

    if "bytes_per_token_bf16" in kv:
        # Stored at BF16 (2 bytes/element), so elements = bytes / 2.
        return float(kv["bytes_per_token_bf16"]) / 2.0, 0.0, 0.0, source

    per_token_elems = 0.0
    fixed_elems = 0.0
    flat_bytes = 0.0

    for spec in kv.get("growing", []) or []:
        per_token_elems += _layer_elems(spec) * spec["layers"]

    for spec in kv.get("windowed", []) or []:
        # A sliding-window layer holds at most `window` tokens, so its cost is
        # flat once the sequence exceeds the window.
        fixed_elems += _layer_elems(spec) * spec["layers"] * spec["window"]

    for spec in kv.get("recurrent", []) or []:
        # Linear-attention / delta-rule layers keep a heads x d_k x d_v matrix
        # per sequence, independent of sequence length.
        fixed_elems += (
            spec["heads"] * spec["head_k_dim"] * spec["head_v_dim"] * spec["layers"]
        )

    for spec in kv.get("fixed_rate", []) or []:
        flat_bytes += spec["bytes_per_token_per_layer"] * spec["layers"]

    return per_token_elems, fixed_elems, flat_bytes, source


def _layer_elems(spec: dict) -> float:
    kind = spec.get("kind", "gqa")
    if kind == "mla":
        # MLA caches one compressed latent plus the decoupled RoPE key.
        return spec["kv_lora_rank"] + spec["qk_rope_head_dim"]
    if kind == "gqa":
        return spec["kv_heads"] * (spec["k_head_dim"] + spec["v_head_dim"])
    raise SystemExit(f"unknown kv layer kind {kind!r}")


def estimate(
    model: Model,
    context: int,
    weights_precision: str = "int4",
    kv_precision: str = "fp8",
    recurrent_precision: str = "bf16",
    prefill_chunk: int = DEFAULT_PREFILL_CHUNK,
) -> Breakdown:
    w_gb, measured = weights_gb(model, weights_precision)

    elems_per_token, fixed_elems, flat_bytes, source = kv_geometry(model)
    kv_b = KV_BYTES[kv_precision]
    rec_b = KV_BYTES[recurrent_precision]

    kv_bytes = elems_per_token * kv_b * context + flat_bytes * context
    recurrent_bytes = fixed_elems * rec_b

    chunk = min(prefill_chunk, max(context, 1))
    act_gb = chunk * model.hidden_size * 2 * ACT_BUFFERS / GB

    return Breakdown(
        weights_gb=w_gb,
        kv_gb=kv_bytes / GB,
        recurrent_gb=recurrent_bytes / GB,
        activations_gb=act_gb,
        fixed_gb=FIXED_OVERHEAD_GB,
        kv_bytes_per_token=elems_per_token * kv_b + flat_bytes,
        kv_source=source,
        weights_precision=(
            model.native_precision if weights_precision == "native" else weights_precision
        ),
        weights_from_measurement=measured,
        context=context,
    )


def effective_context(model: Model, requested: int) -> tuple[int, bool]:
    """Clamp a requested context to what the model can actually accept.

    Returns (context, was_clamped). Models with a published YaRN-extended window
    are allowed up to that; otherwise the native window is the ceiling.
    """
    ceiling = int(model.raw.get("context_extended") or model.context_native)
    return (ceiling, True) if requested > ceiling else (requested, False)


def gpus_needed(total_gb: float, gpu_key: str) -> tuple[int, str]:
    name, mem, _kind = GPUS[gpu_key]
    n = math.ceil(total_gb / (mem * GPU_UTIL))
    return n, name


# ------------------------------------------------------------------------ printing


def fmt_params(b: float) -> str:
    if b >= 1000:
        return f"{b / 1000:.2f}T".replace(".00T", "T")
    if b >= 10:
        return f"{b:.0f}B"
    return f"{b:g}B"


def fmt_gb(v: float) -> str:
    if v >= 1000:
        return f"{v / 1000:.2f} TB"
    if v >= 10:
        return f"{v:.0f} GB"
    if v >= 0.1:
        return f"{v:.1f} GB"
    return f"{v * 1000:.0f} MB"


def fmt_ctx(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_048_576:.0f}M" if n % 1_048_576 == 0 else f"{n / 1e6:.1f}M"
    return f"{n // 1024}K"


SCORE_LABELS = {
    "gpqa_diamond": "GPQA-D",
    "aa_index": "AA Index",
    "swe_bench_verified": "SWE-bench V",
    "mmlu_pro": "MMLU-Pro",
}


def headline_metric(m: Model) -> tuple[str, float] | None:
    """Best available headline score and which benchmark it came from.

    The benchmarks are not on a common scale, so the metric name travels with
    the number rather than being collapsed into one ranking column.
    """
    for key in SCORE_LABELS:
        if key in m.scores:
            return SCORE_LABELS[key], float(m.scores[key])
    return None


def headline_score(m: Model) -> float | None:
    hit = headline_metric(m)
    return hit[1] if hit else None


def smallest_fit(total_gb: float) -> str:
    """Smallest single discrete accelerator that holds the call, else a GPU count."""
    for name, mem, kind in sorted(GPUS.values(), key=lambda g: g[1]):
        if kind == "discrete" and total_gb <= mem * GPU_UTIL:
            return f"1x {name}"
    n_h200, _ = gpus_needed(total_gb, "h200")
    n_b300, _ = gpus_needed(total_gb, "b300")
    return f"{n_h200}x H200 / {n_b300}x B300"


def print_table(models: list[Model], args, markdown: bool = False) -> None:
    rows = []
    any_clamped = False
    for m in models:
        ctx, clamped = effective_context(m, args.context)
        any_clamped = any_clamped or clamped
        b = estimate(
            m,
            context=ctx,
            weights_precision=args.weights,
            kv_precision=args.kv,
            prefill_chunk=args.prefill_chunk,
        )
        n_gpu, gpu_name = gpus_needed(b.total_gb, args.gpu)
        score = headline_score(m)
        rows.append(
            {
                "name": m.name,
                "vendor": m.vendor,
                "total": fmt_params(m.total_params_b),
                "active": fmt_params(m.active_params_b),
                "share": f"{m.active_share * 100:.1f}%",
                "license": m.license,
                "ctx": fmt_ctx(m.context_native) + ("!" if clamped else ""),
                "weights": fmt_gb(b.weights_gb),
                "kv": fmt_gb(b.kv_gb + b.recurrent_gb),
                "total_ram": fmt_gb(b.total_gb),
                "gpus": f"{n_gpu}x",
                "flag": "" if b.kv_source in ("config", "documented") else "*",
                "score": f"{score:g}" if score is not None else "-",
                "_sort": b.total_gb,
            }
        )

    headers = [
        ("name", "Model"),
        ("vendor", "Vendor"),
        ("total", "Total"),
        ("active", "Active"),
        ("share", "Act%"),
        ("ctx", "Context"),
        ("license", "License"),
        ("weights", "Weights"),
        ("kv", "KV+state"),
        ("total_ram", "RAM / call"),
        ("gpus", gpu_name),
        ("score", "Score"),
    ]

    def cell(row: dict, key: str) -> str:
        return row[key] + (row["flag"] if key == "kv" else "")

    if markdown:
        print("| " + " | ".join(h for _, h in headers) + " |")
        print("|" + "|".join(["---"] * len(headers)) + "|")
        for r in rows:
            print("| " + " | ".join(cell(r, k) for k, _ in headers) + " |")
    else:
        widths = {
            k: max(len(h), max(len(cell(r, k)) for r in rows)) for k, h in headers
        }
        print("  ".join(h.ljust(widths[k]) for k, h in headers))
        print("  ".join("-" * widths[k] for k, _ in headers))
        for r in rows:
            print("  ".join(cell(r, k).ljust(widths[k]) for k, _ in headers))

    print()
    print(
        f"Context {args.context:,} tokens, concurrency 1, weights {args.weights}, "
        f"KV cache {args.kv}. * = KV geometry estimated, not read from a config."
    )
    if any_clamped:
        print(
            "! = model's window is shorter than the requested context, so the "
            "figure is for its maximum instead."
        )


def print_detail(m: Model, args) -> None:
    print(f"{m.name}  ({m.vendor}, released {m.released})")
    print("=" * (len(m.name) + len(m.vendor) + 26))
    print(f"  license           {m.license}")
    print(f"  total parameters  {fmt_params(m.total_params_b)}")
    print(
        f"  active per token  {fmt_params(m.active_params_b)} "
        f"({m.active_share * 100:.1f}% of total)"
        + ("  [dense: every parameter, every token]" if m.is_dense else "")
    )
    print(f"  native precision  {m.native_precision}")
    print(f"  native context    {m.context_native:,} tokens")
    if m.arch.get("attention"):
        print(f"  attention         {m.arch['attention']}")
    if m.arch.get("experts"):
        e = m.arch["experts"]
        print(
            f"  experts           {e.get('total')} total, "
            f"{e.get('routed_per_token')} routed + {e.get('shared', 0)} shared per token"
        )
    print()

    contexts = sorted({4096, 32768, 131072, min(m.context_native, 1_048_576)})
    for precision in ("bf16", "fp8", "int4"):
        b0 = estimate(m, context=4096, weights_precision=precision, kv_precision=args.kv)
        print(f"  weights @ {precision:<5} {fmt_gb(b0.weights_gb):>10}")
    if m.checkpoint_gb:
        print(
            f"  weights @ native {fmt_gb(float(m.checkpoint_gb)):>10}"
            f"   (measured {m.native_precision} checkpoint)"
        )
    print()

    print("  RAM for one call")
    print(f"  {'context':>10}  {'weights':>10}  {'KV+state':>10}  {'act':>8}  {'total':>10}  {GPUS[args.gpu][0]}")
    for ctx in contexts:
        b = estimate(
            m,
            context=ctx,
            weights_precision=args.weights,
            kv_precision=args.kv,
            prefill_chunk=args.prefill_chunk,
        )
        n, _ = gpus_needed(b.total_gb, args.gpu)
        print(
            f"  {ctx:>10,}  {fmt_gb(b.weights_gb):>10}  "
            f"{fmt_gb(b.kv_gb + b.recurrent_gb):>10}  "
            f"{fmt_gb(b.activations_gb):>8}  {fmt_gb(b.total_gb):>10}  {n}x"
        )

    b = estimate(m, context=args.context, weights_precision=args.weights, kv_precision=args.kv)
    print()
    print(
        f"  KV cache costs {b.kv_bytes_per_token / 1024:.1f} KiB/token at {args.kv} "
        f"(geometry: {b.kv_source})"
    )
    if b.recurrent_gb > 0:
        print(
            f"  Fixed per-sequence state (recurrent / sliding-window): "
            f"{fmt_gb(b.recurrent_gb)}, independent of context length"
        )
    kv = m.arch.get("kv") or {}
    if kv.get("note"):
        print(f"  note: {' '.join(kv['note'].split())}")
    if m.raw.get("notes"):
        print()
        print("  " + "\n  ".join(_wrap(" ".join(m.raw["notes"].split()), 76)))


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


# ---------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RAM/VRAM required for one inference call to an open-weight LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", help="show a detailed breakdown for one model id")
    p.add_argument("--detail", action="store_true", help="force detailed view")
    p.add_argument("--context", type=int, default=32768, help="prompt+output tokens for the call")
    p.add_argument(
        "--weights",
        default="int4",
        choices=[*WEIGHT_BYTES, "native"],
        help="weight precision to serve at",
    )
    p.add_argument("--kv", default="fp8", choices=list(KV_BYTES), help="KV cache precision")
    p.add_argument("--gpu", default="h200", choices=list(GPUS), help="accelerator to size against")
    p.add_argument("--prefill-chunk", type=int, default=DEFAULT_PREFILL_CHUNK)
    p.add_argument("--tier", help="filter by tier (frontier, strong, mid, small, legacy)")
    p.add_argument("--max-ram", type=float, help="only show models fitting in this many GB")
    p.add_argument("--sort", default="params", choices=["params", "ram", "score", "active"])
    p.add_argument("--markdown", action="store_true", help="emit a markdown table")
    p.add_argument("--list-gpus", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_gpus:
        for key, (name, mem, kind) in GPUS.items():
            print(f"  {key:<12} {name:<32} {mem:>4} GB  {kind}")
        return 0

    models, _ = load_models()

    if args.model:
        matches = [m for m in models if m.id == args.model]
        if not matches:
            close = [m.id for m in models if args.model.lower() in m.id.lower()]
            print(f"no model with id {args.model!r}." + (f" Did you mean: {', '.join(close)}?" if close else ""))
            return 1
        print_detail(matches[0], args)
        return 0

    if args.tier:
        models = [m for m in models if m.tier == args.tier]

    def total_for(m: Model) -> float:
        ctx, _ = effective_context(m, args.context)
        return estimate(m, ctx, args.weights, args.kv, prefill_chunk=args.prefill_chunk).total_gb

    if args.max_ram:
        models = [m for m in models if total_for(m) <= args.max_ram]

    if args.sort == "params":
        models.sort(key=lambda m: -m.total_params_b)
    elif args.sort == "active":
        models.sort(key=lambda m: -m.active_params_b)
    elif args.sort == "ram":
        models.sort(key=total_for)
    else:
        models.sort(key=lambda m: -(headline_score(m) or 0))

    if not models:
        print("no models match those filters")
        return 1

    print_table(models, args, markdown=args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
