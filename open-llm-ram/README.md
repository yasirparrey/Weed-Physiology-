# Open-weight LLM leaderboard + inference memory calculator

A curated database of every notable open-weight LLM with its **total parameters**,
**active parameters per token**, and the **RAM one inference call actually
requires** — plus a calculator so you can recompute any of it for your own
context length, precision and hardware.

**Start here: [LEADERBOARD.md](LEADERBOARD.md).**

## Why both parameter counts matter

They answer different questions, and conflating them is the most common costing
mistake in the open-weight ecosystem:

| | Decides | Kimi K3 |
|---|---|---|
| **Total parameters** | how much memory you must have | 2.8T → ~1.6 TB at 4-bit |
| **Active parameters** | how fast it runs, cost per token | 104B → mid-size speed |

Every weight in a Mixture-of-Experts model has to be resident, because any token
might route to any expert. So Kimi K3 computes like a 104B model and has to be
stored like a 2.8T one. "Only 3.7% active" is a statement about FLOPs, never
about memory.

## Quick start

```bash
pip install -r requirements.txt

# the whole board, one call at 32K context, 4-bit weights
python llmram.py

# what a specific model costs, at several context lengths and precisions
python llmram.py --model kimi-k3
python llmram.py --model glm-52 --gpu b300

# what can I run on one 24 GB card?
python llmram.py --gpu rtx4090 --max-ram 22 --context 8192

# everything one machine can run, with decode speed and time-to-first-token
python llmram.py --host m3ultra --context 32768
python llmram.py --host dgxspark

# I run X and want better without losing speed - what should I switch to?
python llmram.py --upgrade-from qwen3-coder-30b-a3b --host m3ultra

# long-context serving on an 8x H200 node
python llmram.py --context 1000000 --weights fp8 --gpu h200

# regenerate the report
python generate_leaderboard.py > LEADERBOARD.md
```

Useful flags: `--context`, `--weights {bf16,fp8,int4,nvfp4,mxfp4,q3,q2,native}`,
`--kv {fp32,bf16,fp8}`, `--gpu` / `--host` (see `--list-gpus`), `--tier`,
`--max-ram`, `--sort {params,ram,score,active}`, `--markdown`.

## Capacity is not the same as usability

`--host` answers "what can this machine actually run" rather than "what fits",
because three different limits bind at different points:

- **Capacity** — total parameters. Sets what loads at all.
- **Bandwidth** — decode reads every *active* parameter once per token, so
  tokens/second is bandwidth ÷ active-parameter bytes. Total parameters do not
  appear. This is why a sparse 400B model generates faster than a dense 253B one.
- **Compute** — prefill costs ~2 FLOPs per active parameter per token, so it sets
  time-to-first-token on a long prompt. This is the limit that bites
  unified-memory hardware, and the one most "can it run X" tables ignore.

On a 512 GB Mac Studio the consequence is stark: GLM-5.2 fits and generates at
about 11 tok/s, but takes over three minutes to read a 32K prompt. Qwen3.5-35B-A3B
fits twenty times over, generates at ~146 tok/s, and reads the same prompt in 12
seconds. Both "fit".

Decode efficiency is modelled separately for dense and MoE models (62% vs 30% of
theoretical bandwidth) because MoE decode does small gathered GEMMs over
scattered experts. Both figures are calibrated against measured Apple Silicon
benchmarks, not assumed.

The practical consequence, which `--upgrade-from` ranks directly: when memory is
abundant and bandwidth is scarce, **total parameters are nearly free and active
parameters are what you pay for**. Ling 3.0 Flash (124B/5.1B) beats DeepSeek
V4-Flash (284B/13B) on SWE-bench Pro while reading prompts 2.5× faster, and
Qwen3-Coder-Next (80B/3B) is a strictly free upgrade over Qwen3-Coder-30B-A3B
(30B/3.3B) — more than twice the total parameters at identical speed.

## What "RAM for one call" includes

Concurrency of 1 — a single request, start to finish:

```
weights + KV cache(context) + fixed per-sequence state + activations + workspace
```

This is the floor to run the model at all, not a production serving target. Each
additional concurrent request adds another KV cache and recurrent state.

The interesting part is that the KV cache term is driven by **architecture, not
size**. Three examples from the database:

- **Hy3 (295B)** runs plain GQA over 80 layers: 160 KiB/token, so a 256K-token
  call needs 43 GB of cache on top of the weights.
- **Qwen3.5-397B** is bigger but keeps a growing cache on only 15 of its 60
  layers (the rest are Gated DeltaNet with fixed-size recurrent state):
  15 KiB/token, 4 GB at 262K.
- **Kimi K3 (2.8T)** costs 13.5 KiB/token at FP8 thanks to 24 gated-MLA layers
  and 69 KDA layers, so its full 1M-token context is only 13.5 GiB. Its problem
  is entirely the 1.56 TB of weights.

## Is the estimator trustworthy?

It is calibrated against published measurements, and those checks run as tests:

```bash
python -m pytest test_llmram.py -q     # 37 passed
```

It reproduces, without per-model fudge factors:

| Published figure | Source | Estimator |
|---|---|---|
| Kimi K3 KV cache 13.5 KiB/token FP8, 27.0 KiB BF16 | Particula sizing analysis | matches to 1% |
| Kimi K3 full 1M context = 13.5 GiB | same | matches to 2% |
| Kimi K3 MXFP4 checkpoint = 1,560.9 GB | 96 safetensors shards, measured | 1,561.9 GB computed |
| GLM-5.2 262K session = 11.8 GiB, 1M = 47.3 GiB | GLM-5.2 parameterization note | matches to 5% |
| DeepSeek V4-Flash repo = 166.9 GB | HF repository | 167.7 GB computed |
| Nemotron 3 Ultra minimum 4x B200 / 8x H100 | NVIDIA NIM model card | matches |
| gpt-oss-120b on one 80 GB GPU | OpenAI | matches |
| Qwen3.5-35B-A3B ≈ 22 GB at Q4 on an RTX 4090 | Qwen setup guides | matches |
| Qwen3.8-27B ≈ 56 / 28 / 14–17 GB at BF16 / FP8 / 4-bit | Qwen model card | matches to 5% |
| Qwen3.8-27B Gated DeltaNet state ≈ 150 MB | architecture analysis | 151 MB at FP32 |
| Kimi K2.5 GGUF: UD-Q2_K_XL 375 GB, UD-TQ1_0 240 GB | Unsloth quant sizes | matches to 5% |
| DeepSeek V4-Flash MLX peak 147–159 GB on M3 Ultra | oMLX benchmarks | 161 GB computed |
| DeepSeek V4-Flash 25 tok/s baseline, 35–43 with MTP | oMLX + maclocal-api | 34 tok/s |
| Kimi K2.5 8–21 tok/s on a 512 GB Mac Studio | community reports | 14–20 tok/s |
| ~127 tok/s for a 7B at Q4 on M3 Ultra | llmrun device data | ~100 tok/s at 9B |

The one calibration worth knowing about: sub-8-bit formats are modelled at
4.25–4.5 effective bits plus a 5% uplift, because real checkpoints leave
embeddings, norms, routers, the LM head and vision towers at higher precision.
That is why Kimi K3's MXFP4 release is 11.5% larger than naive `2.8T x 4 bits`.

## Files

| File | Purpose |
|---|---|
| [`models.yaml`](models.yaml) | The database: parameters, licenses, contexts, benchmark scores, and per-model KV geometry with a provenance tag on each entry |
| [`llmram.py`](llmram.py) | Calculator and CLI |
| [`generate_leaderboard.py`](generate_leaderboard.py) | Renders `LEADERBOARD.md` from the database |
| [`test_llmram.py`](test_llmram.py) | Calibration tests against published measurements |
| [`LEADERBOARD.md`](LEADERBOARD.md) | Generated report |

## Data quality

Each model's KV geometry carries a `source` tag:

- `config` — read off the published `config.json` or model card. Trust it.
- `documented` — stated in a vendor or vendor-adjacent write-up.
- `estimated` — inferred from the attention family and scale. Marked `*` in the
  tables; treat as ±50%. Weight figures for these models are still solid, since
  they only depend on the parameter count.

Parameter counts are cross-checked against vendor model cards, Hugging Face
configs, and the LLM Architecture Gallery's active-parameter table. Where sources
disagree — GLM-5.2 is the worst case at 744B official, 753B on NVIDIA's card and
743B computed by vLLM — the choice is recorded in the model's `notes`.

Models that appear on "open source LLM" leaderboards but do not ship weights
(Qwen3.8 Max, the full Inkling) are listed in the `excluded` section of
`models.yaml` rather than silently dropped.

## Adding or correcting a model

Edit `models.yaml`, then:

```bash
python -m pytest test_llmram.py -q
python generate_leaderboard.py > LEADERBOARD.md
```

A model needs at minimum `id`, `name`, `vendor`, `total_params_b`,
`active_params_b`, `license`, `context_native`, `native_precision`, and a
`arch.kv` block — either real layer geometry or a `bytes_per_token_bf16`
estimate with `source: estimated`.
