# Open-Weight LLM Leaderboard, with the RAM one call actually costs

Generated from [`models.yaml`](models.yaml) by
[`generate_leaderboard.py`](generate_leaderboard.py). Data as of **2026-08-27**.
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
is the single most misread number in the open-weight ecosystem: 2.80T of
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
| GPU headroom | usable memory is 95% of nameplate |

Those constants are not guesses; they are pinned by
[`test_llmram.py`](test_llmram.py) against published measurements. The estimator
reproduces Kimi K3's 13.5 KiB/token FP8 cache and its measured 1,560.9 GB MXFP4
checkpoint, GLM-5.2's 11.8 GiB cache at 262K context, DeepSeek V4-Flash's
166.9 GB repository, and NVIDIA's stated 4x B200 minimum for Nemotron 3 Ultra.

Rows marked `*` have a KV geometry inferred from the attention family rather than
read from a published config. Their weight figures are still solid; treat their
cache figures as +/- 50%.

## Leaderboard: capability against memory

Grouped by benchmark, because these scores are **not on a common scale**: a GPQA
Diamond percentage and an Artificial Analysis Intelligence Index score cannot be
ranked against each other. Within each group, higher is better. RAM is for one
call at 32,768 tokens of context, 4-bit weights, FP8 KV cache.


### Ranked by GPQA-D

| Model | Vendor | GPQA-D | Total params | Active params | Active % | RAM / call | Fits on | License |
|---|---|---|---|---|---|---|---|---|
| **Kimi K3** | Moonshot AI | 93.5 | 2.80T | 104B | 3.7% | 1.66 TB | 13x H200 / 7x B300 | Modified MIT |
| **MiniMax M3** | MiniMax | 93 | 427B | 23B | 5.4% | 256 GB \* | 1x B300 | MIT |
| **Qwen3.8-2.4T-A95B (open Qwen3.8 Max)** | Alibaba (Qwen) | 92.6 | 2.40T | 95B | 4.0% | 1.42 TB \* | 11x H200 / 6x B300 | Qwen custom (commercial agreement above $50M revenue) |
| **GLM-5.2** | Z.ai (Zhipu AI) | 91.2 | 753B | 40B | 5.3% | 450 GB | 4x H200 / 2x B300 | MIT |
| **Kimi K2.6** | Moonshot AI | 90.5 | 1T | 32B | 3.2% | 596 GB | 5x H200 / 3x B300 | Modified MIT |
| **Hy3 295B-A21B** | Tencent Hunyuan | 90.4 | 295B | 21B | 7.1% | 182 GB | 1x MI300X | Tencent Hunyuan Community |
| **DeepSeek V4-Pro** | DeepSeek | 90.1 | 1.60T | 49B | 3.1% | 948 GB \* | 8x H200 / 4x B300 | MIT |
| **Kimi K2.7 Code** | Moonshot AI | 89.6 | 1T | 32B | 3.2% | 596 GB | 5x H200 / 3x B300 | Modified MIT |
| **Inkling-Small** | Thinking Machines Lab | 89.5 | 276B | 12B | 4.3% | 166 GB \* | 1x B200 | Apache 2.0 |
| **Qwen3.5-397B-A17B** | Alibaba (Qwen) | 88.4 | 397B | 17B | 4.3% | 238 GB | 1x B300 | Apache 2.0 |
| **DeepSeek V4-Flash** | DeepSeek | 88.1 | 284B | 13B | 4.6% | 171 GB | 1x MI300X | MIT |
| **Kimi K2.5** | Moonshot AI | 87.6 | 1T | 32B | 3.2% | 596 GB | 5x H200 / 3x B300 | Modified MIT |
| **gpt-oss-120b** | OpenAI | 80.1 | 117B | 5.1B | 4.4% | 72 GB | 1x A100 80GB | Apache 2.0 |
| **Nemotron Ultra 253B** | NVIDIA | 76 | 253B | 253B | 100.0% | 160 GB \* | 1x B200 | NVIDIA Open Model License |
| **Llama 4 Scout** | Meta | 73.7 | 109B | 17B | 15.6% | 71 GB | 1x A100 80GB | Llama 4 Community |
| **DeepSeek-R1** | DeepSeek | 71.5 | 671B | 37B | 5.5% | 401 GB | 3x H200 / 2x B300 | MIT |
| **Llama 4 Maverick** | Meta | 69.8 | 400B | 17B | 4.2% | 242 GB | 1x B300 | Llama 4 Community |
| **Mistral Large 3** | Mistral AI | 43.9 | 673B | 41B | 6.1% | 403 GB \* | 4x H200 / 2x B300 | Apache 2.0 |

### Ranked by SWE-bench V

| Model | Vendor | SWE-bench V | Total params | Active params | Active % | RAM / call | Fits on | License |
|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-27B** | Alibaba (Qwen) | 72.4 | 27B | 27B | 100.0% | 19 GB \* | 1x RTX 4090 | Apache 2.0 |
| **Qwen3.5-122B-A10B** | Alibaba (Qwen) | 72 | 122B | 10B | 8.2% | 75 GB | 1x A100 80GB | Apache 2.0 |
| **Qwen3.5-35B-A3B** | Alibaba (Qwen) | 69.2 | 35B | 3B | 8.6% | 23 GB | 1x RTX 5090 | Apache 2.0 |
| **Qwen3-Coder-Next 80B-A3B** | Alibaba (Qwen) | 68 | 80B | 3B | 3.8% | 50 GB | 1x A100 80GB | Apache 2.0 |
| **Qwen3-Coder-30B-A3B** | Alibaba (Qwen) | 51.6 | 30B | 3.3B | 10.8% | 21 GB | 1x RTX 4090 | Apache 2.0 |

### Ranked by AA Index

| Model | Vendor | AA Index | Total params | Active params | Active % | RAM / call | Fits on | License |
|---|---|---|---|---|---|---|---|---|
| **GLM-5.3-Flash** | Z.ai (Zhipu AI) | 57 | 320B | 18B | 5.6% | 192 GB \* | 1x B300 | MIT |
| **Nemotron 3 Ultra 550B-A55B** | NVIDIA | 38 | 550B | 55B | 10.0% | 329 GB \* | 3x H200 / 2x B300 | OpenMDW-1.1 |
| **MiniMax M2.7 230B** | MiniMax | 38 | 230B | 10B | 4.3% | 139 GB \* | 1x B200 | MIT |
| **Ling 3.0 Flash** | InclusionAI / Ant Group | 38 | 124B | 5.1B | 4.1% | 76 GB \* | 1x A100 80GB | MIT |
| **MiMo-V2.5 310B** | Xiaomi | 37 | 310B | 15B | 4.8% | 186 GB | 1x B300 | Permissive (MiMo) |
| **Qwen3.5-9B** | Alibaba (Qwen) | 32 | 9B | 9B | 100.0% | 7.3 GB \* | 1x RTX 4090 | Apache 2.0 |
| **Qwen3.5-4B** | Alibaba (Qwen) | 27 | 4B | 4B | 100.0% | 4.1 GB \* | 1x RTX 4090 | Apache 2.0 |
| **Qwen3.5-2B** | Alibaba (Qwen) | 16 | 2B | 2B | 100.0% | 2.6 GB \* | 1x RTX 4090 | Apache 2.0 |
| **Qwen3.5-0.8B** | Alibaba (Qwen) | 9 | 0.8B | 0.8B | 100.0% | 1.8 GB \* | 1x RTX 4090 | Apache 2.0 |

## Every model, by size, at three serving precisions

RAM for one call at 32,768 tokens. `Native` uses the measured official
checkpoint where one is published, otherwise the model's shipped precision.

| Model | Total | Active | BF16 | FP8 | 4-bit | Native | Context | Attention |
|---|---|---|---|---|---|---|---|---|
| **Kimi K3** | 2.80T | 104B | 5.60 TB | 2.86 TB | 1.66 TB | 1.57 TB (mxfp4) | 1M | 3:1 Kimi Delta Attention + gated MLA |
| **Qwen3.8-2.4T-A95B (open Qwen3.8 Max)** | 2.40T | 95B | 4.80 TB | 2.45 TB | 1.42 TB | 4.80 TB (bf16) | 256K | Gated DeltaNet + Gated Attention (Qwen3.5 hybrid design) |
| **DeepSeek V4-Pro** | 1.60T | 49B | 3.20 TB | 1.64 TB | 948 GB | 948 GB (fp4_fp8_mixed) | 1M | Compressed Sparse Attention + Heavily Compressed Attention |
| **MiMo-V2.5-Pro** | 1.02T | 42B | 2.04 TB | 1.04 TB | 607 GB | 1.04 TB (fp8) | 1M | GQA with 6:1 sliding-window/global attention |
| **Kimi K2.7 Code** | 1T | 32B | 2.00 TB | 1.02 TB | 596 GB | 1.02 TB (fp8) | 256K | MLA |
| **Kimi K2.6** | 1T | 32B | 2.00 TB | 1.02 TB | 596 GB | 1.02 TB (fp8) | 256K | MLA |
| **Kimi K2.5** | 1T | 32B | 2.00 TB | 1.02 TB | 596 GB | 1.02 TB (fp8) | 256K | MLA |
| **Ling 2.6 1T** | 1T | 63B | 2.00 TB | 1.02 TB | 594 GB | 2.00 TB (bf16) | 128K | Lightning Attention + MLA |
| **GLM-5.3** | 753B | 40B | 1.51 TB | 773 GB | 450 GB | 773 GB (fp8) | 1M | MLA + DeepSeek Sparse Attention with IndexShare |
| **GLM-5.2** | 753B | 40B | 1.51 TB | 773 GB | 450 GB | 773 GB (fp8) | 1M | MLA + DeepSeek Sparse Attention with IndexShare |
| **Mistral Large 3** | 673B | 41B | 1.35 TB | 692 GB | 403 GB | 692 GB (fp8) | 256K | MLA |
| **DeepSeek V3.2** | 671B | 37B | 1.35 TB | 689 GB | 401 GB | 689 GB (fp8) | 160K | MLA + DeepSeek Sparse Attention |
| **DeepSeek-R1** | 671B | 37B | 1.35 TB | 689 GB | 401 GB | 689 GB (fp8) | 128K | MLA |
| **Nemotron 3 Ultra 550B-A55B** | 550B | 55B | 1.10 TB | 565 GB | 329 GB | 329 GB (nvfp4) | 1M | Mamba-2 + GQA hybrid LatentMoE with MTP |
| **MiniMax M3** | 427B | 23B | 858 GB | 440 GB | 256 GB | 858 GB (bf16) | 512K | GQA with MiniMax Sparse Attention |
| **Llama 4 Maverick** | 400B | 17B | 806 GB | 414 GB | 242 GB | 806 GB (bf16) | 1M | GQA |
| **Arcee Trinity Large 400B** | 400B | 13B | 803 GB | 411 GB | 239 GB | 803 GB (bf16) | 128K | 3:1 sliding-window/global gated GQA |
| **Qwen3.5-397B-A17B** | 397B | 17B | 797 GB | 408 GB | 238 GB | 797 GB (bf16) | 256K | 3:1 Gated DeltaNet + Gated Attention |
| **GLM-4.7 355B** | 355B | 32B | 719 GB | 371 GB | 219 GB | 719 GB (bf16) | 200K | GQA |
| **GLM-5.3-Flash** | 320B | 18B | 643 GB | 329 GB | 192 GB | 329 GB (fp8) | 1M | 3:1 Kimi Delta Attention + MLA/DSA |
| **Motif 3 Beta** | 314B | 13B | 631 GB | 323 GB | 188 GB | 631 GB (bf16) | 128K | GDLA with 3:1 sliding-window/full attention |
| **MiMo-V2.5 310B** | 310B | 15B | 623 GB | 320 GB | 186 GB | 320 GB (fp8) | 1M | 5:1 sliding-window/global attention |
| **Hy3 295B-A21B** | 295B | 21B | 598 GB | 309 GB | 182 GB | 598 GB (bf16) | 256K | GQA |
| **DeepSeek V4-Flash** | 284B | 13B | 571 GB | 293 GB | 171 GB | 170 GB (fp4_fp8_mixed) | 1M | Compressed Sparse Attention + Heavily Compressed Attention |
| **Inkling-Small** | 276B | 12B | 555 GB | 284 GB | 166 GB | 555 GB (bf16) | 1M | 5:1 sliding-window/global GQA |
| **Nemotron Ultra 253B** | 253B | 253B | 516 GB | 268 GB | 160 GB | 516 GB (bf16) | 128K | GQA (dense transformer) |
| **Solar Open 2** | 250B | 15B | 503 GB | 258 GB | 150 GB | 503 GB (bf16) | 128K | 3:1 Kimi Delta Attention + gated GQA |
| **Qwen3 235B-A22B** | 235B | 22B | 476 GB | 245 GB | 145 GB | 476 GB (bf16) | 256K | GQA |
| **MiniMax M2.7 230B** | 230B | 10B | 463 GB | 238 GB | 139 GB | 463 GB (bf16) | 200K | GQA |
| **Command A+ 218B-A25B** | 218B | 25B | 439 GB | 225 GB | 132 GB | 439 GB (bf16) | 256K | 16:1 GQA with 3:1 sliding-window/global attention |
| **Step 3.5 Flash 196B** | 196B | 11B | 395 GB | 203 GB | 119 GB | 395 GB (bf16) | 128K | 3:1 sliding-window GQA |
| **Ling 3.0 Flash** | 124B | 5.1B | 251 GB | 129 GB | 76 GB | 251 GB (bf16) | 128K | 5:1 Kimi Delta Attention + gated MLA |
| **Qwen3.5-122B-A10B** | 122B | 10B | 247 GB | 127 GB | 75 GB | 247 GB (bf16) | 256K | 3:1 Gated DeltaNet + Gated Attention |
| **Nemotron 3 Super 120B-A12B** | 120B | 12B | 243 GB | 125 GB | 74 GB | 243 GB (bf16) | 128K | Mamba-2 + GQA |
| **Mistral Small 4** | 119B | 6.63B | 242 GB | 125 GB | 74 GB | 242 GB (bf16) | 128K | MLA |
| **Laguna S 2.1** | 118B | 8B | 239 GB | 123 GB | 73 GB | 239 GB (bf16) | 128K | 3:1 sliding-window/global gated GQA |
| **gpt-oss-120b** | 117B | 5.1B | 237 GB | 122 GB | 72 GB | 68 GB (mxfp4) | 128K | Alternating sliding-window/global GQA |
| **Llama 4 Scout** | 109B | 17B | 224 GB | 117 GB | 71 GB | 224 GB (bf16) | 192K | GQA with iRoPE |
| **GLM-4.5-Air** | 106B | 12B | 218 GB | 114 GB | 68 GB | 218 GB (bf16) | 128K | GQA |
| **INTELLECT-3** | 106B | 12B | 216 GB | 112 GB | 67 GB | 216 GB (bf16) | 128K | GQA |
| **Qwen3-Coder-Next 80B-A3B** | 80B | 3B | 162 GB | 84 GB | 50 GB | 162 GB (bf16) | 256K | 3:1 Gated DeltaNet + Gated Attention |
| **Qwen3 Next 80B-A3B** | 80B | 3B | 162 GB | 84 GB | 50 GB | 162 GB (bf16) | 256K | 3:1 Gated DeltaNet + Gated Attention |
| **LongCat-Flash-Lite 68.5B-A3B** | 68B | 3B | 140 GB | 73 GB | 44 GB | 140 GB (bf16) | 128K | MLA |
| **Qwen3.5-35B-A3B** | 35B | 3B | 72 GB | 38 GB | 23 GB | 72 GB (bf16) | 256K | 3:1 Gated DeltaNet + Gated Attention |
| **Qwen3.6-35B-A3B** | 35B | 3B | 72 GB | 38 GB | 23 GB | 72 GB (bf16) | 256K | 3:1 Gated DeltaNet + Gated Attention |
| **Laguna XS 2.1** | 33B | 3B | 69 GB | 36 GB | 22 GB | 69 GB (bf16) | 128K | 3:1 sliding-window/global gated GQA |
| **Qwen3-Coder-30B-A3B** | 30B | 3.3B | 64 GB | 35 GB | 21 GB | 64 GB (bf16) | 256K | GQA |
| **Sarvam 30B** | 30B | 2.4B | 63 GB | 34 GB | 21 GB | 63 GB (bf16) | 32K | GQA |
| **Nemotron 3.5 Lightning 30B-A3B** | 30B | 3B | 63 GB | 33 GB | 20 GB | 63 GB (bf16) | 128K | Mamba-2 + GQA |
| **Qwen3.8-27B** | 28B | 28B | 60 GB | 33 GB | 21 GB | 60 GB (bf16) | 256K | 3:1 Gated DeltaNet + Gated Attention |
| **Qwen3.5-27B** | 27B | 27B | 57 GB | 30 GB | 19 GB | 57 GB (bf16) | 256K | Dense with Gated DeltaNet interleave |
| **Gemma 4 26B-A4B** | 25B | 3.8B | 53 GB | 28 GB | 18 GB | 53 GB (bf16) | 128K | 5:1 sliding-window/global GQA |
| **gpt-oss-20b** | 21B | 3.6B | 45 GB | 24 GB | 15 GB | 14 GB (mxfp4) | 128K | Alternating sliding-window/global GQA |
| **Mellum2 Thinking 12B-A2.5B** | 12B | 2.5B | 27 GB | 15 GB | 9.8 GB | 27 GB (bf16) | 128K | 3:1 sliding-window/full GQA |
| **Qwen3.5-9B** | 9B | 9B | 20 GB | 11 GB | 7.3 GB | 20 GB (bf16) | 256K | Dense, 32 layers |
| **ZAYA1-8B** | 8.4B | 0.76B | 19 GB | 11 GB | 7.6 GB | 19 GB (bf16) | 32K | CCA with 4:1 GQA |
| **LFM2.5 8B-A1B** | 8.3B | 1.5B | 19 GB | 11 GB | 7.6 GB | 19 GB (bf16) | 32K | LIV convolution blocks + GQA + MoE |
| **Qwen3.5-4B** | 4B | 4B | 9.7 GB | 5.8 GB | 4.1 GB | 9.7 GB (bf16) | 256K | Dense, 32 layers |
| **Qwen3.5-2B** | 2B | 2B | 5.5 GB | 3.5 GB | 2.6 GB | 5.5 GB (bf16) | 256K | Dense, 24 layers |
| **Qwen3.5-0.8B** | 0.8B | 0.8B | 2.9 GB | 2.2 GB | 1.8 GB | 2.9 GB (bf16) | 256K | Dense, 24 layers |

## The long-context bill: KV cache per token

Weights are a fixed cost; the KV cache is what turns a 32K call into a 1M call.
Architecture matters more than size here. Hy3 is a 295B model that pays 160
KiB/token because it runs plain GQA over 80 layers. Qwen3.5-397B is a *larger*
model that pays 15 KiB/token because only 15 of its 60 layers keep a growing
cache. At 262K context that is a 43 GB cache versus a 4 GB one.

| Model | Attention | KV / token (FP8) | KV @ 128K | KV @ full context | Fixed state | Source |
|---|---|---|---|---|---|---|
| **GLM-4.7 355B** | GQA | 184.0 KiB | 25 GB | 39 GB @ 200K | - | documented |
| **Hy3 295B-A21B** | GQA | 160.0 KiB | 21 GB | 43 GB @ 256K | - | config |
| **Llama 4 Maverick** | GQA | 96.0 KiB | 13 GB | 103 GB @ 1M | - | documented |
| **Llama 4 Scout** | GQA with iRoPE | 96.0 KiB | 13 GB | 19 GB @ 192K | - | documented |
| **Qwen3 235B-A22B** | GQA | 94.0 KiB | 13 GB | 25 GB @ 256K | - | config |
| **GLM-4.5-Air** | GQA | 92.0 KiB | 12 GB | 12 GB @ 128K | - | documented |
| **Nemotron Ultra 253B** | GQA (dense transformer) | 81.0 KiB | 11 GB | 11 GB @ 128K | - | estimated |
| **Mistral Large 3** | MLA | 49.5 KiB | 6.6 GB | 13 GB @ 256K | - | estimated |
| **Qwen3-Coder-30B-A3B** | GQA | 48.0 KiB | 6.4 GB | 13 GB @ 256K | - | config |
| **GLM-5.3** | MLA + DeepSeek Sparse Attention with IndexShare | 46.6 KiB | 6.3 GB | 50 GB @ 1M | - | config |
| **GLM-5.2** | MLA + DeepSeek Sparse Attention with IndexShare | 46.6 KiB | 6.3 GB | 50 GB @ 1M | - | config |
| **INTELLECT-3** | GQA | 46.0 KiB | 6.2 GB | 6.2 GB @ 128K | - | estimated |
| **Kimi K2.7 Code** | MLA | 34.3 KiB | 4.6 GB | 9.2 GB @ 256K | - | config |
| **Kimi K2.6** | MLA | 34.3 KiB | 4.6 GB | 9.2 GB @ 256K | - | config |
| **Kimi K2.5** | MLA | 34.3 KiB | 4.6 GB | 9.2 GB @ 256K | - | documented |
| **DeepSeek V3.2** | MLA + DeepSeek Sparse Attention | 34.3 KiB | 4.6 GB | 5.8 GB @ 160K | - | config |
| **DeepSeek-R1** | MLA | 34.3 KiB | 4.6 GB | 4.6 GB @ 128K | - | config |
| **Qwen3.8-27B** | 3:1 Gated DeltaNet + Gated Attention | 32.0 KiB | 4.3 GB | 8.6 GB @ 256K | 0.2 GB | config |
| **Mistral Small 4** | MLA | 31.5 KiB | 4.2 GB | 4.2 GB @ 128K | - | estimated |
| **MiniMax M2.7 230B** | GQA | 31.0 KiB | 4.2 GB | 6.5 GB @ 200K | - | estimated |
| **MiMo-V2.5-Pro** | GQA with 6:1 sliding-window/global attention | 25.0 KiB | 3.4 GB | 27 GB @ 1M | 79 MB | config |
| **DeepSeek V4-Pro** | Compressed Sparse Attention + Heavily Compressed Attention | 24.2 KiB | 3.2 GB | 26 GB @ 1M | - | estimated |
| **DeepSeek V4-Flash** | Compressed Sparse Attention + Heavily Compressed Attention | 24.2 KiB | 3.2 GB | 26 GB @ 1M | - | config |
| **Sarvam 30B** | GQA | 24.0 KiB | n/a | 0.8 GB @ 32K | - | estimated |

(Truncated to the 24 most cache-hungry models; run `python llmram.py --model <id>` for any single model.)

## What fits on what

For each hardware budget, the largest and the highest-scoring model that fits one
call at 32K context in 4-bit. "Fits" means the total lands inside the usable
fraction of nameplate memory.

| Budget | Usable | Largest that fits | Best-scoring that fits |
|---|---|---|---|
| Laptop / 8 GB GPU | 8 GB | Qwen3.5-9B (9B) | Qwen3.5-9B (32) |
| RTX 4090 / 5090 (24-32 GB) | 23 GB | Qwen3.6-35B-A3B (35B) | Qwen3.5-27B (72.4) |
| RTX 6000 Pro (96 GB) | 91 GB | Ling 3.0 Flash (124B) | gpt-oss-120b (80.1) |
| 1x H100 / A100 (80 GB) | 76 GB | Ling 3.0 Flash (124B) | gpt-oss-120b (80.1) |
| 1x H200 (141 GB) | 134 GB | Command A+ 218B-A25B (218B) | gpt-oss-120b (80.1) |
| Mac Studio M3 Ultra (512 GB unified) | 486 GB | GLM-5.3 (753B) | MiniMax M3 (93) |
| 8x H100 node (640 GB) | 608 GB | MiMo-V2.5-Pro (1.02T) | MiniMax M3 (93) |
| 8x H200 node (1,128 GB) | 1072 GB | DeepSeek V4-Pro (1.60T) | MiniMax M3 (93) |
| 8x B300 node (2,304 GB) | 2189 GB | Kimi K3 (2.80T) | Kimi K3 (93.5) |

## Not included, and why

"Open source LLM" leaderboards routinely list models whose weights you cannot
download. These are excluded here on purpose:

- **Qwen3.5-Max / 3.6-Max / 3.7-Max** — API-only. Every Qwen Max generation before 3.8 stayed closed; Qwen3.8 broke that pattern, so the 2.4T-A95B checkpoint IS included above. Trackers that still list Qwen3.8 Max as proprietary are stale - the weights went up on ModelScope on 2026-08-12.
- **Inkling (full)** — 975B/41B. Thinking Machines' own comparison table places Inkling in the closed-weights group; only Inkling-Small (276B/12B) is Apache 2.0.
- **Gemini 3.1 Pro / Claude Opus 4.6 / GPT-5.6** — Closed weights. Included in some tables above only as reference points.

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
Generated 2026-08-28 from data as of 2026-08-27.

