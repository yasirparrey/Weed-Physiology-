"""Calibration tests: the estimator must reproduce independently published figures.

Each test below pins a number that a vendor or a third party measured, so a
change to the estimator that breaks real-world agreement fails loudly.
"""

import pytest

import llmram
from llmram import estimate, load_models, weights_gb

MODELS = {m.id: m for m in load_models()[0]}
KIB = 1024
GIB = 1024**3


def rel(actual, expected):
    return abs(actual - expected) / expected


# ------------------------------------------------------------------- KV geometry


def test_kimi_k3_kv_per_token_matches_published():
    """Published: 13.5 KiB/token at FP8, 27.0 KiB/token at BF16."""
    k3 = MODELS["kimi-k3"]
    fp8 = estimate(k3, context=1024, kv_precision="fp8")
    bf16 = estimate(k3, context=1024, kv_precision="bf16")
    assert fp8.kv_bytes_per_token == pytest.approx(13.5 * KIB, rel=0.01)
    assert bf16.kv_bytes_per_token == pytest.approx(27.0 * KIB, rel=0.01)


def test_kimi_k3_kv_at_full_context_matches_published():
    """Published: a full 1M-token sequence costs 13.5 GiB of KV cache at FP8."""
    b = estimate(MODELS["kimi-k3"], context=1_048_576, kv_precision="fp8")
    assert b.kv_gb * 1e9 / GIB == pytest.approx(13.5, rel=0.02)


def test_kimi_k3_recurrent_state_matches_published():
    """Published: 213-427 MiB of KDA + conv state per concurrent sequence."""
    b = estimate(MODELS["kimi-k3"], context=4096, recurrent_precision="bf16")
    mib = b.recurrent_gb * 1e9 / (1024**2)
    assert 200 <= mib <= 440, mib


def test_glm52_kv_at_262k_matches_published():
    """Published: a 262K-token session holds 11.8 GiB; the full 1M holds 47.3 GiB."""
    at_262k = estimate(MODELS["glm-52"], context=262_144, kv_precision="fp8")
    at_1m = estimate(MODELS["glm-52"], context=1_048_576, kv_precision="fp8")
    assert at_262k.kv_gb * 1e9 / GIB == pytest.approx(11.8, rel=0.05)
    assert at_1m.kv_gb * 1e9 / GIB == pytest.approx(47.3, rel=0.05)


def test_sliding_window_layers_stop_growing():
    """A 6:1 SWA model must not scale its windowed layers with context."""
    m = MODELS["mimo-v25-pro"]
    short = estimate(m, context=8192)
    long = estimate(m, context=1_048_576)
    # Only the 10 global layers grow, so KV must stay far below a full-GQA model.
    assert long.kv_gb < 30
    assert short.recurrent_gb == pytest.approx(long.recurrent_gb)


def test_hybrid_linear_attention_is_cheaper_than_dense_gqa():
    """Qwen3.5-397B caches 15 of 60 layers; Hy3 caches all 80. Per token, Hy3 loses."""
    qwen = estimate(MODELS["qwen35-397b"], context=131_072)
    hy3 = estimate(MODELS["hy3"], context=131_072)
    assert qwen.kv_bytes_per_token < hy3.kv_bytes_per_token / 5


# ---------------------------------------------------------------- weight sizing


def test_kimi_k3_mxfp4_checkpoint_matches_measurement():
    """Measured: 1,560.9 GB across 96 safetensors shards."""
    gb, measured = weights_gb(MODELS["kimi-k3"], "native")
    assert measured is True
    assert gb == pytest.approx(1560.9, rel=0.001)


def test_computed_mxfp4_size_tracks_the_measured_checkpoint():
    """The 4.25-bit + uplift model should land within a few percent unaided."""
    computed, _ = weights_gb(MODELS["kimi-k3"], "mxfp4")
    assert rel(computed, 1560.9) < 0.05


def test_deepseek_v4_flash_repo_size_matches_measurement():
    """Published repository size: 166.9 GB in FP4+FP8 mixed precision."""
    computed, _ = weights_gb(MODELS["deepseek-v4-flash"], "fp4_fp8_mixed")
    assert rel(computed, 166.9) < 0.05


def test_qwen35_397b_int4_matches_community_figures():
    """Community 4-bit builds are quoted at roughly 214-220 GB."""
    computed, _ = weights_gb(MODELS["qwen35-397b"], "int4")
    assert rel(computed, 220) < 0.10


def test_mistral_small_4_fp8_matches_published():
    """Published: roughly 111 GB at FP8, low enough for two A100s."""
    computed, _ = weights_gb(MODELS["mistral-small-4"], "fp8")
    assert rel(computed, 111) < 0.10


def test_gpt_oss_120b_fits_one_80gb_card():
    """The headline claim for gpt-oss-120b: native MXFP4 on a single 80 GB GPU."""
    b = estimate(MODELS["gpt-oss-120b"], context=32768, weights_precision="mxfp4")
    assert b.total_gb < 80
    assert llmram.gpus_needed(b.total_gb, "h100")[0] == 1


# --------------------------------------------------------------- sizing outcomes


def test_kimi_k3_needs_a_multi_node_class_deployment():
    """Published: an 8x H200 node (1,128 GB) is short of the weights alone, so the
    floors are 8x 288 GB accelerators or two H200 nodes. The published ~1.8 TB
    figure includes headroom for concurrency, which one call does not need."""
    b = estimate(MODELS["kimi-k3"], context=131_072, weights_precision="native")
    assert 1500 < b.total_gb < 1800
    assert b.total_gb > 1128, "must not fit a single 8x H200 node"
    assert llmram.gpus_needed(b.total_gb, "h200")[0] > 8
    assert llmram.gpus_needed(b.total_gb, "b300")[0] <= 8


def test_glm52_fp8_fits_a_single_eight_gpu_h200_node():
    """Reported: GLM-5.2 at FP8 serves on one 8x H200 node."""
    b = estimate(MODELS["glm-52"], context=131_072, weights_precision="fp8")
    assert llmram.gpus_needed(b.total_gb, "h200")[0] <= 8


def test_qwen35_35b_fits_a_24gb_consumer_card_at_4bit():
    """Reported: ~22 GB at Q4, runs on an RTX 4090."""
    b = estimate(MODELS["qwen35-35b-a3b"], context=8192, weights_precision="int4")
    assert b.weights_gb < 22
    assert llmram.gpus_needed(b.total_gb, "rtx4090")[0] == 1


def test_nemotron_3_ultra_matches_nvidia_minimum_config():
    """NVIDIA lists 4x B200 or 8x H100 as the minimum for 550B-A55B."""
    b = estimate(MODELS["nemotron-3-ultra"], context=131_072, weights_precision="nvfp4")
    assert llmram.gpus_needed(b.total_gb, "b200")[0] <= 4
    assert llmram.gpus_needed(b.total_gb, "h100")[0] <= 8


# ------------------------------------------------------------- invariants / data


def test_memory_is_driven_by_total_not_active_parameters():
    """The whole point: a 2.8T/104B model needs far more RAM than a 253B dense one."""
    k3 = estimate(MODELS["kimi-k3"], context=8192, weights_precision="int4")
    nemo = estimate(MODELS["nemotron-ultra-253b"], context=8192, weights_precision="int4")
    assert MODELS["kimi-k3"].active_params_b < MODELS["nemotron-ultra-253b"].active_params_b
    assert k3.weights_gb > 8 * nemo.weights_gb


def test_every_model_has_a_kv_model_and_a_source():
    for m in MODELS.values():
        elems, fixed, flat, source = llmram.kv_geometry(m)
        assert source in {"config", "documented", "estimated"}, m.id
        assert elems > 0 or flat > 0, f"{m.id} has no growing KV cache defined"


def test_active_params_never_exceed_total():
    for m in MODELS.values():
        assert m.active_params_b <= m.total_params_b, m.id


def test_totals_are_monotonic_in_context():
    m = MODELS["hy3"]
    sizes = [estimate(m, context=c).total_gb for c in (4096, 32768, 131072, 262144)]
    assert sizes == sorted(sizes)


def test_lower_precision_never_costs_more():
    m = MODELS["glm-52"]
    bf16 = estimate(m, context=32768, weights_precision="bf16").total_gb
    fp8 = estimate(m, context=32768, weights_precision="fp8").total_gb
    int4 = estimate(m, context=32768, weights_precision="int4").total_gb
    assert bf16 > fp8 > int4
