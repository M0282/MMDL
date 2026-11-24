#!/usr/bin/env python3
"""Comprehensive CPU-only unit test suite for mmmu_pro_pipeline.

Covers:
- StructReward MC/open parsing, endpoint consistency (including lowercase 'a' regression),
  MMK12 math equivalence ($$...$$, \\frac/\\dfrac, signed fractions, nested \\boxed,
  intervals, sqrt, pi, degrees, percentages, MC letters), process alignment & gating.
- generate_verified_rationales.is_valid_verified (thinking tag leakage, CJK rejection,
  concise multi-step acceptance, open MMK12 verification).
- prepare_training_pools vision board rendering, 10-option expansion, option cleaning.
- Qwen3-VL processor encoding (1, 2, 3 images + vision_board + open + SFT label masking
  + GRPO stop vs length EOS encoding) using real cached Qwen3-VL-4B-Instruct processor.
- GRPO token_logprobs exact index alignment (no off-by-one), compute_grpo_step_loss
  (clipping & Schulman k3 KL properties), PeftModel disable_adapter(), set_model_use_cache(),
  save_merged_model_for_sglang() compatibility files & roundtrip reload.
- eval_mmmu_pro deterministic official parser & inline multi-image message building.
- SGLang port release helper wait_port_free().
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import socket
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
import torch.nn.functional as F

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

import eval_mmmu_pro
import generate_verified_rationales as gvr
import prepare_training_pools as ptp
import struct_reward as sr
import train_sft
import train_structreward_grpo as grpo

QWEN3_VL_SNAPSHOT = Path(os.environ.get("QWEN3_VL_LOCAL_PATH", "__nonexistent__"))


# ---------------------------------------------------------------------------
# 1. StructReward: MC Parsing, Endpoint Consistency, Process Alignment & Gating
# ---------------------------------------------------------------------------

def test_parse_strict_mc_answer_variants():
    choices = list("ABCDEFGHIJ")
    for text, exp in [
        ("Step 1: Look.\nAnswer: C", ("C", True)),
        ("Step 1: Look.\n**Answer: D**", ("D", True)),
        ("Step 1: Look.\n**Answer:** E", ("E", True)),
        ("Step 1: Look.\n**Answer**: (F)", ("F", True)),
        ("Step 1: Look.\nAnswer: **G**", ("G", True)),
        ("Step 1: Look.\nAnswer: $H$.", ("H", True)),
        ("Step 1: The final answer is B. Extra commentary here.", ("B", False)),
    ]:
        assert sr.parse_strict_mc_answer(text, choices) == exp, f"Failed on {text!r}"

    # Must NOT match lowercase English 'a' in prose!
    prose = "The answer is a classic result in physics, so we need more data."
    assert sr.parse_strict_mc_answer(prose, choices) == (None, False)
    # Out-of-range choice letter rejected
    assert sr.parse_strict_mc_answer("Answer: Z", list("ABCD")) == (None, False)
    assert sr.parse_strict_mc_answer("", list("ABCD")) == (None, False)


def test_endpoint_consistency_no_lowercase_false_positive():
    choices = list("ABCD")
    # Regression test: previously (?i) caused "therefore, a ..." to match 'A' and penalize 'B'!
    resp = (
        "Step 1: Read values from the diagram.\n"
        "Step 2: Therefore, a larger force produces higher acceleration, matching option B.\n"
        "Answer: B"
    )
    assert sr.check_endpoint_consistency(resp, "B", choices) == pytest.approx(0.15)

    # Even without explicit "option B", "Therefore, a larger force..." must NOT trigger false contradiction!
    resp_neutral = (
        "Step 1: Compute 2 + 2 = 4.\n"
        "Step 2: Therefore, a straightforward check confirms 4.\n"
        "Answer: B"
    )
    assert sr.check_endpoint_consistency(resp_neutral, "B", choices) == pytest.approx(0.0)

    # Genuine contradiction MUST be penalized
    resp_bad = (
        "Step 1: Compute result.\n"
        "Step 2: This clearly corresponds to option D.\n"
        "**Answer**: C"
    )
    assert sr.check_endpoint_consistency(resp_bad, "C", choices) == pytest.approx(-0.15)


def test_struct_reward_gating_and_length_penalties():
    ref = (
        "Step 1 (Visual Observation): The circuit has V = 12 V and R = 4 ohms.\n"
        "Step 2 (Derivation): By Ohm's law, I = V / R = 12 / 4 = 3 A.\n"
        "Step 3 (Verification): Option C is 3 A.\n"
        "Answer: C"
    )
    ref_steps = sr.build_reference_steps(ref)
    assert len(ref_steps) == 3

    good = (
        "### Step 1: Visual Observation\nThe figure shows voltage 12 V across resistor 4 ohms.\n"
        "### Step 2: Derivation\nUsing Ohm's law, current I = 12 / 4 = 3 A.\n"
        "### Step 3: Verification\nThis matches option C.\n"
        "Answer: C"
    )
    wrong_with_same_steps = good.replace("option C", "option A").replace("Answer: C", "Answer: A")

    sc_good = sr.score_struct_reward(good, "C", "multiple-choice", list("ABCD"), ref_steps, "stop", 120)
    sc_wrong = sr.score_struct_reward(wrong_with_same_steps, "C", "multiple-choice", list("ABCD"), ref_steps, "stop", 120)
    sc_trunc = sr.score_struct_reward(good, "C", "multiple-choice", list("ABCD"), ref_steps, "length", 1280)

    assert sc_good["g_ans"] == 1.0
    assert sc_good["r_process_raw"] > 0.5
    assert sc_good["r_process_gated"] > 0.25
    assert sc_good["reward"] > 1.5

    # Gating guarantee: wrong answer gets ZERO gated process reward even with high r_process_raw!
    assert sc_wrong["g_ans"] == 0.0
    assert sc_wrong["r_process_raw"] > 0.4
    assert sc_wrong["r_process_gated"] == 0.0
    assert sc_wrong["reward"] < sc_good["reward"] - 1.0

    # Truncation penalized by 0.35 and loses exact_format bonus
    assert sc_trunc["truncated"] is True
    assert sc_trunc["r_len_pen"] == pytest.approx(0.35)
    assert sc_trunc["reward"] < sc_good["reward"] - 0.4


# ---------------------------------------------------------------------------
# 2. MMK12 Open-Ended Math Parsing & Equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pred,gold,expected",
    [
        ("$$\\frac{1}{2}$$", "0.5", True),
        ("$-\\frac{3}{4}$", "-0.75", True),
        ("\\dfrac{-3}{4}", "-3/4", True),
        ("\\boxed{\\frac{2}{3}}", "$$\\tfrac{2}{3}$$", True),
        ("(-2, 5)", "$$\\left(-2, 5\\right)$$", True),
        ("(-2,5)", "(-2, 6)", False),
        ("[0, +\\infty)", "$$[0, \\infty)$$", True),
        ("2\\sqrt{3}", "3.4641", True),
        ("\\frac{\\sqrt{2}}{2}", "0.707106", True),
        ("45^\\circ", "45°", True),
        ("45^{\\circ}", "45", True),
        ("50\\%", "50%", True),
        ("50%", "0.5", True),
        ("50%", "50", False),
        ("1,200", "1200", True),
        ("x = 5", "5", True),
        ("(A)", "A", True),
        ("\\text{B}", "B", True),
        ("\\(3\\pi\\)", "9.42478", True),
        ("5 \\text{ cm}", "5 cm", True),
        ("", "", False),
        ("$$", "$$", False),
        ("(-2,5)", "[ -2, 5 ]", False),
    ],
)
def test_mmk12_open_match(pred, gold, expected):
    assert sr.check_open_match(pred, gold) is expected, f"pred={pred!r} gold={gold!r}"


def test_parse_strict_open_answer_nested_boxed_and_markdown():
    r1 = "Step 1: Compute.\nStep 2: Simplify.\n**Answer:** $$\\boxed{\\frac{3}{4}}$$"
    val, exact = sr.parse_strict_open_answer(r1)
    assert exact is True
    assert sr.check_open_match(val, "0.75")

    r2 = "Therefore the result is \\boxed{\\frac{\\sqrt{2}}{2}}."
    val2, exact2 = sr.parse_strict_open_answer(r2)
    assert exact2 is False
    assert val2 == "\\frac{\\sqrt{2}}{2}"
    assert sr.check_open_match(val2, "0.707107")


# ---------------------------------------------------------------------------
# 3. generate_verified_rationales.is_valid_verified
# ---------------------------------------------------------------------------

def test_teacher_is_valid_verified():
    open_row = {"question_type": "open", "display_gold": "$$-\\frac{3}{4}$$", "display_options": []}
    good_open = (
        "Step 1 (Visual Observation): From the coordinate plane, the line passes through (0, 3) and (4, 0).\n"
        "Step 2 (Derivation): The slope is computed as m = (0 - 3) / (4 - 0) = -3/4.\n"
        "Step 3 (Verification): Checking the y-intercept at x = 0 gives y = 3 and x-intercept gives y = 0.\n"
        "Answer: $-\\frac{3}{4}$"
    )
    assert gvr.is_valid_verified(open_row, good_open, "stop") is True
    # Leading empty <think></think> stripped cleanly
    assert gvr.is_valid_verified(open_row, "<think>\n\n</think>\n\n" + good_open, "stop") is True
    # Non-empty thinking leak rejected
    assert gvr.is_valid_verified(open_row, "<think>hmm</think>\n" + good_open, "stop") is False
    # Truncated rejected
    assert gvr.is_valid_verified(open_row, good_open, "length") is False
    # CJK drift rejected
    assert gvr.is_valid_verified(open_row, good_open + "\n这是中文", "stop") is False
    # Shallow one-liner rejected
    assert gvr.is_valid_verified(open_row, "Slope is -3/4.\nAnswer: -3/4", "stop") is False


# ---------------------------------------------------------------------------
# 4. Vision Board Rendering & 10-Option Augmentation
# ---------------------------------------------------------------------------

def test_vision_board_and_10opt(tmp_path, monkeypatch):
    monkeypatch.setattr(ptp, "BOARD_DIR", tmp_path / "boards")
    img1 = tmp_path / "fig1.png"
    img2 = tmp_path / "fig2.png"
    Image.new("RGB", (400, 300), (200, 100, 50)).save(img1)
    Image.new("RGB", (300, 200), (50, 150, 200)).save(img2)

    board_path = ptp.render_vision_board(
        question="In <image 1> and <image 2>, what is the total voltage?",
        options=["1 V", "2 V", "3 V", "4 V", "5 V", "6 V", "7 V", "8 V", "9 V", "10 V"],
        img_paths=[str(img1), str(img2)],
        item_id="test_board_1",
    )
    assert board_path is not None
    assert Path(board_path).is_file()
    with Image.open(board_path) as bim:
        assert bim.width == 960
        assert 400 <= bim.height <= 2200

    row = {"options": ["(A) Alpha", "(B) Beta", "(C) Gamma", "(D) Delta"], "gold_letter": "B"}
    pool = [f"Distractor {i}" for i in range(20)]
    res = ptp.make_10opt_mc(row, pool, random.Random(7))
    assert res is not None
    new_opts, new_gold = res
    assert len(new_opts) == 10
    assert new_opts[ord(new_gold) - 65] == "Beta"


# ---------------------------------------------------------------------------
# 5. GRPO Logprob Alignment Math & Loss Properties (Exact Mock Tensor Verification)
# ---------------------------------------------------------------------------

def test_token_logprobs_exact_alignment():
    """Verify no off-by-one bug in logits[:, prompt_len - 1 : -1, :] vs target = input_ids[:, prompt_len:]."""
    vocab = 17
    prompt_len = 5
    comp_len = 4
    seq_len = prompt_len + comp_len

    # Construct deterministic logits where position t strongly favors token (t + 1) % vocab
    logits = torch.full((1, seq_len, vocab), -10.0)
    input_ids = torch.arange(seq_len).unsqueeze(0) % vocab  # [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # Position t predicts input_ids[t + 1]; give each exact target a distinct logit
    for t in range(seq_len - 1):
        next_tok = int(input_ids[0, t + 1].item())
        logits[0, t, next_tok] = float(t + 1)

    class MockLM:
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=logits)

    got = grpo.token_logprobs(MockLM(), {"input_ids": input_ids}, prompt_len)
    assert got.shape == (1, comp_len)

    # Manually compute expected logprobs for each completion token k = 0..comp_len-1
    expected = []
    for pos in range(prompt_len, seq_len):
        pred_logits = logits[0, pos - 1]
        logp = F.log_softmax(pred_logits, dim=-1)
        expected.append(logp[input_ids[0, pos]].item())
    expected_t = torch.tensor([expected], dtype=torch.float32)
    assert torch.allclose(got, expected_t, atol=1e-6)

    # Boundary validation
    with pytest.raises(ValueError):
        grpo.token_logprobs(MockLM(), {"input_ids": input_ids}, prompt_len=0)
    with pytest.raises(ValueError):
        grpo.token_logprobs(MockLM(), {"input_ids": input_ids}, prompt_len=seq_len)


def test_compute_grpo_step_loss_kl_and_asymmetric_clip():
    old_lp = torch.tensor([[-1.0, -2.0, -0.5, -1.5]])
    ref_lp = old_lp.clone()
    new_lp = old_lp.clone().requires_grad_(True)

    # When new == old == ref, ratio=1, kl=0, pg=-adv
    loss, pg, kl, clip_f = grpo.compute_grpo_step_loss(new_lp, old_lp, ref_lp, adv=1.5, eps_low=0.20, eps_high=0.28, kl_beta=0.015)
    assert kl.item() == pytest.approx(0.0, abs=1e-6)
    assert pg.item() == pytest.approx(-1.5, abs=1e-6)
    assert clip_f.item() == pytest.approx(0.0)
    loss.backward()
    # Positive advantage should push new_lp UP (negative grad w.r.t. new_lp)
    assert (new_lp.grad < 0).all()

    # Non-negative Schulman k3 KL for both positive and negative shifts
    for delta in (-2.0, -0.3, 0.3, 2.0):
        _, _, kl_d, _ = grpo.compute_grpo_step_loss(old_lp + delta, old_lp, ref_lp, adv=1.0, eps_low=0.20, eps_high=0.28, kl_beta=0.015)
        assert kl_d.item() > 0.0

    # Asymmetric clipping: ratio=1.25 is NOT clipped (<= 1.28), ratio=0.75 IS clipped (< 0.80)
    _, _, _, cf_hi = grpo.compute_grpo_step_loss(old_lp + math.log(1.25), old_lp, ref_lp, adv=1.0, eps_low=0.20, eps_high=0.28, kl_beta=0.0)
    _, _, _, cf_lo = grpo.compute_grpo_step_loss(old_lp + math.log(0.75), old_lp, ref_lp, adv=-1.0, eps_low=0.20, eps_high=0.28, kl_beta=0.0)
    assert cf_hi.item() == pytest.approx(0.0)
    assert cf_lo.item() == pytest.approx(1.0)


def test_summarize_rollouts_by_view():
    groups = [
        {"row": {"view_type": "vision_board"}, "rollouts": [{"reward": 1.2, "g_ans": 1.0, "r_process_raw": 0.6, "exact_format": True, "truncated": False, "approx_tokens": 100}]},
        {"row": {"view_type": "standard10_aug"}, "rollouts": [{"reward": 0.1, "g_ans": 0.0, "r_process_raw": 0.2, "exact_format": False, "truncated": True, "approx_tokens": 400}]},
    ]
    res = grpo.summarize_rollouts_by_view(groups)
    assert set(res.keys()) == {"vision_board", "standard10_aug"}
    assert res["vision_board"]["acc_pct"] == 100.0
    assert res["standard10_aug"]["acc_pct"] == 0.0
    assert res["standard10_aug"]["trunc_pct"] == 100.0


# ---------------------------------------------------------------------------
# 6. Qwen3-VL Processor, Tiny PEFT disable_adapter, & SGLang Merged Save Roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not QWEN3_VL_SNAPSHOT.is_dir(), reason="Local Qwen3-VL-4B-Instruct cache not found")
def test_qwen3vl_sft_and_grpo_encoding_multi_image(tmp_path):
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(str(QWEN3_VL_SNAPSHOT), min_pixels=4096, max_pixels=1048576)
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    p_tiny = tmp_path / "tiny.png"
    Image.new("RGB", (256, 256), (255, 0, 0)).save(p1)
    Image.new("RGB", (320, 192), (0, 255, 0)).save(p2)
    Image.new("RGB", (12, 12), (0, 0, 255)).save(p_tiny)

    # Verify resize_pil upscales tiny image safely to >= 32 and >= 4096 px
    im_up = train_sft.resize_pil(str(p_tiny), 786432)
    assert min(im_up.size) >= 32 and im_up.size[0] * im_up.size[1] >= 4096

    # Test 2-image standard10_aug SFT encoding
    row_multi = {
        "view_type": "standard10_aug",
        "question_type": "multiple-choice",
        "question": "Compare <image 1> and <image 2>.",
        "display_options": ["Red", "Green", "Blue", "Yellow", "Black", "White"],
        "display_images": [str(p1), str(p2)],
        "display_gold": "B",
        "verified_rationale": "Step 1: Compare images.\nStep 2: Second is green.\nAnswer: B",
    }
    p_msgs, f_msgs, imgs = train_sft.build_training_sample(row_multi, "calibrated_cot", 786432)
    assert len(imgs) == 2
    batch = train_sft.encode_sft(proc, p_msgs, f_msgs, imgs, max_len=3072)
    assert batch is not None
    assert batch["image_grid_thw"].shape[0] == 2
    labels = batch["labels"][0]
    # Prompt + vision tokens masked (-100); assistant tokens unmasked
    assert (labels[:50] == -100).all()
    assert (labels != -100).sum().item() > 5
    # Verify NO image_pad token is ever unmasked in labels
    img_pad_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    assert not ((batch["input_ids"][0] == img_pad_id) & (labels != -100)).any()

    # Test vision_board encoding with GRPO stop vs length EOS behavior
    row_vb = {
        "view_type": "vision_board",
        "question_type": "multiple-choice",
        "question": "ignored",
        "display_options": ["A", "B", "C", "D"],
        "display_images": [str(p1), str(p2)],  # extra image defensively sliced to 1
        "display_gold": "A",
    }
    _, hf_msgs, vb_imgs = grpo.build_prompt_for_sglang_and_hf(row_vb, "official_cot", 786432)
    assert len(vb_imgs) == 1
    enc_stop = grpo.encode_grpo_sample(proc, hf_msgs, vb_imgs, "Step 1: Done.\nAnswer: A", "stop", 3072)
    enc_len = grpo.encode_grpo_sample(proc, hf_msgs, vb_imgs, "Step 1: Done.\nAnswer: A", "length", 3072)
    assert enc_stop is not None and enc_len is not None
    full_stop, plen_stop = enc_stop
    full_len, plen_len = enc_len
    assert plen_stop == plen_len
    eos_id = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert full_stop["input_ids"][0, -1].item() == eos_id
    assert full_len["input_ids"][0, -1].item() != eos_id
    assert full_stop["input_ids"].shape[1] == full_len["input_ids"].shape[1] + 1


@pytest.mark.skipif(not QWEN3_VL_SNAPSHOT.is_dir(), reason="Local Qwen3-VL-4B-Instruct cache not found")
def test_qwen3vl_peft_disable_adapter_and_sglang_merge_save(tmp_path):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

    cfg = AutoConfig.from_pretrained(str(QWEN3_VL_SNAPSHOT))
    cfg.text_config.num_hidden_layers = 1
    cfg.text_config.hidden_size = 64
    cfg.text_config.intermediate_size = 128
    cfg.text_config.num_attention_heads = 2
    cfg.text_config.num_key_value_heads = 2
    cfg.text_config.head_dim = 32
    cfg.text_config.vocab_size = 256
    cfg.vision_config.depth = 1
    cfg.vision_config.hidden_size = 32
    cfg.vision_config.intermediate_size = 64
    cfg.vision_config.num_heads = 2
    cfg.vision_config.out_hidden_size = 64
    cfg.vision_config.deepstack_visual_indexes = []

    base = Qwen3VLForConditionalGeneration(cfg)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    train_sft.set_model_use_cache(model, False)
    assert model.config.text_config.use_cache is False

    # Perturb LoRA B weights so adapter != reference
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.fill_(0.25)

    ids = torch.randint(0, 200, (1, 10))
    inp = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    model.eval()
    with torch.no_grad():
        lp_adp = grpo.token_logprobs(model, inp, prompt_len=4)
        with model.disable_adapter():
            lp_ref = grpo.token_logprobs(model, inp, prompt_len=4)
        lp_adp_after = grpo.token_logprobs(model, inp, prompt_len=4)

    assert lp_adp.shape == (1, 6)
    assert (lp_adp - lp_ref).abs().max().item() > 1e-3
    assert torch.allclose(lp_adp, lp_adp_after, atol=1e-6)

    # Test save_merged_model_for_sglang creates complete SGLang-compatible directory
    proc = AutoProcessor.from_pretrained(str(QWEN3_VL_SNAPSHOT), min_pixels=4096, max_pixels=1048576)
    merged = model.merge_and_unload()
    out_dir = tmp_path / "merged"
    train_sft.save_merged_model_for_sglang(merged, proc, QWEN3_VL_SNAPSHOT, out_dir)

    for required in (
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "chat_template.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        assert (out_dir / required).is_file(), f"Missing {required} in merged_dir"

    saved_cfg = json.loads((out_dir / "config.json").read_text())
    assert saved_cfg["text_config"]["use_cache"] is True
    # Roundtrip AutoProcessor reload
    proc2 = AutoProcessor.from_pretrained(str(out_dir))
    assert proc2.image_processor.size["longest_edge"] == 1048576


# ---------------------------------------------------------------------------
# 7. eval_mmmu_pro & Port Release Helpers
# ---------------------------------------------------------------------------

def test_eval_mmmu_pro_scoring_and_messages(tmp_path):
    opts = ["Alpha", "Beta", "Gamma", "Delta"]
    sc1 = eval_mmmu_pro.score_item("Step 1: Check.\n**Answer:** C", opts, "C", "q1")
    assert sc1["parsed_pred"] == "C" and sc1["strict_correct"] is True and sc1["random_fallback"] is False

    # Deterministic fallback when response is empty
    sc2a = eval_mmmu_pro.score_item("", opts, "A", "q2")
    sc2b = eval_mmmu_pro.score_item("", opts, "A", "q2")
    assert sc2a["parsed_pred"] == sc2b["parsed_pred"]
    assert sc2a["random_fallback"] is True and sc2a["strict_correct"] is False

    img = tmp_path / "x.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(img)
    row = {
        "question": "Look at <image 1> and <image 99>.",
        "options": opts,
        "image_paths": {"1": str(img)},
    }
    msgs = eval_mmmu_pro.build_messages(row, "standard10", "official_cot", 1048576)
    # Missing <image 99> does NOT crash; valid <image 1> is embedded
    img_items = [c for c in msgs[-1]["content"] if c["type"] == "image_url"]
    assert len(img_items) == 1


def test_wait_port_free():
    # Bind an ephemeral port, verify wait_port_free returns False while held and True after release
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        assert grpo.wait_port_free(port, timeout=0.6) is False
    finally:
        sock.close()
    assert grpo.wait_port_free(port, timeout=2.0) is True
