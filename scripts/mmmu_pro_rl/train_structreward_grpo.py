#!/usr/bin/env python3
"""Stage 2: Online-Batch StructReward Gated GRPO RL for Qwen3-VL-4B-Instruct (BF16).

Architecture (single-GPU BF16 online-batch GRPO):
For each online round r = 1..R:
  Phase A (Fast SGLang Rollout):
    - Serve current policy weights in SGLang (BF16, 128 concurrent requests).
    - Sample G=8 rollouts per prompt for B=128 balanced external prompts (=1,024 rollouts/round)
      spanning Standard MC, 10-opt External MC, Vision-Board External MC, and Open STEM.
    - Stop SGLang cleanly so PyTorch gets full 180GB VRAM.
  Phase B (Deterministic StructReward & Dynamic Group Filtering):
    - Score every rollout with `score_struct_reward`:
        R = G_ans * (1 + alpha * R_process) + R_format + R_consistency - R_length
    - Keep groups with non-zero reward variance (skip all-identical groups).
    - Compute Dr.GRPO/DAPO group-relative advantages (mean-centered, robust std scaling).
  Phase C (PyTorch BF16 GRPO/DAPO Policy Update):
    - Load SFT-merged reference model + current LoRA adapter.
    - Compute reference log-probs via `model.disable_adapter()` (zero extra model memory!)
      and old behavior log-probs before updating.
    - Run K=2 PPO/DAPO epochs with asymmetric clip (eps_low=0.20, eps_high=0.28)
      and per-token reverse KL penalty (beta=0.015).
    - Save LoRA adapter + merged BF16 model for next round's SGLang rollout.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gc
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

from PIL import Image
import requests
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from struct_reward import (
    check_open_match,
    parse_strict_mc_answer,
    parse_strict_open_answer,
    score_struct_reward,
)
from train_sft import (
    PROMPT_STYLES,
    TAG_RE,
    resize_pil,
    save_merged_model_for_sglang,
    set_model_use_cache,
)


def encode_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_prompt_for_sglang_and_hf(row: dict, style_name: str, max_pixels: int):
    """Return both SGLang API messages (base64) and HF processor messages + PIL images."""
    cfg = PROMPT_STYLES[style_name]
    view = row["view_type"]
    px = max(max_pixels, 1048576) if view == "vision_board" else max_pixels
    raw_imgs = row["display_images"][:1] if view == "vision_board" else row["display_images"]
    imgs = [resize_pil(p, px) for p in raw_imgs]

    sg_user = []
    hf_user = []
    if view == "vision_board":
        sg_user.append({"type": "image_url", "image_url": {"url": encode_b64(imgs[0])}})
        sg_user.append({"type": "text", "text": cfg["vision"]})
        hf_user.append({"type": "image"})
        hf_user.append({"type": "text", "text": cfg["vision"]})
    else:
        for im in imgs:
            sg_user.append({"type": "image_url", "image_url": {"url": encode_b64(im)}})
            hf_user.append({"type": "image"})
        q_clean = TAG_RE.sub("", row["question"]).strip()
        if row["question_type"] == "multiple-choice":
            opts_txt = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(row["display_options"]))
            txt = f"{q_clean}\n{opts_txt}\n{cfg['standard']}"
        else:
            txt = f"{q_clean}\n{cfg['open']}"
        sg_user.append({"type": "text", "text": txt})
        hf_user.append({"type": "text", "text": txt})

    sg_msgs, hf_msgs = [], []
    if cfg["system"]:
        sg_msgs.append({"role": "system", "content": cfg["system"]})
        hf_msgs.append({"role": "system", "content": cfg["system"]})
    sg_msgs.append({"role": "user", "content": sg_user})
    hf_msgs.append({"role": "user", "content": hf_user})
    return sg_msgs, hf_msgs, imgs


def wait_port_free(port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(0.5)
    return False


def start_sglang(model_path: str, port: int, log_path: Path) -> subprocess.Popen:
    if not wait_port_free(port, timeout=30.0):
        raise RuntimeError(f"Port 127.0.0.1:{port} is still occupied before SGLang launch")
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--dtype", "bfloat16",
        "--context-length", "12288",
        "--mem-fraction-static", "0.85",
        "--max-running-requests", "128",
        "--chunked-prefill-size", "8192",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(300):
            if proc.poll() is not None:
                raise RuntimeError(f"SGLang exited ({proc.returncode}); check {log_path}")
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).ok:
                    return proc
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("SGLang startup timed out")
    except Exception:
        stop_proc(proc, port=port)
        raise


def stop_proc(proc: subprocess.Popen | None, port: int | None = None) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
    if port is not None:
        wait_port_free(port, timeout=30.0)
    time.sleep(1)


def eval_ext_dev_on_server(
    url: str,
    model_name: str,
    dev_rows: list[dict],
    max_tokens: int,
    max_pixels: int,
    workers: int,
    style: str = "official_cot",
    presence_penalty: float = 0.0,
) -> dict:
    """Evaluate external dev set deterministically on an already-running SGLang server."""
    def eval_one(row: dict) -> dict:
        sg_msgs, _, _ = build_prompt_for_sglang_and_hf(row, style, max_pixels)
        payload = {
            "model": model_name,
            "messages": sg_msgs,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "presence_penalty": presence_penalty,
        }
        txt, fin = "", "error"
        for _ in range(2):
            try:
                r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=300)
                r.raise_for_status()
                ch = r.json()["choices"][0]
                txt = (ch["message"].get("content") or "").strip()
                fin = ch.get("finish_reason") or "stop"
                break
            except Exception:
                time.sleep(1)
        if row["question_type"] == "multiple-choice":
            choices = [chr(65 + i) for i in range(len(row["display_options"]))]
            pred, exact_fmt = parse_strict_mc_answer(txt, choices)
            correct = bool(pred is not None and pred == row["display_gold"])
        else:
            pred, exact_fmt = parse_strict_open_answer(txt)
            correct = bool(check_open_match(pred, row["display_gold"]))
        return {
            "view_type": row["view_type"],
            "correct": correct,
            "exact_format": exact_fmt,
            "truncated": fin == "length",
        }

    recs = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(eval_one, r) for r in dev_rows]
        for fut in as_completed(futs):
            recs.append(fut.result())

    def agg(items: list[dict]) -> dict:
        n = len(items)
        return {
            "count": n,
            "acc_pct": round(sum(x["correct"] for x in items) / max(1, n) * 100, 2),
            "format_pct": round(sum(x["exact_format"] for x in items) / max(1, n) * 100, 2),
            "trunc_pct": round(sum(x["truncated"] for x in items) / max(1, n) * 100, 2),
        }

    by_v = defaultdict(list)
    for r in recs:
        by_v[r["view_type"]].append(r)
    return {
        "overall": agg(recs),
        "by_view": {v: agg(by_v[v]) for v in sorted(by_v)},
    }


def rollout_round(
    model_path: str,
    batch_rows: list[dict],
    styles: list[str],
    group_size: int,
    max_tokens: int,
    max_pixels: int,
    port: int,
    workers: int,
    log_path: Path,
    ext_dev_rows: list[dict] | None = None,
    presence_penalty: float = 0.0,
    len_ramp_start: int = 1050,
    trunc_pen: float = 0.35,
) -> tuple[list[dict], dict | None]:
    """Generate G rollouts per prompt using SGLang and score with StructReward."""
    proc = start_sglang(model_path, port, log_path)
    url = f"http://127.0.0.1:{port}"
    try:
        model_name = requests.get(f"{url}/v1/models", timeout=10).json()["data"][0]["id"]

        def sample_prompt(idx_row_style):
            idx, row, style = idx_row_style
            sg_msgs, _, _ = build_prompt_for_sglang_and_hf(row, style, max_pixels)
            rollouts = []
            # Request n=group_size in a single call for shared prefix KV cache!
            payload = {
                "model": model_name,
                "messages": sg_msgs,
                "n": group_size,
                "max_tokens": max_tokens,
                "temperature": 0.85,
                "top_p": 0.90,
                "top_k": 20,
                "presence_penalty": presence_penalty,
            }
            try:
                resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=480)
                resp.raise_for_status()
                data = resp.json()
                choices_api = data["choices"]
            except Exception:
                # Fallback to sequential calls
                choices_api = []
                for _ in range(group_size):
                    try:
                        r2 = requests.post(
                            f"{url}/v1/chat/completions",
                            json={**payload, "n": 1},
                            timeout=240,
                        )
                        r2.raise_for_status()
                        choices_api.append(r2.json()["choices"][0])
                    except Exception:
                        pass

            valid_choices = (
                [chr(65 + i) for i in range(len(row["display_options"]))]
                if row["question_type"] == "multiple-choice"
                else None
            )
            for ch in choices_api:
                txt = (ch["message"].get("content") or "").strip()
                fin = ch.get("finish_reason") or "stop"
                # Approximate tokens if usage is aggregate
                approx_tok = max(1, int(len(txt) / 3.8))
                sc = score_struct_reward(
                    response=txt,
                    gold=row["display_gold"],
                    question_type=row["question_type"],
                    valid_choices=valid_choices,
                    ref_steps=row.get("reference_steps") or [],
                    finish_reason=fin,
                    completion_tokens=approx_tok,
                    process_weight=0.50,
                    len_ramp_start=len_ramp_start,
                    trunc_pen=trunc_pen,
                )
                rollouts.append({
                    "response": txt,
                    "finish_reason": fin,
                    "approx_tokens": approx_tok,
                    **sc,
                })
            return {"row": row, "style": style, "rollouts": rollouts}

        groups = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(sample_prompt, (i, r, s)) for i, (r, s) in enumerate(zip(batch_rows, styles))]
            for fut in as_completed(futs):
                groups.append(fut.result())

        ext_dev_res = None
        if ext_dev_rows:
            ext_dev_res = eval_ext_dev_on_server(
                url=url,
                model_name=model_name,
                dev_rows=ext_dev_rows,
                max_tokens=max_tokens,
                max_pixels=max_pixels,
                workers=workers,
                presence_penalty=presence_penalty,
            )
        return groups, ext_dev_res
    finally:
        stop_proc(proc, port=port)


def summarize_rollouts_by_view(groups: list[dict]) -> dict[str, dict]:
    """Compute per-view-type (standard_mc, standard10_aug, vision_board, open) rollout metrics."""
    by_view: dict[str, list[dict]] = defaultdict(list)
    for g in groups:
        vt = g["row"].get("view_type", "unknown")
        by_view[vt].extend(g["rollouts"])
    out: dict[str, dict] = {}
    for vt in sorted(by_view):
        rs = by_view[vt]
        n = len(rs)
        out[vt] = {
            "rollouts": n,
            "mean_reward": round(sum(x["reward"] for x in rs) / max(1, n), 4),
            "acc_pct": round(sum(x["g_ans"] for x in rs) / max(1, n) * 100, 2),
            "process_raw": round(sum(x["r_process_raw"] for x in rs) / max(1, n), 4),
            "format_pct": round(sum(x["exact_format"] for x in rs) / max(1, n) * 100, 2),
            "trunc_pct": round(sum(x["truncated"] for x in rs) / max(1, n) * 100, 2),
            "avg_tokens": round(sum(x["approx_tokens"] for x in rs) / max(1, n), 1),
        }
    return out


def encode_grpo_sample(
    processor,
    hf_msgs: list[dict],
    imgs: list[Image.Image],
    response: str,
    finish_reason: str,
    max_seq_len: int,
) -> tuple[dict, int] | None:
    """Encode prompt + rollout completion matching exact SGLang EOS behavior (no post-EOS newline,
    and no synthetic EOS appended to length-truncated rollouts).
    """
    p_txt = processor.apply_chat_template(hf_msgs, tokenize=False, add_generation_prompt=True)
    eos_tok = getattr(processor.tokenizer, "eos_token", None) or "<|im_end|>"
    if finish_reason == "stop":
        f_txt = p_txt + response + eos_tok
    else:
        f_txt = p_txt + response

    proc_kwargs = {"return_tensors": "pt"}
    if imgs:
        proc_kwargs["images"] = imgs
    full = processor(text=[f_txt], **proc_kwargs)
    if full["input_ids"].shape[1] > max_seq_len:
        return None
    prmt = processor(text=[p_txt], **proc_kwargs)
    plen = int(prmt["input_ids"].shape[1])
    if plen < 1 or plen >= full["input_ids"].shape[1] - 1:
        return None
    if not torch.equal(full["input_ids"][0, :plen], prmt["input_ids"][0, :plen]):
        return None
    for key in ("pixel_values", "image_grid_thw"):
        if key in full and key in prmt and not torch.equal(full[key], prmt[key]):
            return None
    return full, plen


def token_logprobs(model, inputs: dict, prompt_len: int) -> torch.Tensor:
    """Compute per-token log-probabilities over the completion tokens [prompt_len:]."""
    seq_len = inputs["input_ids"].shape[1]
    if not (1 <= prompt_len < seq_len):
        raise ValueError(f"Invalid prompt_len={prompt_len} for seq_len={seq_len}")
    out = model(**inputs)
    logits = out.logits[:, prompt_len - 1 : -1, :]  # predict tokens [prompt_len:]
    target = inputs["input_ids"][:, prompt_len:]
    logp = F.log_softmax(logits.float(), dim=-1)
    selected = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return selected  # shape [B, comp_len]


def compute_grpo_step_loss(
    new_lp: torch.Tensor,
    old_lp: torch.Tensor,
    ref_lp: torch.Tensor,
    adv: float,
    eps_low: float,
    eps_high: float,
    kl_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute asymmetric DAPO/GRPO clipped surrogate loss + Schulman k3 reverse KL."""
    log_ratio = (new_lp - old_lp).clamp(-10.0, 10.0)
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
    pg = -torch.minimum(ratio * adv, clipped * adv).mean()

    # Schulman k3 unbiased non-negative KL estimator: exp(ref - new) - (ref - new) - 1
    diff_ref = (ref_lp - new_lp).clamp(-10.0, 10.0)
    kl = (torch.exp(diff_ref) - diff_ref - 1.0).mean()

    loss = pg + kl_beta * kl
    clip_frac = ((ratio < 1.0 - eps_low) | (ratio > 1.0 + eps_high)).float().mean()
    return loss, pg, kl, clip_frac


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-model", required=True, help="Stage 1 SFT-merged BF16 model (also frozen ref)")
    p.add_argument("--rl-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--prompts-per-round", type=int, default=96)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=6e-6)
    p.add_argument("--kl-beta", type=float, default=0.015)
    p.add_argument("--eps-low", type=float, default=0.20)
    p.add_argument("--eps-high", type=float, default=0.28)
    p.add_argument("--rollout-presence-penalty", type=float, default=0.0)
    p.add_argument("--official-ratio", type=float, default=0.80, help="Fraction of prompts using official_cot")
    p.add_argument("--len-ramp-start", type=int, default=1050)
    p.add_argument("--trunc-pen", type=float, default=0.35)
    p.add_argument("--max-tokens", type=int, default=1280)
    p.add_argument("--max-seq-len", type=int, default=3072)
    p.add_argument("--max-pixels", type=int, default=786432)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--eval-ext-dev", type=Path, default=None, help="Optional external dev JSONL evaluated during Phase A")
    p.add_argument("--eval-every", type=int, default=2, help="Run --eval-ext-dev every N rounds (plus round 1 and final)")
    p.add_argument("--resume", action="store_true", help="Resume from latest_adapter + current_policy_merged if present")
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--port", type=int, default=30300)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "rl_metrics.jsonl"

    pool_rows = [json.loads(l) for l in args.rl_file.read_text().splitlines() if l.strip()]
    rng.shuffle(pool_rows)
    print(f"Loaded {len(pool_rows)} external RL prompts.", flush=True)

    ext_dev_rows = None
    if args.eval_ext_dev and args.eval_ext_dev.exists():
        ext_dev_rows = [json.loads(l) for l in args.eval_ext_dev.read_text().splitlines() if l.strip()]
        print(f"Loaded {len(ext_dev_rows)} external dev prompts from {args.eval_ext_dev}.", flush=True)

    processor = AutoProcessor.from_pretrained(args.init_model, min_pixels=4096, max_pixels=1048576)
    current_merged = Path(args.init_model)
    adapter_dir = args.out_dir / "latest_adapter"
    scratch_merged = args.out_dir / "current_policy_merged"
    start_round = 1
    if args.resume and adapter_dir.exists() and scratch_merged.exists():
        current_merged = scratch_merged
        if metrics_path.exists():
            done_rounds = []
            for l in metrics_path.read_text().splitlines():
                if l.strip():
                    try:
                        done_rounds.append(int(json.loads(l)["round"]))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass
            if done_rounds:
                start_round = max(done_rounds) + 1
        print(f"Resuming from round {start_round} using {scratch_merged}", flush=True)
    elif not args.resume:
        # Prevent silent off-policy mismatch from stale adapters in the same --out-dir
        for stale in (adapter_dir, scratch_merged):
            if stale.exists():
                print(f"Removing stale directory before fresh run: {stale}", flush=True)
                shutil.rmtree(stale)

    cursor = ((start_round - 1) * args.prompts_per_round) % max(1, len(pool_rows))
    t_global = time.monotonic()

    for rnd in range(start_round, args.rounds + 1):
        t_rnd = time.monotonic()
        # Draw prompts cyclically with reshuffle
        batch_rows = []
        for _ in range(args.prompts_per_round):
            if cursor >= len(pool_rows):
                rng.shuffle(pool_rows)
                cursor = 0
            batch_rows.append(pool_rows[cursor])
            cursor += 1
        styles = ["official_cot" if rng.random() < args.official_ratio else "calibrated_cot" for _ in batch_rows]

        run_dev_this_round = bool(
            ext_dev_rows and (rnd == 1 or rnd == args.rounds or rnd % max(1, args.eval_every) == 0)
        )
        print(f"\n=== Round {rnd}/{args.rounds}: Phase A (SGLang Rollouts: {len(batch_rows)}x{args.group_size}) ===", flush=True)
        groups, ext_dev_res = rollout_round(
            model_path=str(current_merged),
            batch_rows=batch_rows,
            styles=styles,
            group_size=args.group_size,
            max_tokens=args.max_tokens,
            max_pixels=args.max_pixels,
            port=args.port,
            workers=args.workers,
            log_path=args.out_dir / f"sglang_round_{rnd}.log",
            ext_dev_rows=ext_dev_rows if run_dev_this_round else None,
            presence_penalty=args.rollout_presence_penalty,
            len_ramp_start=args.len_ramp_start,
            trunc_pen=args.trunc_pen,
        )

        # Phase B: StructReward summary & Dynamic non-zero-variance filtering
        all_r = [ro for g in groups for ro in g["rollouts"]]
        mean_rew = sum(x["reward"] for x in all_r) / max(1, len(all_r))
        acc_rate = sum(x["g_ans"] for x in all_r) / max(1, len(all_r)) * 100
        proc_mean = sum(x["r_process_raw"] for x in all_r) / max(1, len(all_r))
        fmt_rate = sum(x["exact_format"] for x in all_r) / max(1, len(all_r)) * 100
        trunc_rate = sum(x["truncated"] for x in all_r) / max(1, len(all_r)) * 100
        avg_tok = sum(x["approx_tokens"] for x in all_r) / max(1, len(all_r))
        by_view = summarize_rollouts_by_view(groups)

        train_items = []
        nzv_groups = 0
        all_correct_groups = 0
        all_wrong_groups = 0
        for g in groups:
            ros = g["rollouts"]
            if len(ros) < 2:
                continue
            rews = [x["reward"] for x in ros]
            g_ans_sum = sum(x["g_ans"] for x in ros)
            if g_ans_sum == len(ros):
                all_correct_groups += 1
            elif g_ans_sum == 0:
                all_wrong_groups += 1
            mean_g = sum(rews) / len(rews)
            var_g = sum((r - mean_g) ** 2 for r in rews) / len(rews)
            std_g = math.sqrt(var_g)
            if std_g < 1e-4:
                continue
            nzv_groups += 1
            # Dr.GRPO-inspired normalization: scale by max(std, 0.25) to avoid exploding tiny variance
            denom = max(std_g, 0.25)
            for ro in ros:
                adv = max(-3.0, min(3.0, (ro["reward"] - mean_g) / denom))
                if abs(adv) < 0.05 or not ro["response"].strip():
                    continue
                train_items.append({
                    "row": g["row"],
                    "style": g["style"],
                    "response": ro["response"],
                    "finish_reason": ro["finish_reason"],
                    "advantage": adv,
                    "reward": ro["reward"],
                })

        print(
            f"Round {rnd} Rollout Stats: mean_R={mean_rew:.3f} | acc={acc_rate:.1f}% | proc={proc_mean:.3f} | "
            f"fmt={fmt_rate:.1f}% | trunc={trunc_rate:.1f}% | tok={avg_tok:.0f} | "
            f"nzv_groups={nzv_groups}/{len(groups)} (all_ok={all_correct_groups}, all_0={all_wrong_groups}) | "
            f"train_items={len(train_items)}",
            flush=True,
        )
        view_line = " | ".join(
            f"{vt}: acc={st['acc_pct']:.1f}% R={st['mean_reward']:.3f} (n={st['rollouts']})"
            for vt, st in by_view.items()
        )
        print(f"  By-View Rollouts: {view_line}", flush=True)
        if ext_dev_res is not None:
            ov = ext_dev_res["overall"]
            dv_line = " | ".join(
                f"{vt}={st['acc_pct']:.1f}%({st['count']})"
                for vt, st in ext_dev_res["by_view"].items()
            )
            print(
                f"  ExtDev Greedy: overall={ov['acc_pct']:.1f}% (fmt={ov['format_pct']:.1f}%, trunc={ov['trunc_pct']:.1f}%) | {dv_line}",
                flush=True,
            )

        if len(train_items) < 16:
            print(f"Round {rnd}: too few active items ({len(train_items)}), skipping update.", flush=True)
            continue

        # Phase C: PyTorch BF16 GRPO/DAPO Update
        print(f"=== Round {rnd}/{args.rounds}: Phase C (GRPO Update on {len(train_items)} rollouts) ===", flush=True)
        base_model = AutoModelForImageTextToText.from_pretrained(
            args.init_model,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        )
        if adapter_dir.exists():
            model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
        else:
            lora_cfg = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.0,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(base_model, lora_cfg)

        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        set_model_use_cache(model, False)

        # Precompute encoded tensors, ref_logprobs, old_logprobs
        prepared = []
        prep_err = 0
        model.eval()
        with torch.no_grad():
            for item in train_items:
                try:
                    _, hf_msgs, imgs = build_prompt_for_sglang_and_hf(item["row"], item["style"], args.max_pixels)
                    encoded = encode_grpo_sample(
                        processor=processor,
                        hf_msgs=hf_msgs,
                        imgs=imgs,
                        response=item["response"],
                        finish_reason=item["finish_reason"],
                        max_seq_len=args.max_seq_len,
                    )
                    if encoded is None:
                        continue
                    full, plen = encoded
                    gpu_in = {k: v.to("cuda", non_blocking=True) for k, v in full.items()}
                    old_lp = token_logprobs(model, gpu_in, plen).cpu()
                    with model.disable_adapter():
                        ref_lp = token_logprobs(model, gpu_in, plen).cpu()
                    prepared.append({
                        "cpu_in": {k: v.cpu() for k, v in full.items()},
                        "plen": plen,
                        "old_lp": old_lp,
                        "ref_lp": ref_lp,
                        "adv": float(item["advantage"]),
                    })
                except Exception as e:
                    prep_err += 1
                    if prep_err <= 5:
                        print(f"  [prepare skip {prep_err}] {item['row'].get('id')}: {type(e).__name__}: {e}", flush=True)
                    continue

        print(f"  Prepared {len(prepared)} valid rollouts for GRPO update (errors={prep_err}).", flush=True)
        if not prepared:
            del model, base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # Cosine LR across rounds
        lr_round = args.lr * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * (rnd - 1) / max(1, args.rounds))))
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr_round,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )
        model.train()

        grad_accum = args.grad_accum
        total_pg_loss = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        cnt_steps = 0
        opt_updates = 0

        for ep in range(args.ppo_epochs):
            rng.shuffle(prepared)
            opt.zero_grad()
            micro = 0
            for ex in prepared:
                gpu_in = {k: v.to("cuda", non_blocking=True) for k, v in ex["cpu_in"].items()}
                old_lp = ex["old_lp"].to("cuda", non_blocking=True)
                ref_lp = ex["ref_lp"].to("cuda", non_blocking=True)
                adv = ex["adv"]

                new_lp = token_logprobs(model, gpu_in, ex["plen"])
                loss, pg, kl, clip_f = compute_grpo_step_loss(
                    new_lp=new_lp,
                    old_lp=old_lp,
                    ref_lp=ref_lp,
                    adv=adv,
                    eps_low=args.eps_low,
                    eps_high=args.eps_high,
                    kl_beta=args.kl_beta,
                )
                if not torch.isfinite(loss):
                    continue
                (loss / grad_accum).backward()
                total_pg_loss += pg.item()
                total_kl += kl.item()
                total_clip_frac += clip_f.item()
                cnt_steps += 1
                micro += 1

                if micro >= grad_accum:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                    opt_updates += 1
                    micro = 0
            if micro > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                opt_updates += 1

        # Save adapter & merge for next round's SGLang rollout
        model.eval()
        model.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)
        if rnd in (4, 6, 8, args.rounds):
            ckpt_adp = args.out_dir / f"adapter_round_{rnd}"
            model.save_pretrained(ckpt_adp)
            processor.save_pretrained(ckpt_adp)

        merged = model.merge_and_unload()
        if scratch_merged.exists():
            shutil.rmtree(scratch_merged)
        save_merged_model_for_sglang(merged, processor, args.init_model, scratch_merged)
        current_merged = scratch_merged

        del merged, model, base_model, opt, prepared
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rec = {
            "round": rnd,
            "lr": lr_round,
            "mean_reward": round(mean_rew, 4),
            "acc_pct": round(acc_rate, 2),
            "process_raw": round(proc_mean, 4),
            "format_pct": round(fmt_rate, 2),
            "trunc_pct": round(trunc_rate, 2),
            "avg_tokens": round(avg_tok, 1),
            "by_view": by_view,
            "ext_dev": ext_dev_res,
            "nzv_groups": nzv_groups,
            "total_groups": len(groups),
            "all_correct_groups": all_correct_groups,
            "all_wrong_groups": all_wrong_groups,
            "opt_updates": opt_updates,
            "pg_loss": round(total_pg_loss / max(1, cnt_steps), 5),
            "kl_div": round(total_kl / max(1, cnt_steps), 5),
            "clip_frac": round(total_clip_frac / max(1, cnt_steps), 4),
            "round_sec": round(time.monotonic() - t_rnd, 1),
            "total_sec": round(time.monotonic() - t_global, 1),
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"Round {rnd} Done in {rec['round_sec']}s | updates={opt_updates} | "
            f"KL={rec['kl_div']:.4f} | clip={rec['clip_frac']:.3f} | total={rec['total_sec']}s",
            flush=True,
        )

    final_merged = args.out_dir / "final_merged"
    if final_merged.exists():
        shutil.rmtree(final_merged)
    shutil.copytree(current_merged, final_merged)
    print(f"=== StructReward-GRPO Complete -> {final_merged} ===", flush=True)


if __name__ == "__main__":
    main()
