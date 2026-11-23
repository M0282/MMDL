#!/usr/bin/env python3
"""Stage 1: Lightweight Knowledge & Concise-Reasoning SFT for Qwen3-VL-4B-Instruct (BF16).

Features:
- Pinned BF16 base model (no quantization), language-only LoRA r=32 alpha=64 by default
  (q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj; optional --include-merger for
  visual.merger.linear_fc1/2 without touching ViT blocks).
- Trains on 50% official_cot and 50% calibrated_cot prompt templates so gains transfer strongly
  to BOTH exact official MMMU-Pro prompts AND calibrated prompts.
- Assistant-only loss masking (-100 on prompt + vision tokens).
- Exports both LoRA adapter and clean SGLang-compatible merged BF16 weights for fast eval & Stage 2 RL.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time

from PIL import Image
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

TAG_RE = re.compile(r"<image\s*\d*>", re.I)

PROMPT_STYLES = {
    "official_cot": {
        "system": None,
        "standard": (
            "Answer the preceding multiple choice question. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
            "Think step by step before answering."
        ),
        "vision": (
            "Write out the multiple-choice question in the image and then solve it. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
            "Think step by step before answering."
        ),
        "open": (
            "Answer the preceding question. "
            "The last line of your response should be of the following format: "
            "'Answer: <FINAL_ANSWER>'. Think step by step before answering."
        ),
    },
    "calibrated_cot": {
        "system": (
            "You are a precise multimodal reasoning expert. Carefully inspect the image(s), "
            "reason concisely step by step without repetition, verify your result against "
            "the exact option labels, and end with 'Answer: $LETTER' on the final line."
        ),
        "standard": (
            "Solve this multiple-choice question concisely step by step:\n"
            "1. Extract the exact visual/numerical details needed from the image(s).\n"
            "2. Perform the domain reasoning or calculation clearly and concisely.\n"
            "3. Match your final result to the exact option letter and verify there is no label mismatch.\n"
            "End your response with the final line in the exact format: Answer: $LETTER"
        ),
        "vision": (
            "Write out the question and relevant options from the image concisely, then solve step by step:\n"
            "1. Transcribe the question, key visual/numerical details, and relevant options clearly.\n"
            "2. Perform the domain reasoning or calculation concisely without repetition.\n"
            "3. Verify which exact lettered option in the image matches your result.\n"
            "End your response with the final line in the exact format: Answer: $LETTER"
        ),
        "open": (
            "Solve this question concisely step by step:\n"
            "1. Extract the exact visual/numerical details needed from the image(s).\n"
            "2. Perform the domain reasoning or calculation clearly and concisely.\n"
            "3. Verify your final value.\n"
            "End your response with the final line in the exact format: Answer: <FINAL_ANSWER>"
        ),
    },
}


def resize_pil(path: str, max_pixels: int) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w * h > max_pixels:
            sc = math.sqrt(max_pixels / (w * h))
            nw = max(32, int(w * sc) // 32 * 32)
            nh = max(32, int(h * sc) // 32 * 32)
            im = im.resize((nw, nh), Image.Resampling.BICUBIC)
        elif w * h < 4096 or min(w, h) < 32:
            sc = max(math.sqrt(4096 / max(1, w * h)), 32.0 / max(1, min(w, h)))
            nw = max(32, math.ceil((w * sc) / 32) * 32)
            nh = max(32, math.ceil((h * sc) / 32) * 32)
            im = im.resize((nw, nh), Image.Resampling.BICUBIC)
        return im.copy()


def set_model_use_cache(model, enabled: bool) -> None:
    """Properly toggle use_cache on Qwen3-VL (lives on config.text_config) and standard models."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "use_cache"):
        text_cfg.use_cache = enabled
    elif hasattr(cfg, "use_cache"):
        cfg.use_cache = enabled


def save_merged_model_for_sglang(merged, processor, base_model_path: str | Path, out_dir: Path) -> None:
    """Save merged BF16 Qwen3-VL weights + processor with full SGLang/transformers 5.x compatibility."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_model_use_cache(merged, True)
    merged.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

    # Transformers 5.x saves processor_config.json + chat_template.jinja; also emit/preserve
    # legacy compatibility files (preprocessor_config.json, chat_template.json, etc.) for SGLang.
    pconf_path = out_dir / "processor_config.json"
    preproc_path = out_dir / "preprocessor_config.json"
    if not preproc_path.exists() and pconf_path.exists():
        pconf = json.loads(pconf_path.read_text(encoding="utf-8"))
        img_cfg = dict(pconf.get("image_processor") or {})
        if img_cfg:
            img_cfg.setdefault("processor_class", pconf.get("processor_class", "Qwen3VLProcessor"))
            preproc_path.write_text(json.dumps(img_cfg, indent=2) + "\n", encoding="utf-8")

    src = Path(base_model_path)
    if src.is_dir():
        for fn in (
            "chat_template.json",
            "video_preprocessor_config.json",
            "vocab.json",
            "merges.txt",
            "generation_config.json",
            "preprocessor_config.json",
        ):
            src_f = src / fn
            dst_f = out_dir / fn
            if src_f.is_file() and not dst_f.exists():
                shutil.copy2(src_f, dst_f)


def build_training_sample(row: dict, style_name: str, max_pixels: int):
    cfg = PROMPT_STYLES[style_name]
    view = row["view_type"]
    raw_imgs = row["display_images"][:1] if view == "vision_board" else row["display_images"]
    imgs = [resize_pil(p, max_pixels if view != "vision_board" else max(max_pixels, 1048576)) for p in raw_imgs]

    user_content = []
    if view == "vision_board":
        user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": cfg["vision"]})
    else:
        for _ in imgs:
            user_content.append({"type": "image"})
        q_clean = TAG_RE.sub("", row["question"]).strip()
        if row["question_type"] == "multiple-choice":
            opts_txt = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(row["display_options"]))
            user_content.append({"type": "text", "text": f"{q_clean}\n{opts_txt}\n{cfg['standard']}"})
        else:
            user_content.append({"type": "text", "text": f"{q_clean}\n{cfg['open']}"})

    prompt_msgs = []
    if cfg["system"]:
        prompt_msgs.append({"role": "system", "content": cfg["system"]})
    prompt_msgs.append({"role": "user", "content": user_content})

    full_msgs = prompt_msgs + [{"role": "assistant", "content": row["verified_rationale"]}]
    return prompt_msgs, full_msgs, imgs


def encode_sft(processor, prompt_msgs, full_msgs, imgs, max_len: int):
    prompt_text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    proc_kwargs = {"return_tensors": "pt"}
    if imgs:
        proc_kwargs["images"] = imgs
    full = processor(text=[full_text], **proc_kwargs)
    ids = full["input_ids"]
    if ids.shape[1] > max_len:
        return None
    prmt = processor(text=[prompt_text], **proc_kwargs)
    plen = prmt["input_ids"].shape[1]
    if plen >= ids.shape[1] or not torch.equal(ids[0, :plen], prmt["input_ids"][0, :plen]):
        return None
    for key in ("pixel_values", "image_grid_thw"):
        if key in full and key in prmt and not torch.equal(full[key], prmt[key]):
            return None
    labels = ids.clone()
    labels[0, :plen] = -100
    full["labels"] = labels
    return full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--max-len", type=int, default=3072)
    p.add_argument("--max-pixels", type=int, default=786432)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--include-merger", action="store_true", help="Also apply LoRA to visual.merger.linear_fc1/2 (not ViT blocks)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in args.train_file.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("verified_rationale")]
    n_tv = sum(1 for r in rows if r.get("teacher_verified"))
    rng.shuffle(rows)
    if args.max_samples:
        rows = rows[: args.max_samples]
    print(f"Loaded {len(rows)} verified SFT rows (teacher_verified={n_tv}).", flush=True)

    processor = AutoProcessor.from_pretrained(args.base_model, min_pixels=4096, max_pixels=1048576)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.include_merger:
        target_modules += ["merger.linear_fc1", "merger.linear_fc2"]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    set_model_use_cache(model, False)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Trainable LoRA params: {trainable/1e6:.2f}M / {total_p/1e6:.1f}M ({trainable/total_p*100:.2f}%)", flush=True)

    total_examples = int(len(rows) * args.epochs)
    total_opt_steps = max(1, math.ceil(total_examples / args.grad_accum))
    warmup_steps = min(max(1, total_opt_steps // 2), max(5, int(total_opt_steps * 0.06)))

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps)

    t0 = time.monotonic()
    log_path = args.out_dir / "train_metrics.jsonl"
    opt.zero_grad()

    accum_loss = 0.0
    accum_cnt = 0
    opt_step = 0
    skipped = 0
    seen = 0

    for ep in range(int(math.ceil(args.epochs))):
        rng.shuffle(rows)
        for row in rows:
            if seen >= total_examples:
                break
            seen += 1
            style = "official_cot" if rng.random() < 0.5 else "calibrated_cot"
            try:
                p_msgs, f_msgs, imgs = build_training_sample(row, style, args.max_pixels)
                batch = encode_sft(processor, p_msgs, f_msgs, imgs, args.max_len)
                if batch is None:
                    skipped += 1
                    continue
                batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                (loss / args.grad_accum).backward()
                accum_loss += loss.item()
                accum_cnt += 1
            except Exception as e:
                skipped += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if skipped <= 10:
                    print(f"  [skip {skipped}] {row['id']}: {e}", flush=True)
                continue

            if accum_cnt >= args.grad_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                opt.step()
                sched.step()
                opt.zero_grad()
                opt_step += 1
                avg_loss = accum_loss / accum_cnt
                lr_now = sched.get_last_lr()[0]
                rec = {
                    "opt_step": opt_step,
                    "seen": seen,
                    "loss": round(avg_loss, 4),
                    "lr": lr_now,
                    "grad_norm": round(float(grad_norm), 4),
                    "skipped": skipped,
                    "elapsed_sec": round(time.monotonic() - t0, 1),
                }
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                if opt_step % 10 == 0 or opt_step == 1 or opt_step == total_opt_steps:
                    vram = torch.cuda.max_memory_allocated() / (1024**3)
                    print(
                        f"  step {opt_step}/{total_opt_steps} | loss={avg_loss:.4f} | lr={lr_now:.2e} | "
                        f"gnorm={grad_norm:.2f} | vram={vram:.1f}GB | {rec['elapsed_sec']:.0f}s",
                        flush=True,
                    )
                accum_loss = 0.0
                accum_cnt = 0

    if accum_cnt > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        opt_step += 1

    adapter_dir = args.out_dir / "adapter"
    model.eval()
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    print("Merging LoRA adapter into clean BF16 model ...", flush=True)
    merged_dir = args.out_dir / "merged"
    merged = model.merge_and_unload()
    save_merged_model_for_sglang(merged, processor, args.base_model, merged_dir)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "train_file": str(args.train_file),
        "examples_loaded": len(rows),
        "examples_seen": seen,
        "skipped": skipped,
        "opt_steps": opt_step,
        "elapsed_sec": round(time.monotonic() - t0, 1),
        "merged_dir": str(merged_dir),
    }
    (args.out_dir / "sft_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"=== SFT Complete in {summary['elapsed_sec']}s -> {merged_dir} ===", flush=True)


if __name__ == "__main__":
    main()
