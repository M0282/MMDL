#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E9: SFT 학습 (A군 / B군), LoRA, Colab Pro
================================================
E8에서 만든 train_A.jsonl / train_B.jsonl 로 Qwen3-VL-4B를 LoRA 학습한다.
E2에서 L4로 환경이 되는 걸 확인했으니, 같은 설정(r=16, bf16)으로 진짜 학습.

A군과 B군을 '따로' 한 번씩 돌린다 (--train 으로 파일 지정):
    python e9_train.py --train train_A.jsonl --image_root <이미지폴더> --out adapter_A --tag A
    python e9_train.py --train train_B.jsonl --image_root <이미지폴더> --out adapter_B --tag B

핵심 설계:
- 이미지 경로는 jsonl에 상대경로로 돼 있지만, 파일명(해시)이 고유해서 --image_root 아래를
  한 번 훑어 '파일명 → 실제경로' 색인을 만들고 그걸로 찾는다. (상대경로 깨짐 방지)
  → 이미지를 못 찾은 문항은 건너뛰고 몇 건인지 보고한다.
- loss는 '어시스턴트(정답/풀이) 부분'에만 건다. 문제(사용자) 부분은 마스킹(-100)한다.
  이게 SFT 기본이고, E2(스모크)와 다른 점이다.

설치(셀):
  !pip install -q -U transformers accelerate peft datasets pillow bitsandbytes
드라이브 마운트(셀):
  from google.colab import drive; drive.mount('/content/drive')
"""

import argparse, json, os, time, glob
import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

def log(m): print(f"[E9] {m}", flush=True)
def vram():
    return torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0.0

def build_image_index(image_root):
    """ image_root 아래 모든 이미지의 '파일명 → 전체경로' 색인 """
    idx = {}
    exts = (".jpg", ".jpeg", ".png", ".webp", ".gif")
    for dp, _, files in os.walk(image_root):
        for f in files:
            if f.lower().endswith(exts):
                idx.setdefault(f, os.path.join(dp, f))
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train_A.jsonl 또는 train_B.jsonl")
    ap.add_argument("--image_root", required=True, help="이미지들이 들어있는 폴더(하위까지 훑음)")
    ap.add_argument("--out", required=True, help="LoRA 어댑터 저장 폴더")
    ap.add_argument("--tag", default="")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--max_pixels", type=int, default=602112, help="이미지 해상도 상한(=768*28*28). 메모리/길이 조절")
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="앞 N개만 (0=전체). 스모크용")
    ap.add_argument("--save_every", type=int, default=500, help="N스텝마다 중간 저장")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU 런타임이 아님!"
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log("이미지 색인 만드는 중...")
    img_idx = build_image_index(args.image_root)
    log(f"  이미지 {len(img_idx)}개 색인")

    log("모델 로드 중...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map="cuda")
    # 이미지 해상도 상한을 둬서 이미지 토큰 수를 안정화 (안 두면 큰 이미지가 터뜨림)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, max_pixels=args.max_pixels, min_pixels=256*28*28)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    # 메모리 절약: gradient checkpointing (긴 멀티모달 학습 필수) + 입력에 grad 전달
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train(); model.config.use_cache = False
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"LoRA 학습 파라미터 {tr/1e6:.1f}M · max_pixels={args.max_pixels} · max_len={args.max_len}")

    # ---- 데이터 로드 + 이미지 매칭 ----
    rows = []
    miss = 0
    for line in open(args.train):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        paths = []
        ok = True
        for p in r.get("images", []):
            fn = os.path.basename(p)
            if fn in img_idx:
                paths.append(img_idx[fn])
            else:
                ok = False; break
        if not ok:
            miss += 1; continue
        rows.append((r["messages"], paths))
        if args.limit and len(rows) >= args.limit:
            break
    log(f"학습 문항 {len(rows)}개 (이미지 못 찾아 건너뜀 {miss}개)")
    if not rows:
        log("학습할 데이터가 없음! --image_root 경로를 확인하세요."); return

    def encode(messages, paths):
        """ 전체(문제+정답) 토큰화 + 문제 부분 라벨 마스킹(-100).
        길이가 max_len을 넘으면 None 반환(건너뜀). 이미지 토큰은 절대 자르지 않는다
        (자르면 이미지 조각 수와 자리표시 토큰 수가 안 맞아 터짐). """
        imgs = [Image.open(p).convert("RGB") for p in paths]
        full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_msgs = [messages[0]]
        prompt_text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full = processor(text=[full_text], images=[imgs], return_tensors="pt")
        prmt = processor(text=[prompt_text], images=[imgs], return_tensors="pt")
        ids = full["input_ids"]
        if ids.shape[1] > args.max_len:        # 너무 길면 통째로 건너뜀 (자르지 않음)
            return None
        labels = ids.clone()
        plen = prmt["input_ids"].shape[1]
        if plen < ids.shape[1] and torch.equal(ids[0, :plen], prmt["input_ids"][0, :plen]):
            labels[0, :plen] = -100            # 문제 부분은 loss에서 제외
        full = {k: v.to("cuda") for k, v in full.items()}
        full["labels"] = labels.to("cuda")
        return full

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    n_steps = int(len(rows) * args.epochs)
    log(f"학습 시작: {args.epochs}에폭 ≈ {n_steps}스텝, grad_accum {args.grad_accum}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    opt.zero_grad()
    step = 0; skipped_long = 0
    for ep in range(int(args.epochs) + 1):
        for i, (messages, paths) in enumerate(rows):
            if step >= n_steps: break
            try:
                batch = encode(messages, paths)
                if batch is None:               # max_len 초과 → 건너뜀
                    skipped_long += 1; continue
                out = model(**batch)
                loss = out.loss / args.grad_accum
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); log(f"  [OOM] step {step} 건너뜀"); opt.zero_grad(); step+=1; continue
            if (step + 1) % args.grad_accum == 0:
                opt.step(); opt.zero_grad()
            if step % 20 == 0:
                el = (time.time()-t0)/60
                log(f"  step {step}/{n_steps}  loss={out.loss.item():.3f}  {el:.1f}분  VRAM {vram():.1f}GB")
            if args.save_every and step>0 and step % args.save_every == 0:
                model.save_pretrained(args.out); log(f"  중간 저장 → {args.out}")
            step += 1
        if step >= n_steps: break

    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    log("=" * 50)
    log(f"학습 완료 ({args.tag})")
    log(f"  어댑터 저장: {args.out}")
    log(f"  총 {step}스텝, {(time.time()-t0)/60:.0f}분, peak VRAM {vram():.1f}GB")
    log(f"  길이 초과로 건너뛴 문항: {skipped_long}개 (max_len {args.max_len} 넘김)")
    log("  다음: 이 어댑터를 base에 merge해서 run_stage0_vllm.py의 --model_path로 평가")

if __name__ == "__main__":
    main()
