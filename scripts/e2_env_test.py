#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2: 학습 환경 테스트 (담당: 김소리, Colab Pro)
================================================
목적: 진짜 학습(E9) 전에, Qwen3-VL-4B를 LoRA로 Colab에서 학습시킬 수 있는지
      '더미 데이터'로 미리 확인한다. 모델을 제대로 가르치는 게 아니라,
      (1) 에러 없이 도는가  (2) peak VRAM 얼마  (3) 1스텝 몇 초  만 본다.

→ 이 결과로 "900문항 1에폭에 몇 시간"을 어림하고, 안 되면(OOM 등) 계획을 바꾼다.

주의:
- 반드시 GPU 런타임. Colab Pro면 런타임 → 런타임 유형 변경 → GPU(A100/L4 권장).
- 더미 데이터 16개(이미지 포함)로 8스텝만 돈다. 결과 모델은 버린다.
- 처음엔 모델 다운로드(약 8GB)로 몇 분 걸린다.

설치 (셀에서 먼저):
  !pip install -q -U transformers accelerate peft datasets pillow bitsandbytes
실행:
  !python e2_env_test.py
"""

import time, random
import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
N_DUMMY = 16        # 더미 샘플 수
N_STEPS = 8         # 돌려볼 스텝 수
BATCH = 1           # 멀티모달은 보통 배치 1에서 시작

def log(msg): print(f"[E2] {msg}", flush=True)

def vram():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0

def main():
    assert torch.cuda.is_available(), "GPU 런타임이 아님! 런타임 유형을 GPU로 바꾸세요."
    gpu = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    log(f"GPU: {gpu} (총 {total:.1f}GB)")

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    # 1) 모델 로드 (bf16) --------------------------------------------------------
    log("모델 로드 중... (처음이면 다운로드로 몇 분)")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    log(f"모델 로드 완료 ({time.time()-t0:.0f}초), 로드 후 VRAM {vram():.1f}GB")

    # 2) LoRA 붙이기 -------------------------------------------------------------
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 언어부 attention
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    log(f"LoRA 적용: 학습 파라미터 {trainable/1e6:.1f}M / 전체 {total_p/1e9:.2f}B "
        f"({100*trainable/total_p:.3f}%)")
    model.train()
    model.config.use_cache = False

    # 3) 더미 데이터 만들기 (이미지 포함 멀티모달) --------------------------------
    log("더미 데이터 생성 (이미지 포함)")
    samples = []
    for i in range(N_DUMMY):
        img = Image.new("RGB", (448, 448),
                        (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        letter = random.choice("ABCD")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "What is the answer?\n(A) a\n(B) b\n(C) c\n(D) d"}]},
            {"role": "assistant", "content": [
            {"type": "text", "text": f"Let me think. The answer is clear.\nAnswer: ({letter})"}]}]
        samples.append(messages)

    def collate(batch_msgs):
        texts, imgs = [], []
        for msgs in batch_msgs:
            texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
            # 이미지 추출
            img_list = []
            for m in msgs:
                for c in m["content"]:
                    if isinstance(c, dict) and c.get("type") == "image":
                        img_list.append(c["image"])
            imgs.append(img_list)
        enc = processor(text=texts, images=imgs, return_tensors="pt", padding=True)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        enc["labels"] = enc["input_ids"].clone()
        return enc

    # 4) 몇 스텝 학습 ------------------------------------------------------------
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    log(f"{N_STEPS}스텝 학습 시작 (batch={BATCH})")
    torch.cuda.reset_peak_memory_stats()
    step_times = []
    for step in range(N_STEPS):
        batch = collate(samples[step*BATCH:(step+1)*BATCH] or samples[:BATCH])
        t = time.time()
        out = model(**batch)
        loss = out.loss
        loss.backward()
        opt.step(); opt.zero_grad()
        dt = time.time() - t
        step_times.append(dt)
        log(f"  step {step+1}/{N_STEPS}  loss={loss.item():.3f}  {dt:.1f}s  VRAM {vram():.1f}GB")

    # 5) 요약 --------------------------------------------------------------------
    warm = step_times[1:] if len(step_times) > 1 else step_times  # 1스텝째는 컴파일 때문에 느림
    avg = sum(warm)/len(warm)
    peak = vram()
    log("=" * 50)
    log(f"결과 요약")
    log(f"  GPU: {gpu} ({total:.1f}GB)")
    log(f"  peak VRAM: {peak:.1f}GB  (여유 {total-peak:.1f}GB)")
    log(f"  스텝 시간(평균, 1스텝째 제외): {avg:.1f}초/스텝  @ batch {BATCH}")
    log(f"  → 900문항 1에폭 어림: {900*avg/BATCH/60:.0f}분")
    log(f"  → 3에폭 어림: {3*900*avg/BATCH/60:.0f}분")
    log("※ 실제 데이터는 풀이가 길어 토큰 수가 더 많으니 이보다 더 걸릴 수 있음.")
    log("※ OOM 없이 여기까지 왔으면 이 설정(r=16, batch=1, bf16)으로 E9 가능.")

if __name__ == "__main__":
    main()
