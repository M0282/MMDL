# MMMU-Pro Zero-Leakage StructReward-GRPO Pipeline

`Qwen/Qwen3-VL-4B-Instruct` (revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, BF16)의 **MMMU-Pro 전체 3,460문항 (`Standard10` 1,730 + `Vision` 1,730)** 성능 향상을 위한 Zero-Leakage 사후학습(SFT 진단 + Base 직결 On-Policy StructReward-GRPO RL) 및 SGLang 고속 평가 파이프라인입니다.

## 주요 구성
- `setup_mmmu_pro_eval.py`: 고정 리비전 모델 및 MMMU-Pro 평가 세트(`eval_data/mmmu_pro/`) 격리 Export
- `prepare_training_pools.py` / `prepare_hard_stem_rl_pool.py`: 영어 전용 필터링, 이미지 pHash 그룹 완전 분리(SFT/RL/ExtDev), 10지선다 외부 오답 확장(`standard10_aug`), 합성 Vision 보드 렌더링(`vision_board`)
- `generate_verified_rationales.py`: 오프라인 텍스트 전용 정답 검증 풀이 생성 (Zero Logit Distillation, 100% Gold Match)
- `struct_reward.py`: 결정론적 정답 게이트 프로세스 보상 ($R = G_{\text{ans}}(1 + \alpha R_{\text{process}}) + R_{\text{format}} + R_{\text{consistency}} - R_{\text{length}}$)
- `train_sft.py`: Stage 1 진단용 Off-Policy Concise-CoT SFT (BF16 LoRA `r=32`)
- `train_structreward_grpo.py`: Stage 2/3 Base 직결 Pure On-Policy StructReward-GRPO (Vision+Merger 100% 동결, Schulman $k_3$ Reverse KL, DAPO 비대칭 클리핑)
- `eval_mmmu_pro.py`: SGLang BF16 기반 MMMU-Pro 전체 3,460문항 평가 (`official_cot`, `calibrated_cot`, `budget_cot`, `deep_verify_cot`, `--num-samples` Self-Consistency 지원)
- `test_pipeline.py`: 보상 함수·파서·인코딩·체크포인트 저장 등 36개 단위 테스트

## 재현 명령어 예시

```bash
export MMMU_PRO_ROOT=/path/to/workspace

# 1. 단위 테스트 실행
pytest -q scripts/mmmu_pro_rl/test_pipeline.py

# 2. MMMU-Pro 평가 데이터 격리 준비
python scripts/mmmu_pro_rl/setup_mmmu_pro_eval.py

# 3. Stage 2a: Base 직결 Balanced On-Policy StructReward-GRPO (우승 모델)
python scripts/mmmu_pro_rl/train_structreward_grpo.py \
  --init-model "$MMMU_PRO_ROOT/models/qwen3-vl-4b-instruct" \
  --rl-file "$MMMU_PRO_ROOT/data/prepared_pools/rl_pool.jsonl" \
  --eval-ext-dev "$MMMU_PRO_ROOT/data/prepared_pools/ext_dev.jsonl" \
  --out-dir "$MMMU_PRO_ROOT/runs/rl_structreward_v1" \
  --rounds 12 --prompts-per-round 96 --group-size 8 \
  --lr 8e-6 --kl-beta 0.02 --max-tokens 1536 --len-ramp-start 1150

# 4. SGLang BF16 기반 MMMU-Pro 전체 3,460문항 평가 (8K 예산)
python scripts/mmmu_pro_rl/eval_mmmu_pro.py \
  --model-path "$MMMU_PRO_ROOT/runs/rl_structreward_v1/final_merged" \
  --eval-dir "$MMMU_PRO_ROOT/eval_data/mmmu_pro" \
  --out-dir "$MMMU_PRO_ROOT/runs/eval_rl_v1_8k" \
  --profiles official_cot --max-tokens 8192 --presence-penalty 1.5
```
