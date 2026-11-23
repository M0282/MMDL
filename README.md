# 2026 MMDL — Qwen3-VL-4B MMMU 성능 개선 (8조)

Qwen3-VL-4B-Instruct를 파인튜닝해 MMMU / MMMU-Pro 점수를 baseline보다 올리고,
오른(또는 안 오른) 원인을 분석하는 프로젝트입니다.

## 프로젝트 개요

- **목표:** baseline(답만 출력) 대비 성능 개선 + 원인 분석
- **접근:** 풀이를 허용하면 모델이 더 잘 풀지만, 공식 파서가 그 풀이의 답을 못 읽어 점수가 깎인다.
  → 풀이를 쓰되 **공식 파서가 읽을 수 있는 형식(`Answer: (X)`)으로 끝내도록 SFT**로 학습.

## 고정 평가 스펙

- Model: `Qwen/Qwen3-VL-4B-Instruct`
- Model revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- dtype: BF16 / repository default, no quantization
- Dataset: `MMMU/MMMU`
- Dataset revision: `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`
- Split: `validation` — 30 subject configs × 30 samples = 900 samples
- baseline: **51.89%** (467/900, 공식 파서 기준)
- 제약: MMMU·MMMU-Pro 문항은 학습 데이터로 사용하지 않음 (leakage 방지)

## 저장소 구조

```
README.md
SUBMISSION_TEMPLATE.md            제출 템플릿
docs/assignment_guidance.md       과제 지침 사본
reports/mmmu_baseline.md          baseline 보고서

scripts/
  run_mmmu_eval.py                baseline 평가 (51.89%)
  run_mmmu_eval_v2.py             baseline 재실행용 (전체 응답 저장, seed 흔들림 측정)
  run_stage0_hf.py / _vllm.py     Stage 0 조건 A·B·C 평가, SFT 어댑터 평가 (--adapter_path, --split test)
  run_stage0.sh                   Stage 0 실행 래퍼
  summarize_stage0.py             Stage 0 / SFT validation 결과 집계
  summarize_test_eval.py          MMMU test 결과 집계
  prepare_mmmu_local.py           MMMU 로컬 준비
  verify_format_spec.py           형식 사양 재채점 검증
  e2_env_test.py                  LoRA 학습 환경 테스트
  e9_train.py                     SFT 학습 (A군/B군, LoRA)
  MMDL_E5_E6_Colab.ipynb          외부 데이터 수집·정제 + MMMU 중복 검사 (E5·E6)
  MMDL_E8_External_AB_(1).ipynb   외부 데이터 A군/B군 학습셋 구성 (E8)
  MMDL_E13_MMMU_Pro_AB.ipynb      MMMU-Pro 평가 (E13)

results/
  mmmu_results.json               baseline 원자료
  stage0/                         조건 A·B·C 전체 응답(jsonl) + 설정(meta.json) + summary.md
  format_verification/summary.md  형식 사양 재채점 검증 결과
  sft/                            SFT 학습 로그 (lr 1e-4, lr 2e-5) + 설정 (어댑터 위치 안내)
  sft_eval/                       SFT 어댑터 MMMU validation 평가 (A/B, A_lr2e-5/B_lr2e-5, seed 3회)
  test_eval/                      MMMU test 평가 (학습 전 A·B, SFT A·B)
  mmmu_pro/                       MMMU-Pro 평가 (학습 전, SFT A·B)
```

용량이 큰 이미지(21,298장) · 학습 데이터(train_A/B.jsonl) · LoRA 어댑터는
저장소에 포함하지 않고 공유 드라이브에 보관합니다.

## 파이프라인

| 단계 | 내용 | 코드 | 결과 |
| --- | --- | --- | --- |
| baseline | 답만 출력, 900문항 평가 | `scripts/run_mmmu_eval.py` | 51.89% |
| Stage 0 | 조건 A(답만)·B(풀이)·C(풀이+해상도↑)를 seed 3회씩 평가 | `scripts/run_stage0_*.py` | `results/stage0/` |
| 형식 검증 | 저장된 B 응답을 `Answer: (X)`로 재채점, 공식 점수 회복분 측정 | `scripts/verify_format_spec.py` | `results/format_verification/` |
| 데이터 점검·중복 검사 | 외부 데이터 점검(E5), MMMU 문항과 겹치는 것 제거(E6) | `scripts/MMDL_E5_E6_Colab.ipynb` | 공유 드라이브 |
| 데이터 구축 | 외부 데이터 풀이 + 기준 정답을 형식 사양대로 변환 (A군/B군, 5,890건) | `scripts/MMDL_E8_External_AB_(1).ipynb` | 공유 드라이브 |
| 학습 환경 | LoRA 학습이 Colab(L4)에서 되는지 확인 | `scripts/e2_env_test.py` | — |
| SFT | A군(답만)·B군(풀이+답) LoRA 학습 (lr 1e-4, 재학습 lr 2e-5) | `scripts/e9_train.py` | `results/sft/` |
| SFT 평가 | 어댑터를 붙여 MMMU validation, seed 3회 | `scripts/run_stage0_hf.py --adapter_path` | `results/sft_eval/` |
| 최종 평가 | MMMU test (10,500) + MMMU-Pro (1,730) | `scripts/run_stage0_hf.py --split test`, `scripts/MMDL_E13_MMMU_Pro_AB.ipynb` | `results/test_eval/`, `results/mmmu_pro/` |

## 재현 방법

```bash
pip install -U transformers accelerate peft datasets pillow bitsandbytes
```

```bash
# Stage 0 평가 + 집계
python scripts/run_stage0_hf.py          # 옵션은 scripts/STAGE0.md 참고
python scripts/summarize_stage0.py results/stage0

# 형식 사양 재채점 검증 (GPU 불필요)
python scripts/verify_format_spec.py --stage0_dir results/stage0 --cond B

# SFT 학습 (GPU 필요, L4로 확인됨)
python scripts/e9_train.py --train <train_B.jsonl> --image_root <이미지폴더> \
  --out adapter_B --tag B --epochs 1
# A군은 train_A.jsonl / --out adapter_A / --tag A
# 재학습은 --lr 2e-5

# SFT 어댑터 평가 (validation / test)
python scripts/run_stage0_hf.py --condition B --seed 42 --adapter_path adapters/adapter_B
python scripts/run_stage0_hf.py --condition B --seed 42 --adapter_path adapters/adapter_B --split test
python scripts/summarize_test_eval.py --in_dir results/test_eval --out results/test_eval/summary.md

# MMMU-Pro: Colab에서 scripts/MMDL_E13_MMMU_Pro_AB.ipynb 실행 (results/mmmu_pro/README.md 참고)
```

## 결과 요약

### Stage 0 (조건별, seed 3회 평균)

| 조건 | 공식 파서 | 형식 보정 파서 | 잘림 |
| --- | --- | --- | --- |
| A (답만) | 52.67 ± 0.19 | 52.78 ± 0.19 | 0.3% |
| B (풀이) | 49.93 ± 0.06 | 59.96 ± 0.93 | 31% |
| C (풀이+해상도↑) | 49.26 ± 0.39 | 60.22 ± 1.09 | 31% |

- 풀이 허용: +7.18%p (보정 기준). 해상도 4배: 효과 없음 (+0.26%p).
- 같은 응답인데 공식 파서로 채점하면 −2.74%p로 뒤집힘 → 파서 문제.

### 형식 사양 재채점 검증

- 저장된 B 응답 끝에 `Answer: (기준정답)`을 붙여 공식 파서로 재채점: 객관식 순증 **+86.0건**
- 전체 900문항 기준 **약 +9.6%p** (공식 49.9% → 약 59.5%)
- 상세: `results/format_verification/summary.md`

### SFT

- 학습 데이터 5,890건(A/B군 공통)으로 LoRA 학습 완료 (L4, 1 epoch).
  - A군(답만): 67분, loss 4.14 → 약0.1 / B군(풀이): 140분, loss 1.74 → 약0.7
- 상세: `results/sft/` (학습 로그 포함)

### SFT 후 평가 (공식 파서, 같은 조건끼리 비교)

| 데이터 | 학습 전 A | SFT A군 | 학습 전 B | SFT B군 | 상세 |
| --- | --- | --- | --- | --- | --- |
| MMMU validation (lr 1e-4, seed 3회) | 52.67 ± 0.19 | 44.19 ± 0.61 | 49.93 ± 0.06 | 50.37 ± 1.60 | `results/sft_eval/A`, `B` |
| MMMU validation (lr 2e-5, seed 3회) | 52.67 ± 0.19 | 47.56 ± 1.64 | 49.93 ± 0.06 | **52.74 ± 2.00** | `results/sft_eval/A_lr2e-5`, `B_lr2e-5` |
| MMMU test (같은 1,500문항) | 51.40 | 42.73 | 47.73 | 47.00 | `results/test_eval/` |
| MMMU-Pro (같은 577문항)* | 42.63 | 27.21 | 42.63 | 36.22 | `results/mmmu_pro/` |

\* MMMU-Pro는 세 모델 모두 형식 사양 프롬프트 하나로 평가해서 학습 전 기준선이 하나다 (greedy, 공식 `evaluate.py`).

- A군(답만 학습)은 세 벤치마크 모두에서 학습 전보다 떨어진다.
- B군(풀이 학습)은 A군보다 일관되게 높다 (test +4.27%p, Pro +9.01%p).
- B군은 lr 2e-5 재학습에서 학습 전 조건 B보다 +2.81%p (seed 흔들림 ±2.0 수준이라 추가 확인 필요).
- 형식 사양은 학습됨: 조건 B 잘림 30~36% → SFT 후 0~1.6%.

## MMMU-Pro Zero-Leakage On-Policy RL (StructReward-GRPO) 확장

Off-Policy SFT에서 확인된 치명적 망각(Catastrophic Forgetting) 문제를 해결하기 위해, **56,860 QA (30과목 · 183세부분야) Zero-Leakage 외부 데이터셋**과 **Base 직결 Pure On-Policy StructReward-GRPO RL**을 구축하여 **MMMU-Pro 전체 3,460문항 (`Standard10` 1,730 + `Vision` 1,730)**에서 순수 가중치 성능 향상을 검증했습니다.

- **핵심 방법**: Vision Encoder + `visual.merger` 100% 동결, Language-only LoRA (`r=32`), Base 직결 Schulman $k_3$ Reverse KL ($\beta=0.02$), 정답 게이트 결정론적 프로세스 보상(`StructReward`), SGLang BF16 고속 롤아웃 및 평가
- **주요 성과 (`official_cot` 수정 없음, 전체 3,460문항)**:
  - **2K 예산 (`max_tokens=2048`)**: Base `44.62%` → Stage 1 SFT `41.99%` (-2.63%p) → **Stage 2a RL v1 `46.65%` (+2.03%p vs Base, +4.66%p vs SFT)**
  - **8K 예산 (`max_tokens=8192`)**: Base 8K `49.37%` → **★ Stage 2a RL v1 8K `51.53%` (`Standard10 53.12%` / `Vision 49.94%`, +2.16%p vs 8K Base / +6.91%p vs 2K Base)**, 8K 잘림률 절반 이하(`18.4% → 8.4%`), 평균 토큰 `-28%` 절감
  - **8K Test-Time Scaling (`maj@5`)**: RL v1 8K `maj@5` **`52.63%`** (`Standard10 52.89%` / **`Vision 52.37%`**, **+8.01%p vs 2K Base / +3.26%p vs 8K Base**, `pass@5` 상한 `65.2% / 69.3%`)
- **세부 파일**:
  - 코드 및 단위 테스트(36개): [`scripts/mmmu_pro_rl/`](scripts/mmmu_pro_rl/README.md)
  - 전체 평가 JSON 및 RL 학습 로그: [`results/mmmu_pro_rl/`](results/mmmu_pro_rl/README.md)
  - Zero-Leakage 데이터셋 큐레이션 문서: [`docs/mmmu_pro_dataset_curation_ko.md`](docs/mmmu_pro_dataset_curation_ko.md)
  - 상세 실험 보고서 및 차트 5종: [`reports/mmmu_pro_rl_report_ko.md`](reports/mmmu_pro_rl_report_ko.md)

## 한계

- 조건 B는 응답의 31%가 2048토큰에서 잘려, 59.96%는 상한이 아니라 하한선.
- 외부 데이터 풀이 품질은 사람 표본 검토 미완(자동 일치 기준 선별).
- 중간 발표에서 계획한 RL(GRPO) 단계는 이번 범위에서 제외, 향후 과제로 남김.

## 참고문헌

- Qwen3-VL Technical Report. arXiv:2511.21631
- Yue et al. MMMU. CVPR 2024. arXiv:2311.16502 / MMMU-Pro. arXiv:2409.02813
- Hu et al. LoRA. ICLR 2022. arXiv:2106.09685
- Shao et al. DeepSeekMath (GRPO). arXiv:2402.03300
- Yu et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476
- Li et al. StructReward: Efficient Structured Process Rewards for Self-Correcting Multimodal Reasoning. 2026
- Zheng et al. SGLang: Efficient Execution of Structured Language Model Programs. NeurIPS 2024. arXiv:2312.07104
