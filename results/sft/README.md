# SFT 학습 결과

외부 데이터 5,890건(MMMU 중복 제거 + 정답 검증)으로 Qwen3-VL-4B를 LoRA 학습.
재현: `scripts/e9_train.py` (학습 환경 확인은 `scripts/e2_env_test.py`).

## 설정

- LoRA: r=16, alpha=32, target = q/k/v/o_proj, bf16
- gradient checkpointing, max_pixels=602112, max_len=4096
- GPU: Colab Pro NVIDIA L4 (24GB), batch 1, grad_accum 8, 1 epoch

## A군 / B군

| 군 | 학습 응답 | 설명 |
| --- | --- | --- |
| A | `Answer: (X)` (답만) | 풀이 없이 정답만 학습 (대조군) |
| B | 풀이 + `Answer: (X)` | 풀이 과정까지 학습 (본 가설) |

A·B군은 같은 문항 집합을 쓰며, 차이는 학습 응답에 풀이가 들어가는지 여부뿐이다.

## B군 학습 로그 (요약)

전체 로그: `train_B_log.txt`

- 5,888 steps (1 epoch), 140분, peak VRAM 11.4GB
- loss 1.74 → 약 0.5~0.9 (초반 급감 후 안정)
- 길이 초과로 건너뛴 문항 0개

## 산출물 (용량 때문에 저장소 미포함)

LoRA 어댑터는 공유 드라이브에 보관:
- `adapter_B/` — B군 (풀이 포함)
- `adapter_A/` — A군 (답만)

## 다음 단계

학습한 어댑터를 base 모델에 붙여 MMMU validation / test / MMMU-Pro 평가.
학습 전 조건(A 52.67% / B 49.93%)과 baseline(51.89%) 대비 개선을 측정.
