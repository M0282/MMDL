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
  run_stage0_hf.py / _vllm.py     Stage 0 조건 A·B·C 평가
  run_stage0.sh                   Stage 0 실행 래퍼
  summarize_stage0.py             Stage 0 결과 집계
  prepare_mmmu_local.py           MMMU 로컬 준비
  verify_format_spec.py           형식 사양 재채점 검증
  e2_env_test.py                  LoRA 학습 환경 테스트
  e9_train.py                     SFT 학습 (A군/B군, LoRA)

results/
  stage0/                         조건 A·B·C 전체 응답(jsonl) + 설정(meta.json) + summary.md
  format_verification/summary.md  형식 사양 재채점 검증 결과
  sft/                            SFT 학습 로그 + 설정 (어댑터 위치 안내)
```

용량이 큰 이미지(21,298장) · 학습 데이터(train_A/B.jsonl) · LoRA 어댑터는
저장소에 포함하지 않고 공유 드라이브에 보관합니다.

> 데이터 구축(E5·E6·E8) 노트북은 별도로 추가됩니다.
> 

## 파이프라인

| 단계 | 내용 | 코드 | 결과 |
| --- | --- | --- | --- |
| baseline | 답만 출력, 900문항 평가 | `scripts/run_mmmu_eval.py` | 51.89% |
| Stage 0 | 조건 A(답만)·B(풀이)·C(풀이+해상도↑)를 seed 3회씩 평가 | `scripts/run_stage0_*.py` | `results/stage0/` |
| 형식 검증 | 저장된 B 응답을 `Answer: (X)`로 재채점, 공식 점수 회복분 측정 | `scripts/verify_format_spec.py` | `results/format_verification/` |
| 데이터 구축 | 외부 데이터 풀이 + 기준 정답을 형식 사양대로 변환 (A군/B군, 5,890건) | (E8 노트북) | 공유 드라이브 |
| 학습 환경 | LoRA 학습이 Colab(L4)에서 되는지 확인 | `scripts/e2_env_test.py` | — |
| SFT | A군(답만)·B군(풀이+답) LoRA 학습 | `scripts/e9_train.py` | `results/sft/` |
| 최종 평가 | 학습 모델로 MMMU test + MMMU-Pro (예정) | — | — |

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

### SFT (진행 중)

- 학습 데이터 5,890건(A/B군 공통)으로 LoRA 학습. 학습 후 val/test/Pro 평가 예정.
- 상세: `results/sft/`

## 한계

- 조건 B는 응답의 31%가 2048토큰에서 잘려, 59.96%는 상한이 아니라 하한선.
- 외부 데이터 풀이 품질은 사람 표본 검토 미완(자동 일치 기준 선별).
- 중간 발표에서 계획한 RL(GRPO) 단계는 이번 범위에서 제외, 향후 과제로 남김.

## 참고문헌

- Qwen3-VL Technical Report. arXiv:2511.21631
- Yue et al. MMMU. CVPR 2024. arXiv:2311.16502 / MMMU-Pro. arXiv:2409.02813
- Hu et al. LoRA. ICLR 2022. arXiv:2106.09685
