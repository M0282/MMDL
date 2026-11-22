# Results

실험 결과와 요약 통계를 보관하는 폴더. 모델 가중치나 대용량 Hugging Face 캐시는 커밋하지 않는다.

## 하위 폴더

| 폴더 | 내용 |
|---|---|
| `stage0/` | Stage 0 진단 실험 (조건 A/B/C) 원자료 + 요약 |
| `format_verification/` | 형식 사양 재채점 검증 (GPU 없이, +9.6%p 상한 근거) |
| `sft/` | LoRA SFT 학습 결과 (A군·B군 학습 로그, 설정) |
| `sft_eval/` | SFT 어댑터 A/B군 MMMU validation 평가 (seed 3회) |
| `mmmu_pro/` | SFT 어댑터 A/B군 + 학습 전 MMMU-Pro 평가 (E13) |

## 파일

- `mmmu_results.json` — baseline 평가 원자료 (`reports/mmmu_baseline.md`의 근거)
