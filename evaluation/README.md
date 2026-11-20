# Evaluation

최종 MMMU 평가 관련 안내 폴더.

평가 **코드는 `scripts/`에 있다**:
- `scripts/run_mmmu_eval.py` — MMMU validation 900문항 평가 (baseline/체크포인트 공용)
- 학습한 LoRA 어댑터를 base 모델에 붙여 평가하려면 `--model_path`에 어댑터 경로를 지정

baseline 평가 보고서는 `reports/mmmu_baseline.md` 참고.
