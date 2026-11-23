# Scripts

평가·진단·학습을 재현하기 위한 실행 스크립트.

## 평가 (baseline / 체크포인트 공용)

- `run_mmmu_eval_v2.py` — baseline 재실행용 개정판. 900문항 전체 응답·잘림·fallback 저장, seed 흔들림(A_seed42/43/44) 측정에 사용
- `run_mmmu_eval.py` — MMMU validation 900문항 평가. `--model_path`로 baseline 또는 fine-tune 체크포인트, `--adapter_path`로 LoRA 어댑터 지정
- `prepare_mmmu_local.py` — MMMU 데이터 로컬 준비
- `requirements-minimal.txt` — 핵심 6개 패키지 / `requirements.txt` — Colab 런타임 전체 freeze

## Stage 0 진단

- `run_stage0.sh` — 조건 A/B/C 일괄 실행 (상세: `STAGE0.md`)
- `run_stage0_hf.py`, `run_stage0_vllm.py` — 백엔드별 실행 스크립트
- `run_stage0_hf.py`는 SFT 어댑터 평가에도 쓴다: `--adapter_path`로 어댑터, `--split test`로 MMMU test
- `summarize_stage0.py` — 결과 요약표 생성 (Stage 0, SFT validation)
- `summarize_test_eval.py` — MMMU test 결과 요약표 생성 (`results/test_eval/summary.md`)
- `verify_format_spec.py` — 형식 사양 재채점 검증 (GPU 불필요, 입력은 `results/stage0/B_seed*.jsonl`)

## 학습 (SFT)

- `e2_env_test.py` — LoRA 학습 환경 점검 (더미 데이터, VRAM·동작 확인용)
- `e9_train.py` — LoRA SFT 학습. A군(답만)·B군(풀이) 공용, 학습 데이터만 교체. 재학습은 `--lr 2e-5`

## Colab 노트북

- `MMDL_E5_E6_Colab.ipynb` — 외부 데이터 수집·정제 및 MMMU 중복 검사 (E5·E6)
- `MMDL_E8_External_AB_(1).ipynb` — 외부 데이터 A군/B군 학습셋 구성 (E8)
- `MMDL_E13_MMMU_Pro_AB.ipynb` — MMMU-Pro 평가. 어댑터 A/B군 + 학습 전 기준선 (E13)

재현 상세는 각 결과 폴더의 README 참고: `results/stage0/`, `results/format_verification/`, `results/sft/`, `results/sft_eval/`, `results/test_eval/`, `results/mmmu_pro/`.
