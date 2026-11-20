# Scripts

평가·진단·학습을 재현하기 위한 실행 스크립트.

## 평가 (baseline / 체크포인트 공용)

- `run_mmmu_eval.py` — MMMU validation 900문항 평가. `--model_path`로 baseline 또는 fine-tune 체크포인트 지정
- `prepare_mmmu_local.py` — MMMU 데이터 로컬 준비
- `requirements-minimal.txt` — 핵심 6개 패키지 / `requirements.txt` — Colab 런타임 전체 freeze

## Stage 0 진단

- `run_stage0.sh` — 조건 A/B/C 일괄 실행 (상세: `STAGE0.md`)
- `run_stage0_hf.py`, `run_stage0_vllm.py` — 백엔드별 실행 스크립트
- `summarize_stage0.py` — 결과 요약표 생성
- `verify_format_spec.py` — 형식 사양 재채점 검증 (GPU 불필요, 입력은 `results/stage0/B_seed*.jsonl`)

## 학습 (SFT)

- `e2_env_test.py` — LoRA 학습 환경 점검 (더미 데이터, VRAM·동작 확인용)
- `e9_train.py` — LoRA SFT 학습. A군(답만)·B군(풀이) 공용, 학습 데이터만 교체

재현 상세는 각 결과 폴더의 README 참고: `results/stage0/`, `results/format_verification/`, `results/sft/`.
