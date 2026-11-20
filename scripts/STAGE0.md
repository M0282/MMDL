# 0단계 진단 실험 실행 가이드

## 파일
- `run_stage0_vllm.py` — 조건 1개 × seed 1개 실행. 결과: `results/stage0/{조건}_seed{seed}.jsonl` (정답 포함 전체 응답) + `.meta.json`
- `run_stage0.sh` — A/B/C × seed 42/43/44 = 9회 실행 후 요약까지
- `summarize_stage0.py` — 노션에 붙일 마크다운 표 생성 (`results/stage0/summary.md`)

## 1. 환경 (처음 한 번)
```bash
nvidia-smi                      # GPU 종류·빈 메모리 확인 (공용이면 다른 사람 사용량도)
cd ~/MMDL                       # 레포 클론 위치
python -m venv .venv && source .venv/bin/activate
pip install "vllm>=0.11.0" datasets pillow   # Qwen3-VL은 vLLM 0.11.0부터 지원
pip freeze > results/stage0_requirements.txt  # 재현성용으로 실제 설치 버전 기록
```
HF 다운로드가 막혀 있으면 `huggingface-cli login` 또는 `HF_HOME`을 넉넉한 디스크로 지정.

## 2. 스모크 테스트 (과목당 2문항, 조건당 1회 → 몇 분)
```bash
SMOKE=1 bash scripts/run_stage0.sh
```
확인할 것: 에러 없이 끝나는지, `results/stage0_smoke/B_seed42.jsonl`에서 응답이 "Answer: X"로 끝나는지, `truncated`가 너무 많지 않은지.

## 3. 본 실험 (tmux 안에서 → SSH 끊겨도 계속 돎)
```bash
tmux new -s stage0
source .venv/bin/activate
bash scripts/run_stage0.sh
# Ctrl+b 누르고 d → 빠져나오기 / 다시 보기: tmux attach -t stage0
```
- 끊기면 그냥 다시 실행 → 끝난 조건은 건너뜀
- GPU 메모리를 줄여야 하면: `EXTRA_ARGS="--gpu_memory_utilization 0.5" bash scripts/run_stage0.sh`
- 특정 조건만: `CONDS="B C" bash scripts/run_stage0.sh`

## 4. 결과
`results/stage0/summary.md` 내용을 노션 페이지에 붙여넣기 (노션은 마크다운 표 붙여넣기 지원).
VRAM 기록: `results/stage0/logs/nvidia_smi.csv`

## 주의 (해석할 때)
- 백엔드가 baseline(transformers)과 달라서 조건 A가 51.89%와 똑같이 안 나올 수 있음 → A/B/C끼리 비교가 핵심
- B는 프롬프트와 토큰 길이를 같이 바꿈 → B−A 차이를 둘 중 어느 쪽 효과인지 나눌 수는 없음
- CoT 응답은 마지막 "Answer:" 뒤만 공식 파서에 넘김 (풀이 중 다른 선택지를 잡는 오채점 방지)
- 채점 두 가지 기록: **수정 파서**(메인, Chem_28 유형 수정) / **공식 파서**(Assignment #1과 같은 기준)
- 발표 슬라이드 원칙 "baseline과 같은 스크립트에서 --model_path만 바꿔 재평가"
  → 제안: 0단계 결과를 '새 baseline'으로 삼고, 이후 SFT/GRPO 체크포인트도 이 스크립트로 `--model_path`만 바꿔 평가 (팀 합의 필요)
  → LoRA 체크포인트는 base에 merge해서 저장한 폴더를 `--model_path`로 넘기기
