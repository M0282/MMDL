#!/usr/bin/env bash
# 0단계 진단 실험 전체 실행: 조건 A/B/C × seed 42/43/44 = 9회
# 사용: bash scripts/run_stage0.sh            (전체)
#       SMOKE=1 bash scripts/run_stage0.sh     (과목당 2문항, 조건별 seed 1개 → 설치 확인용)
# 끊겨도 이미 끝난 실행(.meta.json 있음)은 건너뛰니 그냥 다시 실행하면 됨.
set -u
cd "$(dirname "$0")/.."
OUT=results/stage0
mkdir -p "$OUT/logs"

if [ "${SMOKE:-0}" = "1" ]; then
  OUT=results/stage0_smoke; mkdir -p "$OUT/logs"
  SEEDS="42"; LIMIT="--limit_per_subject 2"
else
  SEEDS="42 43 44"; LIMIT=""
fi
CONDS="${CONDS:-A B C}"
BACKEND="${BACKEND:-hf}"   # hf = transformers 배치 (기본), vllm = run_stage0_vllm.py
SCRIPT=scripts/run_stage0_${BACKEND}.py
# 로컬 HF 캐시에서 모델·데이터를 읽기만 함 (다운로드 안 함). 경로는 환경에 맞게 바꾸기
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hub}"
export HF_HUB_OFFLINE=1
export MMMU_DIR="${MMMU_DIR:-$HOME/mmmu_98e6ac}"   # prepare_mmmu_local.py로 만든 폴더
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/mmdl_datasets}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# VRAM 기록 (5초 간격) — 끝나면 자동 종료
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
  --format=csv -l 5 > "$OUT/logs/nvidia_smi.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null' EXIT

for c in $CONDS; do
  for s in $SEEDS; do
    name="${c}_seed${s}"
    if [ -f "$OUT/$name.meta.json" ]; then echo "[skip] $name 이미 완료"; continue; fi
    echo "=== $name 시작 $(date '+%F %T') ==="
    python $SCRIPT --condition "$c" --seed "$s" --out_dir "$OUT" $LIMIT \
      ${EXTRA_ARGS:-} 2>&1 | tee "$OUT/logs/$name.log"
  done
done

python scripts/summarize_stage0.py --in_dir "$OUT" --out "$OUT/summary.md"
echo "완료: $OUT/summary.md"
