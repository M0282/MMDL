# MMMU test 평가 (E13)

학습 전 모델과 SFT 어댑터 A군·B군을 MMMU **test split**(10,500문항)에서 평가한 결과.
MMMU test 정답이 2026-02-12에 공개돼 로컬 채점이 가능했다.

- 결과 요약: [`summary.md`](summary.md)
- 실행 스크립트: [`../../scripts/run_stage0_hf.py`](../../scripts/run_stage0_hf.py) (`--split test`)
- 요약 스크립트: [`../../scripts/summarize_test_eval.py`](../../scripts/summarize_test_eval.py)

## 폴더

| 폴더 | 모델 | 조건 | 문항 수 |
|---|---|---|---|
| `base/` | 학습 전 (어댑터 없음) | A (답만, 512토큰) | 10,500 |
| `base_B_sub50/` | 학습 전 (어댑터 없음) | B (풀이, 2048토큰) | 1,500 (과목당 50) |
| `adapter_A/` | SFT A군 | A (답만, 512토큰) | 10,500 |
| `adapter_B/` | SFT B군 | B (풀이, 2048토큰) | 10,500 |

각 폴더에 `{조건}_seed42.jsonl`(문항별 응답·파싱·채점)과 `.meta.json`(설정·GPU·정확도)이 있다.
학습 전 조건 B는 문항당 생성이 길어(중앙값 615토큰, 잘림 30%) 시간 제약으로 과목당 50문항만 돌렸다.
그래서 네 run을 직접 비교할 때는 같은 1,500문항 부분집합 기준을 쓴다 (`summary.md` 2번 표).

## 설정

Stage 0과 같다. 바꾼 것은 split과 어댑터뿐이다.

| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen3-VL-4B-Instruct` (rev `ebb281e`, bf16) |
| 데이터 | `MMMU/MMMU` test (rev `98e6ac0`) |
| 샘플링 | temperature 0.7, top_p 0.8, top_k 20, presence penalty 1.5, seed 42 1회 |
| 해상도 | max_pixels 1,003,520 |
| 백엔드 | transformers 5.2.0, batched (bs 48) |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q |

## 재현

```bash
# 학습 전 조건 A
python scripts/run_stage0_hf.py --condition A --split test --seed 42
# SFT 어댑터 (어댑터는 공유 드라이브 adapter_A / adapter_B)
python scripts/run_stage0_hf.py --condition B --split test --seed 42 --adapter_path adapters/adapter_B
# 요약표
python scripts/summarize_test_eval.py --in_dir results/test_eval --out results/test_eval/summary.md
```

옵션 전체는 [`../../scripts/STAGE0.md`](../../scripts/STAGE0.md) 참고.
