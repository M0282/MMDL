# Stage 0 진단 실험 결과

학습 전 진단 단계. 같은 900문항(MMMU validation)을 **세 조건**으로 돌린 원자료와 요약을 보관한다.

- 조건 A: 답만 출력 / 512토큰 (현재 baseline 설정)
- 조건 B: 풀이 허용 / 2048토큰
- 조건 C: 풀이 허용 / 2048토큰 + 고해상도(max_pixels 5120×28²)

## 파일

- `summary.md` — 조건별 정확도, 분야·과목별 표, 문항 단위 변화(A→B/B→C), 오답 핵심 사례 추적
- `{A,B,C}_seed{42,43,44}.jsonl` — 각 조건·시드의 **문항별 응답 원자료** (총 9개)
- `{A,B,C}_seed{42,43,44}.meta.json` — 각 실행의 설정·요약 메타

## 요약 수치 (seed 평균)

| 조건 | 수정 파서(%) | 공식 파서(%) | 응답 잘림 | 생성 토큰 중앙값 |
|---|---|---|---|---|
| A | 52.78 | 52.67 | 0.3% | 4 |
| B | 59.96 | 49.93 | 30.7% | 656 |
| C | 60.22 | 49.26 | 31.0% | 648 |

> 참고: Assignment #1 baseline 51.89% (공식 파서, seed 42 1회). 조건 B에서 수정 파서와 공식 파서 점수가 크게 벌어지는 것이 형식 사양 문제의 근거다 (`results/format_verification/` 참고).

## 재현

```
bash scripts/run_stage0.sh        # 조건 A/B/C 일괄 실행 (상세: scripts/STAGE0.md)
python scripts/summarize_stage0.py   # summary 재생성
```
