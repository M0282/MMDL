# 형식 사양 재채점 검증

학습 전에, **"풀이 끝을 `Answer: (X)` 형식으로 맞추기만 해도 공식 파서 점수가 얼마나 오르는지"** 를 GPU 없이 측정한 결과.

## 파일

- `summary.md` — 결과 요약. seed 42/43/44 기준 객관식 순증 평균 **+86.0건 → 전체 900문항 대비 약 +9.6%p**. 방법·원인·한계 포함.

## 핵심

MMMU 공식 파서 `parse_multi_choice_response`는 응답에서 괄호 표기 `(A)~(D)` 중 **가장 뒤에 나온 것**을 정답으로 읽는다. 풀이 중 선택지를 나열하면 `Answer: B`로 끝내도 마지막 `(D)`가 찍혀 오답 처리된다. 학습 응답을 `Answer: (X)`로 끝내도록 형식만 고정해도 이 손실을 회복할 수 있음을 보인다.

## 재현

```
python scripts/verify_format_spec.py --stage0_dir results/stage0 --cond B
```

입력은 `results/stage0/B_seed*.jsonl` (Stage 0 조건 B 응답). 모델을 다시 돌리지 않고 저장된 응답 끝에 `Answer: (기준정답)`을 붙여 공식 파서로 재채점한다.
