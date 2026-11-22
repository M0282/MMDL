# MMMU-Pro 평가 (E13)

SFT 어댑터 A군·B군과 학습 전 모델을 MMMU-Pro `standard (10 options)`에서 평가한 결과.

- 결과 요약: [`summary.md`](summary.md)
- 실행 노트북: [`../../scripts/MMDL_E13_MMMU_Pro_AB.ipynb`](../../scripts/MMDL_E13_MMMU_Pro_AB.ipynb)

## 왜 Pro도 봤는가

validation·test에서 SFT 후 정확도가 떨어졌다. 그 하락이 특정 데이터셋의 성질인지, 학습 자체의 문제인지 가르려면 **같은 모델을 다른 분포에서** 한 번 더 봐야 한다. MMMU-Pro는 선택지를 10개로 늘리고 후보를 걸러낸 더 어려운 세트라서, 같은 방향의 하락이 나오면 데이터셋 탓이 아니라고 말할 수 있다.

## 설정

| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen3-VL-4B-Instruct` (rev `ebb281e`, bf16) |
| 데이터 | `MMMU/MMMU_Pro`, `standard (10 options)`, test split 1,730문항 |
| 채점 | MMMU-Pro 공식 `evaluate.py` (rev `b26bdef`) 그대로 |
| 디코딩 | greedy (`do_sample=False`), `max_new_tokens=2048` |
| 해상도 | `max_pixels=602112` |
| 프롬프트 | 형식 사양 프롬프트 (마지막 줄 `Answer: (X)`) |
| GPU | A군 Tesla T4(1,700) + NVIDIA L4(30), B군·학습 전 NVIDIA L4 |

세 모델 모두 같은 문제·프롬프트·해상도·생성 예산을 쓴다. 바꾼 것은 어댑터뿐이다.

## 재현 방법

1. Colab GPU 런타임에서 노트북을 위에서부터 실행한다. 셀 1~4가 환경·데이터·모델 준비, 셀 5가 어댑터 A/B 평가, 셀 7이 학습 전 기준선이다.
2. 어댑터는 `MyDrive/adapter/adapter_A`, `adapter_B`에 둔다.
3. 결과는 10문항 단위 `chunk_*.jsonl`로 Drive에 저장되므로 런타임이 끊기면 같은 설정으로 다시 실행하면 이어진다.
4. 출력 폴더 이름은 설정 메타의 SHA256이고 메타에 GPU 이름이 들어간다. **GPU를 바꾸면 폴더가 새로 생기므로** 이전 청크를 복사해야 이어받는다 (노트북 마지막 셀 참고).

빠른 점검은 `LIMIT`를 3으로 두고 돌린다.

## 원자료

응답 원문 jsonl은 용량 때문에 Drive에 둔다.

| 파일 | 위치 |
|---|---|
| A군·B군 전체 1,730문항 | `MMDL_E13_MMMU_Pro/c1358a2778830849/` (`adapter_A.jsonl`, `adapter_B.jsonl`) |
| 학습 전 577문항 | `MMDL_E13_MMMU_Pro/c1358a2778830849_base_step3/base.jsonl` |
| A군 T4 구간 청크 | `MMDL_E13_MMMU_Pro/1467073e950aeaab/A/` |
| 부분집합 문항 목록 | `c1358a2778830849_base_step3/subset_ids.json` |

각 행: `id`, `options`, `answer`, `response`(원문), `generated_tokens`, `hit_limit`, 채점 결과.
