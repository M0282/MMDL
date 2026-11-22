# 0단계 진단 실험 결과 (MMMU validation)

- 조건 A: 답만 출력 / 512토큰 / max_pixels 1280×28² (현재 baseline 설정)
- 조건 B: 풀이 허용 / 2048토큰 / max_pixels 1280×28²
- 조건 C: 풀이 허용 / 2048토큰 / max_pixels 5120×28²
- 백엔드 transformers-batched, sampling recipe는 baseline과 동일. 정확도 = 과목별 정확도 평균(macro). 표기는 seed 평균 ± 표준편차
- 정확도 = 수정 파서(Chem_28 유형 수정 + CoT는 마지막 'Answer:' 뒤만 채점). 공식 파서 기준 = Assignment #1과 같은 채점
- 참고: Assignment #1 baseline 51.89% (transformers 백엔드, seed 42 1회)

## 1. 조건별 전체 결과

| 조건 | seed 수 | 정확도(%) | 공식 파서 기준(%) | 약점 17과목 | 강점 13과목 | 응답 잘림 | 생성 토큰 중앙값 | 실행 시간(분, 1회 평균) |
|---|---|---|---|---|---|---|---|---|
| B | 3 | 53.30 ± 2.13 | 52.74 ± 2.00 | 43.79 ± 3.13 | 65.73 ± 0.97 | 1.1% | 85 | 48.2 |

## 2. 분야별 정확도 (seed 평균)

| 분야 | B |
|---|---|
| 예술·디자인 | 60.3 |
| 비즈니스 | 55.3 |
| 과학 | 45.1 |
| 보건·의학 | 58.9 |
| 인문·사회 | 67.5 |
| 기술·공학 | 41.6 |

## 3. 과목별 정확도 (seed 평균, 약점 과목 ★)

| 과목 | B |
|---|---|
| Art | 60.0 |
| Art_Theory | 66.7 |
| Design | 80.0 |
| ★ Music | 34.4 |
| ★ Accounting | 51.1 |
| ★ Economics | 67.8 |
| ★ Finance | 52.2 |
| ★ Manage | 44.4 |
| Marketing | 61.1 |
| ★ Biology | 41.1 |
| ★ Chemistry | 41.1 |
| ★ Geography | 48.9 |
| ★ Math | 48.9 |
| ★ Physics | 45.6 |
| Basic_Medical_Science | 68.9 |
| Clinical_Medicine | 56.7 |
| ★ Diagnostics_and_Laboratory_Medicine | 30.0 |
| Pharmacy | 65.6 |
| Public_Health | 73.3 |
| History | 60.0 |
| Literature | 82.2 |
| Psychology | 64.4 |
| Sociology | 63.3 |
| Agriculture | 52.2 |
| ★ Architecture_and_Engineering | 35.6 |
| ★ Computer_Science | 47.8 |
| ★ Electronics | 33.3 |
| ★ Energy_and_Power | 43.3 |
| ★ Materials | 35.6 |
| ★ Mechanical_Engineering | 43.3 |

## 5. 오답 분석 핵심 사례 추적 (seed별 정답 수)

| 문항 | B |
|---|---|
| Chemistry_13 | 3/3 |
| Chemistry_8 | 0/3 |
| Chemistry_28 | 3/3 |
| Math_13 | 1/3 |
| Math_15 | 0/3 |
| Electronics_2 | 0/3 |
| Geography_16 | 3/3 |
