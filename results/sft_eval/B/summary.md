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
| B | 3 | 50.63 ± 1.57 | 50.37 ± 1.60 | 41.90 ± 1.28 | 62.05 ± 2.45 | 0.1% | 65 | 15.0 |

## 2. 분야별 정확도 (seed 평균)

| 분야 | B |
|---|---|
| 예술·디자인 | 60.6 |
| 비즈니스 | 54.4 |
| 과학 | 41.3 |
| 보건·의학 | 54.2 |
| 인문·사회 | 61.9 |
| 기술·공학 | 39.8 |

## 3. 과목별 정확도 (seed 평균, 약점 과목 ★)

| 과목 | B |
|---|---|
| Art | 60.0 |
| Art_Theory | 76.7 |
| Design | 72.2 |
| ★ Music | 33.3 |
| ★ Accounting | 53.3 |
| ★ Economics | 66.7 |
| ★ Finance | 47.8 |
| ★ Manage | 46.7 |
| Marketing | 57.8 |
| ★ Biology | 33.3 |
| ★ Chemistry | 36.7 |
| ★ Geography | 46.7 |
| ★ Math | 45.6 |
| ★ Physics | 44.4 |
| Basic_Medical_Science | 58.9 |
| Clinical_Medicine | 54.4 |
| ★ Diagnostics_and_Laboratory_Medicine | 27.8 |
| Pharmacy | 62.2 |
| Public_Health | 67.8 |
| History | 61.1 |
| Literature | 76.7 |
| Psychology | 53.3 |
| Sociology | 56.7 |
| Agriculture | 48.9 |
| ★ Architecture_and_Engineering | 37.8 |
| ★ Computer_Science | 41.1 |
| ★ Electronics | 40.0 |
| ★ Energy_and_Power | 38.9 |
| ★ Materials | 36.7 |
| ★ Mechanical_Engineering | 35.6 |

## 5. 오답 분석 핵심 사례 추적 (seed별 정답 수)

| 문항 | B |
|---|---|
| Chemistry_13 | 3/3 |
| Chemistry_8 | 0/3 |
| Chemistry_28 | 2/3 |
| Math_13 | 1/3 |
| Math_15 | 0/3 |
| Electronics_2 | 1/3 |
| Geography_16 | 1/3 |
