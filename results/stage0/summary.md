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
| A | 3 | 52.78 ± 0.19 | 52.67 ± 0.19 | 40.85 ± 0.11 | 68.38 ± 0.53 | 0.3% | 4 | 4.5 |
| B | 3 | 59.96 ± 0.93 | 49.93 ± 0.06 | 49.80 ± 1.60 | 73.25 ± 1.16 | 30.7% | 656 | 130.8 |
| C | 3 | 60.22 ± 1.09 | 49.26 ± 0.39 | 50.39 ± 1.93 | 73.08 ± 1.28 | 31.0% | 648 | 118.9 |

## 2. 분야별 정확도 (seed 평균)

| 분야 | A | B | C |
|---|---|---|---|
| 예술·디자인 | 64.4 | 62.2 | 62.2 |
| 비즈니스 | 46.0 | 69.1 | 68.7 |
| 과학 | 42.9 | 54.9 | 54.9 |
| 보건·의학 | 57.1 | 66.4 | 66.4 |
| 인문·사회 | 72.5 | 72.2 | 72.2 |
| 기술·공학 | 43.7 | 44.1 | 45.6 |

## 3. 과목별 정확도 (seed 평균, 약점 과목 ★)

| 과목 | A | B | C | B−A | C−B |
|---|---|---|---|---|---|
| Art | 66.7 | 64.4 | 64.4 | -2.2 | +0.0 |
| Art_Theory | 81.1 | 78.9 | 81.1 | -2.2 | +2.2 |
| Design | 77.8 | 78.9 | 77.8 | +1.1 | -1.1 |
| ★ Music | 32.2 | 26.7 | 25.6 | -5.6 | -1.1 |
| ★ Accounting | 43.3 | 57.8 | 57.8 | +14.4 | +0.0 |
| ★ Economics | 50.0 | 80.0 | 80.0 | +30.0 | +0.0 |
| ★ Finance | 31.1 | 60.0 | 60.0 | +28.9 | +0.0 |
| ★ Manage | 43.3 | 63.3 | 62.2 | +20.0 | -1.1 |
| Marketing | 62.2 | 84.4 | 83.3 | +22.2 | -1.1 |
| ★ Biology | 51.1 | 52.2 | 53.3 | +1.1 | +1.1 |
| ★ Chemistry | 26.7 | 50.0 | 48.9 | +23.3 | -1.1 |
| ★ Geography | 53.3 | 52.2 | 52.2 | -1.1 | +0.0 |
| ★ Math | 41.1 | 50.0 | 50.0 | +8.9 | +0.0 |
| ★ Physics | 42.2 | 70.0 | 70.0 | +27.8 | +0.0 |
| Basic_Medical_Science | 73.3 | 75.6 | 74.4 | +2.2 | -1.1 |
| Clinical_Medicine | 64.4 | 66.7 | 68.9 | +2.2 | +2.2 |
| ★ Diagnostics_and_Laboratory_Medicine | 33.3 | 31.1 | 33.3 | -2.2 | +2.2 |
| Pharmacy | 62.2 | 85.6 | 83.3 | +23.3 | -2.2 |
| Public_Health | 52.2 | 73.3 | 72.2 | +21.1 | -1.1 |
| History | 68.9 | 70.0 | 71.1 | +1.1 | +1.1 |
| Literature | 80.0 | 81.1 | 81.1 | +1.1 | +0.0 |
| Psychology | 73.3 | 74.4 | 73.3 | +1.1 | -1.1 |
| Sociology | 67.8 | 63.3 | 63.3 | -4.4 | +0.0 |
| Agriculture | 58.9 | 55.6 | 55.6 | -3.3 | +0.0 |
| ★ Architecture_and_Engineering | 42.2 | 32.2 | 31.1 | -10.0 | -1.1 |
| ★ Computer_Science | 47.8 | 63.3 | 66.7 | +15.6 | +3.3 |
| ★ Electronics | 45.6 | 43.3 | 44.4 | -2.2 | +1.1 |
| ★ Energy_and_Power | 35.6 | 42.2 | 46.7 | +6.7 | +4.4 |
| ★ Materials | 35.6 | 34.4 | 33.3 | -1.1 | -1.1 |
| ★ Mechanical_Engineering | 40.0 | 37.8 | 41.1 | -2.2 | +3.3 |

## 4. 문항 단위 변화 (seed 과반 정답 기준)

| 비교 | 틀림→맞음 | 맞음→틀림 | 순변화 |
|---|---|---|---|
| A→B | 161 | 113 | +48 |
| B→C | 14 | 11 | +3 |

<details><summary>A→B 문항 목록</summary>

- 틀림→맞음: validation_Accounting_11, validation_Accounting_13, validation_Accounting_15, validation_Accounting_16, validation_Accounting_18, validation_Accounting_24, validation_Accounting_26, validation_Accounting_27, validation_Accounting_8, validation_Architecture_and_Engineering_22, validation_Architecture_and_Engineering_5, validation_Architecture_and_Engineering_7, validation_Architecture_and_Engineering_8, validation_Art_19, validation_Art_Theory_17, validation_Art_Theory_8, validation_Basic_Medical_Science_10, validation_Basic_Medical_Science_21, validation_Biology_10, validation_Biology_12, validation_Biology_2, validation_Biology_22, validation_Biology_25, validation_Chemistry_1, validation_Chemistry_16, validation_Chemistry_19, validation_Chemistry_2, validation_Chemistry_24, validation_Chemistry_25, validation_Chemistry_29, validation_Chemistry_6, validation_Clinical_Medicine_25, validation_Clinical_Medicine_28, validation_Computer_Science_1, validation_Computer_Science_10, validation_Computer_Science_12, validation_Computer_Science_13, validation_Computer_Science_14, validation_Computer_Science_19, validation_Computer_Science_23, validation_Computer_Science_24, validation_Computer_Science_25, validation_Computer_Science_4, validation_Computer_Science_7, validation_Design_12, validation_Design_15, validation_Diagnostics_and_Laboratory_Medicine_12, validation_Diagnostics_and_Laboratory_Medicine_26, validation_Diagnostics_and_Laboratory_Medicine_3, validation_Economics_11, validation_Economics_12, validation_Economics_13, validation_Economics_15, validation_Economics_17, validation_Economics_29, validation_Economics_3, validation_Economics_5, validation_Economics_6, validation_Economics_8, validation_Electronics_1, validation_Electronics_12, validation_Electronics_21, validation_Electronics_28, validation_Energy_and_Power_13, validation_Energy_and_Power_17, validation_Energy_and_Power_22, validation_Energy_and_Power_23, validation_Energy_and_Power_27, validation_Energy_and_Power_28, validation_Energy_and_Power_29, validation_Energy_and_Power_30, validation_Finance_19, validation_Finance_22, validation_Finance_24, validation_Finance_25, validation_Finance_3, validation_Finance_4, validation_Finance_5, validation_Finance_7, validation_Geography_13, validation_Geography_15, validation_Geography_2, validation_History_11, validation_History_13, validation_History_25, validation_Literature_15, validation_Literature_24, validation_Manage_12, validation_Manage_13, validation_Manage_17, validation_Manage_2, validation_Manage_20, validation_Manage_21, validation_Manage_25, validation_Manage_30, validation_Manage_8, validation_Marketing_1, validation_Marketing_14, validation_Marketing_22, validation_Marketing_24, validation_Marketing_25, validation_Marketing_3, validation_Marketing_5, validation_Marketing_7, validation_Materials_10, validation_Materials_14, validation_Materials_28, validation_Materials_30, validation_Materials_4, validation_Materials_7, validation_Math_14, validation_Math_23, validation_Math_27, validation_Math_29, validation_Math_3, validation_Math_7, validation_Math_8, validation_Mechanical_Engineering_1, validation_Mechanical_Engineering_10, validation_Mechanical_Engineering_17, validation_Mechanical_Engineering_20, validation_Mechanical_Engineering_24, validation_Music_10, validation_Music_16, validation_Music_29, validation_Music_3, validation_Music_30, validation_Music_7, validation_Pharmacy_1, validation_Pharmacy_19, validation_Pharmacy_22, validation_Pharmacy_26, validation_Pharmacy_27, validation_Pharmacy_28, validation_Pharmacy_30, validation_Pharmacy_5, validation_Physics_17, validation_Physics_2, validation_Physics_22, validation_Physics_23, validation_Physics_25, validation_Physics_28, validation_Physics_29, validation_Physics_4, validation_Physics_5, validation_Physics_7, validation_Physics_9, validation_Psychology_20, validation_Public_Health_14, validation_Public_Health_15, validation_Public_Health_17, validation_Public_Health_2, validation_Public_Health_20, validation_Public_Health_21, validation_Public_Health_23, validation_Public_Health_25, validation_Public_Health_30, validation_Public_Health_8, validation_Sociology_1, validation_Sociology_23, validation_Sociology_29
- 맞음→틀림: validation_Accounting_1, validation_Accounting_17, validation_Accounting_19, validation_Accounting_2, validation_Accounting_5, validation_Accounting_6, validation_Accounting_7, validation_Accounting_9, validation_Agriculture_4, validation_Agriculture_7, validation_Architecture_and_Engineering_12, validation_Architecture_and_Engineering_16, validation_Architecture_and_Engineering_18, validation_Architecture_and_Engineering_19, validation_Architecture_and_Engineering_20, validation_Architecture_and_Engineering_21, validation_Architecture_and_Engineering_26, validation_Architecture_and_Engineering_28, validation_Architecture_and_Engineering_6, validation_Architecture_and_Engineering_9, validation_Art_1, validation_Art_26, validation_Art_Theory_14, validation_Art_Theory_16, validation_Art_Theory_20, validation_Basic_Medical_Science_17, validation_Basic_Medical_Science_22, validation_Biology_13, validation_Biology_24, validation_Biology_28, validation_Biology_29, validation_Biology_4, validation_Biology_6, validation_Chemistry_26, validation_Chemistry_28, validation_Chemistry_3, validation_Clinical_Medicine_16, validation_Clinical_Medicine_20, validation_Computer_Science_11, validation_Computer_Science_15, validation_Computer_Science_16, validation_Computer_Science_20, validation_Computer_Science_22, validation_Computer_Science_28, validation_Design_20, validation_Diagnostics_and_Laboratory_Medicine_22, validation_Diagnostics_and_Laboratory_Medicine_23, validation_Diagnostics_and_Laboratory_Medicine_24, validation_Diagnostics_and_Laboratory_Medicine_28, validation_Economics_23, validation_Electronics_13, validation_Electronics_14, validation_Electronics_18, validation_Electronics_27, validation_Energy_and_Power_10, validation_Energy_and_Power_14, validation_Energy_and_Power_5, validation_Energy_and_Power_7, validation_Energy_and_Power_8, validation_Energy_and_Power_9, validation_Geography_27, validation_Geography_30, validation_Geography_5, validation_Geography_6, validation_History_12, validation_History_30, validation_History_7, validation_History_9, validation_Literature_14, validation_Manage_18, validation_Manage_26, validation_Manage_4, validation_Marketing_18, validation_Materials_1, validation_Materials_16, validation_Materials_22, validation_Materials_24, validation_Materials_25, validation_Materials_26, validation_Materials_27, validation_Materials_9, validation_Math_12, validation_Math_19, validation_Math_25, validation_Math_5, validation_Mechanical_Engineering_18, validation_Mechanical_Engineering_19, validation_Mechanical_Engineering_21, validation_Mechanical_Engineering_23, validation_Mechanical_Engineering_26, validation_Mechanical_Engineering_28, validation_Mechanical_Engineering_6, validation_Mechanical_Engineering_9, validation_Music_13, validation_Music_17, validation_Music_19, validation_Music_23, validation_Music_26, validation_Music_27, validation_Music_28, validation_Music_6, validation_Pharmacy_24, validation_Physics_15, validation_Physics_21, validation_Psychology_26, validation_Public_Health_11, validation_Public_Health_16, validation_Public_Health_28, validation_Public_Health_7, validation_Sociology_10, validation_Sociology_13, validation_Sociology_3, validation_Sociology_5
</details>

<details><summary>B→C 문항 목록</summary>

- 틀림→맞음: validation_Art_Theory_12, validation_Biology_13, validation_Biology_28, validation_Chemistry_26, validation_Chemistry_9, validation_Computer_Science_15, validation_Diagnostics_and_Laboratory_Medicine_6, validation_Energy_and_Power_1, validation_Energy_and_Power_14, validation_Energy_and_Power_4, validation_History_12, validation_Math_20, validation_Mechanical_Engineering_19, validation_Mechanical_Engineering_6
- 맞음→틀림: validation_Biology_22, validation_Chemistry_29, validation_Computer_Science_21, validation_Design_8, validation_Materials_15, validation_Materials_28, validation_Math_23, validation_Math_3, validation_Music_16, validation_Music_29, validation_Pharmacy_4
</details>

## 5. 오답 분석 핵심 사례 추적 (seed별 정답 수)

| 문항 | A | B | C |
|---|---|---|---|
| Chemistry_13 | 0/3 | 1/3 | 1/3 |
| Chemistry_8 | 2/3 | 3/3 | 3/3 |
| Chemistry_28 | 3/3 | 1/3 | 1/3 |
| Math_13 | 0/3 | 1/3 | 1/3 |
| Math_15 | 0/3 | 0/3 | 0/3 |
| Electronics_2 | 0/3 | 0/3 | 0/3 |
| Geography_16 | 2/3 | 3/3 | 3/3 |
