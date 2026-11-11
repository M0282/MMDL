# 오답 문제 검색 뷰어 (코랩 셀에 통째로 붙여넣기)
# 사전 준비: 구글 드라이브 마운트 (from google.colab import drive; drive.mount('/content/drive'))
import json
from datasets import load_dataset
from IPython.display import display, clear_output
import ipywidgets as widgets

REV = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
MY_SUBJECTS = ["Materials", "Music", "Accounting", "Manage", "Physics"]  # 내 과목 (바꿔도 됨)
RESULT_JSON = "/content/drive/MyDrive/mmmu_results.json"  # 결과 json 경로

fails = {f["id"]: f for f in json.load(open(RESULT_JSON))["failure_cases"]}
rows = {}
for s in MY_SUBJECTS:
    ds = load_dataset("MMMU/MMMU", s, split="validation", revision=REV)
    for r in ds:
        if r["id"] in fails:
            rows[r["id"]] = r
print(len(rows), "개 오답 문제 로드 완료")


def find(q, rows):
    q = q.strip().lower()
    if not q:
        return []
    exact = [k for k in rows if k.lower() in (q, "validation_" + q)]
    if exact:
        return exact
    return [k for k, r in rows.items() if q in k.lower() or q in r["question"].lower()]


def show_one(qid):
    r, f = rows[qid], fails[qid]
    print("=" * 70)
    print(qid)
    print("질문:", r["question"])
    print("선택지:", r["options"])
    print("정답:", f["ground_truth"], "| 모델 답:", f["model_response"])
    for i in range(1, 8):
        img = r.get(f"image_{i}")
        if img is not None:
            display(img)
    if r.get("explanation"):
        print("해설:", r["explanation"])


def on_search(change):
    with out:
        clear_output()
        hits = find(change["new"], rows)
        if not hits:
            print("검색 결과 없음")
        elif len(hits) > 3:
            print(f"{len(hits)}건 검색됨. 더 구체적으로 입력하세요:")
            for k in hits:
                print(" -", k, "|", rows[k]["question"][:60].replace("\n", " "))
        else:
            for k in hits:
                show_one(k)


box = widgets.Text(placeholder="id 또는 문제 속 단어 (예: Accounting_22, Z-score)",
                   description="검색:", continuous_update=False,
                   layout=widgets.Layout(width="520px"))
out = widgets.Output()
box.observe(on_search, names="value")
display(box, out)
