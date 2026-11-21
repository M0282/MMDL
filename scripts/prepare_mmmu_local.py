#!/usr/bin/env python
"""
MMMU validation/test를 baseline과 같은 revision(98e6ac0c)으로 로컬 폴더에 구성한다.

로컬 HF 캐시(/data/hub)에는 다른 revision(876ce5cb, 2026-10-05)만 있음.
HF API로 비교한 결과 validation 30개 파일 중 Finance만 내용이 다르고 29개는 동일 (2026-11-07 확인).
→ 29과목: 공용 캐시 파일을 심볼릭 링크(복사 없이 가리키기)
→ Finance: 98e6ac0c 버전을 허브에서 새로 받음
결과: ~/mmmu_98e6ac/<과목>/validation-*.parquet
"""
import glob
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

REV = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
CACHED_REV = "876ce5cb130f7f7e290ce4d9984357737d4db5cf"
CACHE = "/data/hub/datasets--MMMU--MMMU/snapshots/" + CACHED_REV
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("out", nargs="?", default="~/mmmu_98e6ac")
_ap.add_argument("--split", default="validation", choices=["validation", "test", "dev"])
_ap.add_argument("--check_only", action="store_true", help="다른 파일 개수만 확인하고 받지 않음")
_args = _ap.parse_args()
OUT = os.path.expanduser(_args.out)
SPLIT = _args.split

api = HfApi()


def oids(rev):
    return {f.path: (f.lfs.sha256 if getattr(f, "lfs", None) else f.blob_id)
            for f in api.list_repo_tree("MMMU/MMMU", repo_type="dataset", revision=rev, recursive=True)
            if f.path.endswith(".parquet") and f"/{SPLIT}-" in f.path}


want, have = oids(REV), oids(CACHED_REV)
diff = [p for p in want if have.get(p) != want[p]]
print(f"[{SPLIT}] 파일 {len(want)}개 중 캐시 버전과 내용이 다른 파일 {len(diff)}개: {diff}")
if _args.check_only:
    sys.exit(0)
os.makedirs(OUT, exist_ok=True)
n_link = n_dl = 0
for path, oid in sorted(want.items()):
    dst = os.path.join(OUT, path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src = os.path.join(CACHE, path)
    if have.get(path) == oid and os.path.exists(src):
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(src), dst)
        n_link += 1
    else:
        print(f"[받는 중] {path} ({REV[:8]} 버전)")
        hf_hub_download("MMMU/MMMU", path, repo_type="dataset", revision=REV, local_dir=OUT)
        n_dl += 1
print(f"완료: 링크 {n_link}개 + 새로 받음 {n_dl}개 = {n_link + n_dl}개 → {OUT}")
print("과목 폴더 수:", len({os.path.dirname(p) for p in glob.glob(os.path.join(OUT, "*", f"{SPLIT}-*.parquet"))}))
