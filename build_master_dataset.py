"""
build_master_dataset.py

wanted / jobkorea / saramin 3개 크롤링 소스를 프로젝트 전체(1~4번)가 공통으로 쓰는
마스터 통합 데이터셋(data/master_merged.json)으로 병합한다.

실행: python build_master_dataset.py
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "data"
WANTED_PATH = DATA_DIR / "wanted_cleaned_techs2.json"
JOBKOREA_PATH = DATA_DIR / "jobkorea_cleaned_techs2.json"
SARAMIN_PATH = DATA_DIR / "saramin_cleaned_techs2.json"
OUT_PATH = DATA_DIR / "master_merged.json"


def load(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 0. 원본 로드
# ---------------------------------------------------------------------------
wanted = load(WANTED_PATH)
jobkorea = load(JOBKOREA_PATH)
saramin = load(SARAMIN_PATH)

print("=" * 70)
print("STEP 1. 각 소스 컬럼 목록 확인")
print("=" * 70)
print(f"[wanted]   ({len(wanted)} rows) columns:")
print(" ", list(wanted[0].keys()))
print(f"[jobkorea] ({len(jobkorea)} rows) columns:")
print(" ", list(jobkorea[0].keys()))
print(f"[saramin]  ({len(saramin)} rows) columns:")
print(" ", list(saramin[0].keys()))

print()
print("--- wanted.year_filter unique 값 분포 ---")
s = pd.Series([x.get("year_filter") for x in wanted]).astype(str)
print(s.value_counts())

print()
print("--- jobkorea.연차/경력 unique 값 분포 ---")
s2 = pd.Series([x.get("연차/경력") for x in jobkorea]).astype(str)
print(s2.value_counts())

print()
print("--- saramin: 연차/경력 관련 컬럼 없음 (확인 생략) ---")
print(" ", list(saramin[0].keys()))


# ---------------------------------------------------------------------------
# STEP 2. 연차 표기 통일
# ---------------------------------------------------------------------------
# 공통 카테고리 (5종): '신입' / '1~3년차' / '3~5년차' / '5년이상' / '경력무관'
#
# [wanted.year_filter]
#   원티드 공고의 '경력' 필터 값으로, 요구되는 "최소 연차(숫자)"를 의미한다.
#   실제 값 분포: '' (3261), 0.0 (151), 1.0 (100), 3.0 (499), 5.0 (362)
#     - ''(빈 값)/None : 경력 필터를 지정하지 않음            -> 경력무관
#     - 0.0            : 신입(0년) 필터                      -> 신입
#     - 1.0            : 1년 이상 필터                        -> 1~3년차
#     - 3.0            : 3년 이상 필터                        -> 3~5년차
#     - 5.0 이상       : 5년 이상 필터                        -> 5년이상
#
# [jobkorea.연차/경력]
#   자유 텍스트 형태('경력N년↑', '신입', '경력무관', '신입·경력', '신입·경력N년↑' 등).
#   실제 값 분포(상위): 경력3년↑(1631), 경력무관(1429), 경력5년↑(1075),
#     신입·경력(996), 경력(889), 경력2년↑(681), 경력1년↑(533), 경력10년↑(412),
#     신입(403), 경력6년↑(354), ... '' / None (22)
#     - '경력무관'                : 그대로                    -> 경력무관
#     - '신입' (단독)             : 그대로                    -> 신입
#     - '경력' (숫자 없음, 단독)   : 구체적 연차 명시 없음     -> 경력무관
#     - '신입·경력' (숫자 없음)    : 신입/경력 모두 허용, 연차 제한 없음 -> 경력무관
#     - '경력N년↑'                : 숫자 N을 파싱해 구간 분류 (아래 기준)
#     - '신입·경력N년↑'           : 숫자 N 기준 구간으로 분류
#                                   (※ '신입도 지원 가능'이라는 정보는 이 과정에서 버려짐 -
#                                      공고가 명시한 상한 연차 기준으로 그룹핑하기 위한 근사치)
#     - '' / None                : 정보 없음                  -> 경력무관
#   N 구간 분류 기준 (wanted의 1.0/3.0/5.0 임계값과 동일 선상에서 정렬):
#     - N in {1, 2}   -> '1~3년차'
#     - N in {3, 4}   -> '3~5년차'
#     - N >= 5        -> '5년이상'
#
# [saramin]
#   연차/경력 관련 컬럼 자체가 없음 -> experience_level = NaN 그대로 둠 (드랍하지 않음)

EXPERIENCE_CATEGORIES = ["신입", "1~3년차", "3~5년차", "5년이상", "경력무관"]


def map_wanted_experience(yf):
    if yf is None or yf == "":
        return "경력무관"
    try:
        v = float(yf)
    except (TypeError, ValueError):
        return "경력무관"
    if v <= 0:
        return "신입"
    elif v < 3:
        return "1~3년차"
    elif v < 5:
        return "3~5년차"
    else:
        return "5년이상"


def map_jobkorea_experience(raw):
    if raw is None or str(raw).strip() == "":
        return "경력무관"
    raw = str(raw).strip()
    if raw == "경력무관":
        return "경력무관"
    if raw == "신입":
        return "신입"
    if raw in ("경력", "신입·경력"):
        return "경력무관"
    m = re.search(r"(\d+)\s*년", raw)
    if m:
        n = int(m.group(1))
        if n <= 2:
            return "1~3년차"
        elif n <= 4:
            return "3~5년차"
        else:
            return "5년이상"
    return "경력무관"  # 예상치 못한 패턴 fallback


# ---------------------------------------------------------------------------
# STEP 3. 공통 스키마로 변환
# ---------------------------------------------------------------------------
FINAL_COLUMNS = [
    "source",
    "job_id",
    "company",
    "title",
    "location",
    "experience_level",
    "raw_text",
    "final_techs",
    "hard_skills",
    "soft_skills",
    "preferences",
    "culture_keywords",
    "urgency_score",
    "urgency_reason",
]


def join_text(*parts):
    return "\n\n".join(p for p in parts if p)


def build_wanted_rows(records):
    rows = []
    for x in records:
        rows.append({
            "source": "wanted",
            "job_id": str(x.get("job_id")),
            "company": x.get("company"),
            "title": x.get("position"),
            "location": x.get("location_perfect"),
            "experience_level": map_wanted_experience(x.get("year_filter")),
            "raw_text": join_text(x.get("main_tasks"), x.get("requirements"), x.get("preferred")),
            "final_techs": x.get("final_techs"),
            "hard_skills": x.get("hard_skills"),
            "soft_skills": x.get("soft_skills"),
            "preferences": x.get("preferences"),
            "culture_keywords": x.get("culture_keywords"),
            "urgency_score": x.get("urgency_score"),
            "urgency_reason": x.get("urgency_reason"),
        })
    return rows


def build_jobkorea_rows(records):
    rows = []
    for x in records:
        rows.append({
            "source": "jobkorea",
            "job_id": str(x.get("공고번호")),
            "company": x.get("회사명"),
            "title": x.get("공고제목"),
            "location": x.get("location_clean"),
            "experience_level": map_jobkorea_experience(x.get("연차/경력")),
            "raw_text": x.get("상세내용"),
            "final_techs": x.get("final_techs"),
            "hard_skills": x.get("hard_skills"),
            "soft_skills": x.get("soft_skills"),
            "preferences": x.get("preferences"),
            "culture_keywords": x.get("culture_keywords"),
            "urgency_score": x.get("urgency_score"),
            "urgency_reason": x.get("urgency_reason"),
        })
    return rows


def build_saramin_rows(records):
    rows = []
    for x in records:
        rows.append({
            "source": "saramin",
            "job_id": str(x.get("공고번호")),
            "company": None,
            "title": None,
            "location": None,
            "experience_level": None,  # saramin: 연차 컬럼 없음 -> NaN 유지
            "raw_text": x.get("원문"),
            "final_techs": x.get("final_techs"),
            "hard_skills": x.get("hard_skills"),
            "soft_skills": x.get("soft_skills"),
            "preferences": x.get("preferences"),
            "culture_keywords": x.get("culture_keywords"),
            "urgency_score": x.get("urgency_score"),
            "urgency_reason": x.get("urgency_reason"),
        })
    return rows


all_rows = (
    build_wanted_rows(wanted)
    + build_jobkorea_rows(jobkorea)
    + build_saramin_rows(saramin)
)

df = pd.DataFrame(all_rows)

# ---------------------------------------------------------------------------
# STEP 4. 스키마 외 컬럼 드랍
# ---------------------------------------------------------------------------
# dict 컴프리헨션 단계에서 이미 FINAL_COLUMNS만 뽑아 왔으므로, 아래 컬럼들은
# 애초에 병합 대상에서 제외됨 (=드랍 처리됨):
#   - skill_tags / 기술스택/분야 / 분야   : final_techs로 이미 정제된 필드라 중복
#   - year_filter / 연차/경력 (원본)      : experience_level로 변환 완료, 원본 불필요
#   - tag_id                              : 크롤링 내부용 값
#   - scraped_date / 마감일               : 의미가 달라 통합하지 않음, 현재 프로젝트 범위에서 미사용
#   - location_clean / location_final (wanted 내부 중간 산출물), 원본 location 등
# 안전장치로 한 번 더 컬럼을 명시적으로 select 한다.
df = df[FINAL_COLUMNS]

# ---------------------------------------------------------------------------
# STEP 5. saramin의 company/title/location/experience_level NaN은 의도된 것 -> 그대로 유지
# ---------------------------------------------------------------------------
df = df.where(pd.notnull(df), None)

# ---------------------------------------------------------------------------
# STEP 6. 저장 + 요약
# ---------------------------------------------------------------------------
records_out = df.to_dict(orient="records")
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(records_out, f, ensure_ascii=False, indent=2)

print()
print("=" * 70)
print(f"저장 완료: {OUT_PATH}  (총 {len(df)} rows)")
print("=" * 70)

print()
print("--- 최종 컬럼 목록 ---")
print(list(df.columns))

print()
print("--- 소스별 row 수 ---")
print(df["source"].value_counts())

print()
print("--- 연차별(experience_level) row 수 (NaN 포함) ---")
print(df["experience_level"].value_counts(dropna=False))

print()
print("--- 컬럼별 결측치(NaN) 비율 ---")
na_ratio = (df.isna().mean() * 100).round(2)
print(na_ratio.astype(str) + " %")
