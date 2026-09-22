"""
check_alive_ratio.py — 지금 데이터에서 "아직 지원 가능한 공고"가 몇 %인지 실측

목적
----
03(이력서-공고 매칭)이 로컬 배치 데이터에서 매칭하는데, 수집 시점(2026.06~07)
이후 시간이 꽤 지나서 상당수 공고가 이미 마감됐을 것으로 추정된다. 재크롤링
도입 여부를 정하기 전에, 지금 데이터에서 "오늘 기준으로 아직 열려있는 공고"가
실제로 몇 % 남아있는지부터 세어본다.

urgency_rule.py를 그대로 재사용하지 않는 이유
--------------------------------------------
urgency_rule.py의 parse_application_window()는 일부러 절대 마감일자를 버리고
"접수 창 길이(며칠짜리인가)"만 돌려준다 (점수가 조회 시점에 의존하면 안 되기
때문 — README/urgency_rule.py 주석 참고). 여기서 필요한 건 정반대로 절대
날짜라서, 같은 정규식(RX_START_END, RX_PERIOD_SEG 등)을 재사용하되 마지막에
날짜를 버리지 않고 그대로 돌려주는 함수를 새로 만들었다.

⚠️ urgency_rule.py의 내부 헬퍼(_mkdate 등, 밑줄로 시작하는 "비공개" 이름)에
   의존한다. 그 파일의 정규식 이름/구조가 바뀌면 이 스크립트도 같이 손봐야 함.

실행
----
    python check_alive_ratio.py                       # data/master_merged*.json 자동 탐색
    python check_alive_ratio.py --path 경로/파일.json   # 경로 직접 지정
    python check_alive_ratio.py --source jobkorea      # 특정 소스만 보고 싶을 때

로컬에 파일이 없으면 huggingface_hub으로 자동 다운로드를 시도한다
(01-tech-stack-wordcloud/app.py와 동일한 저장소 사용).
"""

import argparse
import datetime
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent

# urgency_rule.py는 02-urgency-score-analysis/ 아래 있다고 가정 (README 폴더 구조 기준)
sys.path.insert(0, str(HERE.parent / "02-urgency-score-analysis"))

try:
    from urgency_rule import (
        clean_body,
        RX_START_END,
        RX_PERIOD_SEG,
        RX_DATE,
        RX_ROLLING,
        RX_ALWAYS_OPEN,
        _mkdate,
    )
except ImportError:
    sys.exit(
        "urgency_rule.py를 찾을 수 없습니다. 이 스크립트가 03-resume-job-matching/ 안에 있고, "
        "그 옆에 02-urgency-score-analysis/urgency_rule.py가 있는 저장소 구조인지 확인하세요."
    )

HF_REPO_ID = "data-craftee/korean-it-recruit-dataset"
HF_FILENAME_CANDIDATES = ["master_merged_v2.json", "master_merged.json"]


# ---------------------------------------------------------------------------
# 절대 마감일 추출 — urgency_rule.py의 정규식을 재사용하되 날짜를 안 버림
# ---------------------------------------------------------------------------
def extract_deadline(body: str):
    """(절대 마감일 date | None, rolling 여부, 상시채용 여부, 매치 형태 라벨)"""
    body = body or ""

    m = RX_START_END.search(body)  # 형태 A (jobkorea: "시작일 ... 마감일 ...")
    if m:
        start = _mkdate(*m.group(1, 2, 3))
        end = _mkdate(*m.group(4, 5, 6))
        if start and end and 0 <= (end - start).days <= 400:
            seg = body[m.start(): m.end() + 40]
            rolling = bool(RX_ROLLING.search(seg))
            return end, rolling, False, "A(jobkorea 시작~마감)"

    m = RX_PERIOD_SEG.search(body)  # 형태 B (saramin: "접수기간 ... ~ ...")
    if m:
        seg = m.group(1)
        rolling = bool(RX_ROLLING.search(seg))
        ds = RX_DATE.findall(seg)
        if len(ds) >= 2:
            start, end = _mkdate(*ds[0]), _mkdate(*ds[1])
            if start and end and 0 <= (end - start).days <= 400:
                return end, rolling, False, "B(saramin 접수기간)"
        if RX_ALWAYS_OPEN.search(body):
            return None, rolling, True, "B(날짜 파싱 실패, 상시채용 문구는 있음)"
        return None, rolling, False, "B(날짜 파싱 실패)"

    if RX_ALWAYS_OPEN.search(body):
        return None, False, True, "상시채용 문구만 있음(마감일 섹션 자체가 없음)"

    return None, False, False, "매치 없음(형식 자체가 다름 — 주로 wanted)"


def classify(end_date, always_open, today):
    if always_open:
        return "생존(상시·수시채용)"
    if end_date is None:
        return "판단불가(마감정보 없음)"
    return "생존(마감일 미도래)" if end_date >= today else "마감(마감일 지남)"


# ---------------------------------------------------------------------------
# 데이터 로딩 — 로컬 파일 우선, 없으면 Hugging Face에서 다운로드
# ---------------------------------------------------------------------------
def load_records(path_arg: Optional[str]):
    if path_arg:
        candidates = [Path(path_arg)]
    else:
        data_dir = HERE.parent / "data"
        candidates = [
            data_dir / "master_merged.json",
            data_dir / "master_merged_v3.json",
            data_dir / "master_merged_v2.json",
        ]

    for p in candidates:
        if p.exists():
            print(f"[로딩] 로컬 파일 사용: {p}")
            with open(p, encoding="utf-8") as f:
                return json.load(f)

    print("[로딩] 로컬에 파일이 없어 Hugging Face에서 다운로드를 시도합니다...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "로컬 데이터 파일도 없고 huggingface_hub도 설치돼 있지 않습니다.\n"
            "  pip install huggingface_hub  또는  conda activate itjob 후 재실행하세요."
        )

    import os
    try:
        from dotenv import load_dotenv
        load_dotenv(HERE.parent / ".env")
    except ImportError:
        pass
    token = os.environ.get("HF_TOKEN")

    last_err = None
    for fname in HF_FILENAME_CANDIDATES:
        try:
            local_path = hf_hub_download(
                repo_id=HF_REPO_ID, filename=fname, repo_type="dataset", token=token
            )
            print(f"[로딩] Hugging Face에서 다운로드 완료: {fname}")
            with open(local_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001
            last_err = e
    sys.exit(f"Hugging Face 다운로드도 실패했습니다: {last_err}")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="지금 데이터에서 살아있는 공고 비율을 센다")
    parser.add_argument("--path", default=None, help="master_merged json 경로 (생략 시 자동 탐색)")
    parser.add_argument("--source", default=None, help="특정 소스만 (jobkorea/saramin/wanted)")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    records = load_records(args.path)
    if args.source:
        records = [r for r in records if r.get("source") == args.source]

    today = datetime.date.today()
    print(f"\n기준일: {today.isoformat()}  (스크립트를 실행한 오늘 날짜)")
    print(f"대상 건수: {len(records):,}건\n")

    overall = Counter()
    by_source = {}
    match_shape = Counter()
    rolling_but_dated_alive = 0

    for r in records:
        source = r.get("source", "unknown")
        raw = r.get("raw_text", "")
        body = clean_body(raw, source)
        end_date, rolling, always_open, shape = extract_deadline(body)
        label = classify(end_date, always_open, today)

        overall[label] += 1
        by_source.setdefault(source, Counter())[label] += 1
        match_shape[shape] += 1

        if rolling and end_date and end_date >= today:
            rolling_but_dated_alive += 1

    def pct(n, total):
        return f"{n / total * 100:.1f}%" if total else "N/A"

    order = [
        "생존(마감일 미도래)",
        "생존(상시·수시채용)",
        "마감(마감일 지남)",
        "판단불가(마감정보 없음)",
    ]

    total = len(records)
    print("=" * 70)
    print("전체 결과")
    print("=" * 70)
    for label in order:
        n = overall.get(label, 0)
        print(f"  {label:<24} {n:>7,}건  ({pct(n, total)})")

    alive_total = overall.get("생존(마감일 미도래)", 0) + overall.get("생존(상시·수시채용)", 0)
    print(f"\n  >> 생존 추정 합계: {alive_total:,}건 / {total:,}건 = {pct(alive_total, total)}")

    print("\n" + "=" * 70)
    print("소스별 결과")
    print("=" * 70)
    for source, counts in by_source.items():
        s_total = sum(counts.values())
        s_alive = counts.get("생존(마감일 미도래)", 0) + counts.get("생존(상시·수시채용)", 0)
        print(f"\n  [{source}]  n={s_total:,}")
        for label in order:
            n = counts.get(label, 0)
            print(f"    {label:<24} {n:>7,}건  ({pct(n, s_total)})")
        print(f"    {'생존 합계':<24} {s_alive:>7,}건  ({pct(s_alive, s_total)})")

    print("\n" + "=" * 70)
    print("진단 참고")
    print("=" * 70)
    print("  매치 형태 분포:")
    for shape, n in match_shape.most_common():
        print(f"    {shape:<45} {n:>7,}건")
    print(
        f"\n  '채용 시 마감(rolling)' 표시가 있으면서 아직 명시된 마감일이 안 지난 것: "
        f"{rolling_but_dated_alive:,}건"
    )
    print(
        "  → rolling 공고는 명시된 마감일 전에 조기 마감됐을 수 있어, 위 '생존' 수치는\n"
        "     실제보다 다소 낙관적인 상한선일 수 있습니다."
    )
    print(
        "\n  ⚠️ wanted 소스는 urgency_rule.py의 날짜 정규식이 애초에 jobkorea/saramin 형식만\n"
        "     다루기 때문에 대부분 '판단불가'로 나올 가능성이 높습니다 — 이는 실제로 마감된\n"
        "     게 아니라 이 스크립트가 wanted 형식을 아직 못 읽는 것뿐이니 착오하지 마세요."
    )


if __name__ == "__main__":
    main()
