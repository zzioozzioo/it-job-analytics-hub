"""
job_pool.py — 03이 쓰는 공고 데이터의 **단일 출처**.

---------------------------------------------------------------------------
왜 이 파일이 먼저 필요한가
---------------------------------------------------------------------------
`skill_matcher.py`(매칭)와 `eval_proxy.py`(채점)가 **같은 모집단**을 봐야 한다.
프록시 점수(Recall@10 등)는 "후보가 몇 건인가"에 직접 의존하기 때문에, 두 쪽이
각자 데이터를 로드하면 하한선(브리프의 92.5%)이 의미를 잃는다.

그리고 이 저장소는 이미 한 번 당했다 — 라벨 파일을 `(source, job_id)` dict로
조인했다가, 같은 키가 777종(1,554행) 중복돼 **서로 다른 공고가 짝지어졌다.**
그래서 여기서는 위치(행 순서)를 유지하고, 중복은 지우지 않고 **세어서 알린다.**

---------------------------------------------------------------------------
코퍼스와 풀을 나눈다 — 이게 이 모듈의 핵심 설계
---------------------------------------------------------------------------
                    건수        쓰임
    코퍼스        40,348      IDF·동시출현 등 **어휘 통계**
    추천 풀       가변        사용자에게 실제로 보여줄 후보

둘을 나누는 이유는 사람인 25,950건(전체의 64%)에 `title`·`company`·`location`·
`experience_level`이 **전부 비어 있기** 때문이다(수집 단계 누락. 원본
`saramin_cleaned_techs2.json`의 키가 `공고번호`·`원문`·스킬뿐이다).

    - 통계에는 넣어도 된다. `raw_text`와 스킬은 멀쩡하다.
    - 추천 대상에는 넣으면 안 된다. 제목이 없어 화면에 못 띄우고, 더 나쁜 것은
      **경력·지역 하드필터를 항상 통과한다**는 점이다(필터가 "값 없으면 통과"
      설계라서). 신입 이력서에 '경력 5년 이상'이 그대로 추천된다.

저장된 `raw_text`에서 정규식으로 복원해 보려 했으나 불가능했다(3,000건 표본):
    title 9.4% · company 22.3% · 지역(`근무지역` 앵커) 3.1%
앵커 없이 본문에서 시·도명만 긁으면 80.4%가 잡히지만 `서울 소재 고객사` 같은
문장이 그대로 걸려 **조용히 틀린다.** 쓰지 않는다.

→ 그래서 `backfill_saramin.py`(meta description 재조회)가 유일한 길이고,
  이 모듈은 그 산출물이 **있으면 자동으로 쓰고 없으면 무시한다.** 백필 전에
  짠 코드가 백필 후에 그대로 넓은 풀을 쓴다. 실행 중이어도(부분 파일) 된다.

---------------------------------------------------------------------------
`as_of` — 지원 가능 여부는 저장하지 않고 조회 시점에 계산한다
---------------------------------------------------------------------------
마감일은 크롤링 주기와 무관하게 매일 지나간다. `is_open`을 컬럼으로 저장하면
어제 True이던 값이 오늘 거짓말이 된다. 02가 `남은기간`으로 똑같이 당했다
(`urgency_rule.py` [수정 2] — 공고의 속성이 아니라 *내가 언제 봤는가*의 속성).

그래서 `status`는 `as_of`를 받아 그때 계산한다. 기본값은 오늘.

상태는 **세 가지가 아니라 네 가지**다. `unknown`을 `open`에 섞지 않는다:

    open     마감일이 as_of 이후
    closed   마감일이 지남
    rolling  상시·수시채용 (마감일 개념이 없음)
    unknown  마감일을 못 읽음 — '열려 있다'는 뜻이 **아니다**

현 스냅샷(2026.06~07 수집) 기준 실측:
    as_of=2026-06-20 → open 15,500건 / as_of=2026-09-14 → open 2건
즉 오늘 기준으로는 추천이 성립하지 않는다. 이건 코드가 아니라 입력의 문제이고,
`as_of`를 과거로 두면 필터와 순위가 정상 동작하는 것으로 확인할 수 있다.

---------------------------------------------------------------------------
사용법
---------------------------------------------------------------------------
    from job_pool import load_corpus, load_pool, build_idf

    corpus = load_corpus()                 # 40,348건 — IDF 재료
    idf    = build_idf(corpus)             # {기술명: idf}

    pool = load_pool(as_of='2026-06-20', only_open=True)
    pool[['job_id', 'title', 'company', 'deadline', 'status', 'techs']]

    python job_pool.py --report            # 현재 커버리지 출력 (네트워크 없음)
"""
import argparse
import ast
import datetime
import functools
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "02-urgency-score-analysis"))

import hf_data                                    # noqa: E402  데이터 접근 단일 출처
import urgency_rule as U                          # noqa: E402  규칙 정규식 재사용
from tech_normalize import canonical, extract_techs   # noqa: E402  기술명 정규화 단일 출처

# 백필 산출물. 없어도 동작한다(사람인이 추천 풀에서 빠질 뿐).
BACKFILL = hf_data.DATA_DIR / "saramin_meta_backfill.json"

META_FIELDS = ["title", "company", "location", "experience_level"]

# 추천 대상이 되려면 이 필드들이 채워져 있어야 한다.
# 백필 전: jobkorea 10,025 + wanted 4,373 = 14,398건 (전체의 36%)
# 백필 후: + saramin (80건 시험에서 title·company·경력·마감일 100%, location은
#          --with-location 이 있어야 나온다)
REQUIRED_FOR_POOL = ["title", "company"]


# ---------------------------------------------------------------------------
# 마감일 — 세 경로를 순서대로 시도한다
# ---------------------------------------------------------------------------
# 02의 parse_application_window()는 점수 산출용이라 **접수 창 길이**만 돌려주고
# 절대 날짜를 버린다. 03에 필요한 건 종료일 하나뿐이라, 같은 정규식을 쓰되
# 시작일을 요구하지 않는다. 그래서 커버리지가 02보다 높다:
#     saramin  02 방식 54.4%  ->  종료일만 69.6%
RX_DEADLINE_LABEL = re.compile(
    r'마감일\s*[:\s]?\s*(20\d\d)\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})')
RX_DATE_ANY = re.compile(
    r'(20\d\d)\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})')


def _date(y, m, d):
    try:
        return datetime.date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None


def parse_deadline(raw_text: str):
    """(마감일 | None, rolling 여부). 백필 값이 없을 때의 폴백."""
    t = raw_text or ''
    hits = [x for x in (_date(*g) for g in RX_DEADLINE_LABEL.findall(t)) if x]
    if hits:
        return max(hits), bool(U.RX_ROLLING.search(t))

    # 접수기간 블록 안의 마지막 날짜 = 종료일. 시작일은 필요 없다.
    m = U.RX_PERIOD_SEG.search(t)
    if m:
        seg = m.group(1)
        ds = [x for x in (_date(*g) for g in RX_DATE_ANY.findall(seg)) if x]
        if ds:
            return max(ds), bool(U.RX_ROLLING.search(seg))
    return None, bool(U.RX_ROLLING.search(t))


def status_of(deadline, rolling, as_of):
    if deadline is not None:
        return 'open' if deadline >= as_of else 'closed'
    return 'rolling' if rolling else 'unknown'


# ---------------------------------------------------------------------------
# 로딩
# ---------------------------------------------------------------------------
def _as_list(cell):
    """저장된 리스트 컬럼은 문자열로 직렬화돼 있다."""
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        v = ast.literal_eval(cell)
        return v if isinstance(v, list) else []
    except (ValueError, SyntaxError):
        return []


def _blank(v):
    return v is None or (isinstance(v, str) and v.strip() in ('', 'None', 'nan'))


def load_corpus(version: str = 'v4') -> pd.DataFrame:
    """40,348건 전부. 어휘 통계(IDF·동시출현)용 — 메타데이터 결손과 무관하다."""
    path = hf_data.fetch(hf_data.master(version))
    with open(path, encoding='utf-8') as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    for c in META_FIELDS:
        if c in df.columns:
            df[c] = df[c].map(lambda v: None if _blank(v) else v)
    return df


def load_backfill() -> dict:
    """{job_id: {title, company, experience_level, education, deadline, location}}

    실행 중이라 부분만 채워져 있어도 그대로 쓴다. status != 'ok' 인 건은 버린다.
    """
    if not BACKFILL.exists():
        return {}
    with open(BACKFILL, encoding='utf-8') as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if v.get('status') == 'ok'}


def _techs_cache_path(version: str) -> Path:
    return hf_data.DATA_DIR / f"techs_extracted_{version}.json"


def techs_series(corpus: pd.DataFrame, version: str = 'v4',
                 use_text: bool = True, rebuild: bool = False):
    """공고별 (표준명 집합, 비사전 스킬 집합) 리스트. 본문 추출은 캐시한다.

    `extract_techs()`를 40,348건에 돌리면 4~9분이 걸린다(본문 길이에 비례.
    잡코리아·사람인은 4,000자대, 원티드는 785자). 매 실행마다 낼 비용이
    아니므로 `data/techs_extracted_<version>.json`에 저장한다.

    캐시는 **위치(행 순서)로** 맞춘다. `(source, job_id)`로 맞추면 안 된다 —
    같은 키가 777종(1,554행) 중복이라 서로 다른 공고가 짝지어진다. 이 저장소가
    v2→v3 라벨 변경을 11행 과다 보고했던 바로 그 사고다.
    캐시가 코퍼스와 길이가 다르면 버리고 다시 만든다.
    """
    if not use_text:
        return [split_skills(list(_as_list(r.get('final_techs')))
                             + list(_as_list(r.get('hard_skills'))))
                for r in corpus.to_dict('records')]

    p = _techs_cache_path(version)
    if p.exists() and not rebuild:
        try:
            with open(p, encoding='utf-8') as f:
                blob = json.load(f)
            if (blob.get('n') == len(corpus) and blob.get('version') == version
                    and blob.get('schema') == 2):
                return [(set(a), set(b)) for a, b in zip(blob['techs'], blob['extra'])]
        except (ValueError, KeyError):
            pass                                   # 깨졌으면 다시 만든다

    out = [_techs_of(r, True) for r in corpus.to_dict('records')]
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'version': version, 'n': len(corpus), 'schema': 2,
                   'techs': [sorted(a) for a, _ in out],
                   'extra': [sorted(b) for _, b in out]}, f, ensure_ascii=False)
    tmp.replace(p)                                 # 원자적 교체
    return out


@functools.lru_cache(maxsize=None)
def _canon(name: str):
    """`canonical()`은 퍼지 매칭이 붙어 있어 느리다. 같은 이름이 수만 번 반복되므로 캐시한다."""
    return canonical(name)


def split_skills(names) -> tuple:
    """원문 스킬 목록 -> (사전 표준명 집합, 비사전 스킬 집합)

    ---------------------------------------------------------------------
    왜 나누는가 — 실측 근거
    ---------------------------------------------------------------------
    LLM이 만든 `hard_skills`는 표기가 제각각이라 **같은 기술이 쪼개져 있었다**
    (6,000건 표본): `JAVA` 248 + `Java` 967 · `Javascript` 65 + `JavaScript` 517
    · `RESTful API` 65 + `REST API` 282 · `Spring Framework` 74 + `Spring` 416.
    이력서에 "Java"라고 쓰면 `JAVA`만 적힌 공고와 안 맞고, IDF 문서빈도도
    변형끼리 갈라진다. 그래서 사전(`tech_normalize`)으로 표준명에 모은다.

    그런데 표준명만 남기면 **스킬 0개 공고가 6.3% -> 29.1%로 뛴다.** 사전이
    IT 기술 158종만 다루기 때문에 `엑셀`·`정보처리기사`·`ERP`·`MS Office`·`회계`
    같은 실재 요구사항이 통째로 버려진다. 그것도 손실이다.

    그래서 버리지 않고 두 번째 층에 둔다. 매칭 키로 쓸지는 `skill_matcher` /
    `eval_proxy`가 플래그로 정한다(브리프의 "플래그로 두고 프록시로 고른다").

    표준명만 쓰면 가짜 이력서 왕복 재현율이 48.5% -> 99.4%가 된다 —
    프록시가 '매칭 성능'이 아니라 '사전 커버리지'를 재는 것을 막아준다.
    """
    canon, extra = set(), set()
    for x in names:
        if not isinstance(x, str) or not x.strip():
            continue
        c = _canon(x)
        (canon.add(c) if c else extra.add(x.strip().lower()))
    return canon, extra


def _techs_of(row, use_text: bool) -> tuple:
    """(표준명 집합, 비사전 스킬 집합). 브리프의 final_techs ∪ hard_skills + 본문 추출.

    본문 추출(`extract_techs`)은 LLM이 만든 스킬 컬럼을 잡코리아 91.9% ·
    사람인 91.7% 재현하면서 공고당 3.6 / 0.6개를 **추가로** 찾아낸다
    (각 1,500건 표본). 원티드만 67.1%인데, 놓친 것의 88%가 저장된 raw_text에
    아예 없는 문자열이라 파서로 메울 수 없다(수집 단계에서 기술스택 태그가
    누락됨). 그래서 합집합으로 둔다 — 어느 쪽도 상위집합이 아니다.
    """
    canon, extra = split_skills(
        list(_as_list(row.get('final_techs'))) + list(_as_list(row.get('hard_skills'))))
    if use_text:
        canon |= extract_techs(row.get('raw_text') or '')   # 이미 표준명으로 나온다
    return canon, extra


def load_pool(as_of=None,
              only_open: bool = False,
              require_techs: bool = False,
              use_backfill: bool = True,
              use_text_extraction: bool = True,
              sources=None,
              version: str = 'v4',
              corpus: pd.DataFrame = None) -> pd.DataFrame:
    """추천 대상 공고.

    as_of               'YYYY-MM-DD' | date | None(오늘). status 계산 기준일
    only_open           True 면 status in {'open','rolling'} 만 남긴다
    require_techs       True 면 techs 가 빈 공고를 뺀다. 기본 False —
                        지금은 매칭이 안 되지만 4단계 임베딩이 들어오면
                        텍스트로 매칭될 수 있다. 데이터 계층에 "스킬 매칭"이라는
                        전제를 못 박지 않기 위해 기본값은 남겨두고 세어서 알린다
                        (현재 230건 / 1.4%)
    use_backfill        saramin_meta_backfill.json 이 있으면 병합
    use_text_extraction techs 에 본문 추출을 합칠지
    sources             ['jobkorea','wanted'] 처럼 제한
    corpus              이미 로드해 뒀으면 넘긴다 (178MB JSON 재로드 방지)

    반환 컬럼: job_id source title company location experience_level
               techs(set 표준명) skills_extra(set 비사전) deadline(date|None)
               rolling(bool) status raw_text
    """
    if as_of is None:
        as_of = datetime.date.today()
    elif isinstance(as_of, str):
        as_of = datetime.date.fromisoformat(as_of)

    df = corpus if corpus is not None else load_corpus(version)
    bf = load_backfill() if use_backfill else {}
    techs = techs_series(df, version, use_text_extraction)

    recs = []
    for i, row in enumerate(df.to_dict('records')):
        jid, src = str(row.get('job_id')), row.get('source')
        b = bf.get(jid) if src == 'saramin' else None

        title = row.get('title') or (b or {}).get('title')
        company = row.get('company') or (b or {}).get('company')
        location = row.get('location') or (b or {}).get('location')
        exp = row.get('experience_level') or (b or {}).get('experience_level')

        deadline, rolling = None, False
        if b and b.get('deadline'):
            deadline = _date(*b['deadline'].split('-'))      # 백필은 YYYY-MM-DD
        if deadline is None:
            deadline, rolling = parse_deadline(row.get('raw_text'))

        recs.append({
            'job_id': jid, 'source': src,
            'title': title, 'company': company,
            'location': location, 'experience_level': exp,
            'techs': techs[i][0],
            'skills_extra': techs[i][1],
            'deadline': deadline, 'rolling': rolling,
            'status': status_of(deadline, rolling, as_of),
            'raw_text': row.get('raw_text'),
        })

    pool = pd.DataFrame(recs)
    if sources:
        pool = pool[pool['source'].isin(sources)]

    # 메타데이터가 없는 공고는 추천 대상에서 뺀다 (위 docstring 참조).
    keep = pool[REQUIRED_FOR_POOL].notna().all(axis=1)
    pool = pool[keep]

    if require_techs:
        pool = pool[pool['techs'].map(len) > 0]
    if only_open:
        pool = pool[pool['status'].isin(['open', 'rolling'])]
    return pool.reset_index(drop=True)


def build_idf(corpus: pd.DataFrame = None, min_df: int = 5,
              use_text_extraction: bool = True,
              include_extra: bool = False) -> dict:
    """{기술명: idf}. **코퍼스 전체**로 만든다 — 추천 풀이 아니라.

    IDF는 "이 기술이 얼마나 희귀한가"라는 어휘 통계라 신선도·메타데이터 결손에
    둔감하다. 사람인 25,950건은 제목이 없어 추천은 못 해도 이 통계에는 기여한다.
    풀(14,398건)로 만들면 표본이 3분의 1로 줄 뿐 정확해지지 않는다.
    """
    if corpus is None:
        corpus = load_corpus()
    df = Counter()
    for canon, extra in techs_series(corpus, use_text=use_text_extraction):
        df.update(canon | extra if include_extra else canon)
    n = len(corpus)
    return {t: math.log(n / c) for t, c in df.items() if c >= min_df}


# ---------------------------------------------------------------------------
def report(as_of=None):
    as_of = as_of or datetime.date.today().isoformat()
    corpus = load_corpus()
    bf = load_backfill()
    print(f"\n  코퍼스 {len(corpus):,}건 · 백필 파일 {len(bf):,}건 "
          f"({'있음' if BACKFILL.exists() else '없음'})\n")

    pool = load_pool(as_of=as_of, corpus=corpus)
    print(f"  기준일 {as_of} · 추천 풀 {len(pool):,}건 "
          f"(코퍼스의 {len(pool)/len(corpus):.0%})\n")
    print(f"    {'source':10}{'풀':>8}{'open':>8}{'rolling':>9}{'closed':>8}{'unknown':>9}")
    print('    ' + '-' * 52)
    for src, g in pool.groupby('source'):
        c = Counter(g['status'])
        print(f"    {src:10}{len(g):>8,}{c['open']:>8,}{c['rolling']:>9,}"
              f"{c['closed']:>8,}{c['unknown']:>9,}")
    c = Counter(pool['status'])
    print(f"    {'합계':10}{len(pool):>8,}{c['open']:>8,}{c['rolling']:>9,}"
          f"{c['closed']:>8,}{c['unknown']:>9,}")

    notech = int((pool['techs'].map(len) == 0).sum())
    print()
    both = int(((pool['techs'].map(len) == 0)
                & (pool['skills_extra'].map(len) == 0)).sum())
    print(f"    techs(표준명) 0개 {notech:,}건 · 두 층 모두 0개 {both:,}건 "
          f"— 후자는 스킬로 영원히 추천되지 않는다")

    miss = pool[['location', 'experience_level']].isna().sum()
    print(f"\n    하드필터 필드 결손 — location {miss['location']:,} · "
          f"experience_level {miss['experience_level']:,}")

    idf = build_idf(corpus)
    t = techs_series(corpus)
    nz = sum(1 for a, _ in t if a)
    print(f"    IDF 사전 {len(idf):,}종 (문서빈도 5건 이상) · "
          f"분모 N={len(corpus):,} 중 스킬 보유 {nz:,} ({nz/len(corpus):.1%})")

    dup = pool.duplicated(subset=['source', 'job_id']).sum()
    print(f"    (source, job_id) 중복 {dup:,}건 — 지우지 않는다. "
          f"같은 키에 다른 공고가 있다(README 참조)\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', action='store_true',
                    help='커버리지 출력 (인자가 없어도 기본 동작)')
    ap.add_argument('--as-of', default=None, help='YYYY-MM-DD (기본: 오늘)')
    a = ap.parse_args()
    report(a.as_of)
