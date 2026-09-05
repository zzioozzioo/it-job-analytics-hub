"""
urgency_rule.py  —  채용 적극성 라벨 규칙 v3

2-1(모델 학습)과 2-2(Streamlit 앱)가 함께 쓰는 단일 출처.
v2(`rescore_urgency.py`)를 두 군데 고쳤다. 둘 다 v2 라벨의 실측된 결함이다.

---------------------------------------------------------------------------
[수정 1] 지원자 수를 모집인원으로 오인하던 버그
---------------------------------------------------------------------------
v2의 RX_HEADCOUNT_ALT = r'([0-9]{1,3})\\s*명\\s*(?:모집|채용|선발)' 이
jobkorea 본문 꼬리의

    "지원자 현황 통계 지원자 수 8 명 모집인원 ○○ 명"

에서 `8 명 모집`을 잡아 '모집인원 8명'으로 읽었다. jobkorea는 모집인원을
`○○`로 마스킹하기 때문에 정식 패턴이 비고, 바로 옆 지원자 수가 대신 걸린다.

실측: jobkorea 4,161행 / saramin 4행에서 발생.
의미가 정반대다 — 지원자가 많다는 것은 그 공고가 **덜** 급하다는 뜻인데
v2는 +20점(모집 규모 큼)을 얹었다.

라벨 영향: measurable 30,569행 중 3,486행(11.4%)의 등급이 바뀐다.
    4점 -> 2점  1,487행      5점 -> 3점  565행
    3점 -> 2점    549행      5점 -> 4점  437행
    4점 -> 3점    208행      4점 -> 1점  106행
오차가 1등급이 아니라 2등급짜리다.

---------------------------------------------------------------------------
[수정 2] 마감 신호를 소스 무관하게 '접수 창 길이'로 통일
---------------------------------------------------------------------------
v2는 소스마다 다른 물리량을 쟀다.

    jobkorea : `남은기간 N일`        = 마감일 - 크롤링한 날
    saramin  : `접수기간 시작~종료`  = 마감일 - 시작일

이게 2-1 README가 지목한 전이 실패(leave-one-source-out QWK 0.04)의 근본
원인이다. 두 소스의 라벨이 애초에 **다른 타깃**이었으니 한 소스로 배운 것이
다른 소스에 통할 리가 없다.

더 근본적으로, `남은기간`은 공고의 속성이 아니라 **내가 언제 봤는가**의 속성이다.
같은 공고를 하루 뒤에 크롤링하면 라벨이 달라진다. 라벨이 크롤링 타이밍에
의존하면 그 라벨로 학습한 모델은 시점을 외우게 된다.

접수 창 길이는 양쪽에서 뽑히고, 실측 분포도 거의 같다.

    jobkorea (시작일~마감일, 7,577행)         p10=13 p25=30 p50=30 p75=40 p90=60
    saramin  (접수기간 시작~종료, 14,099행)   p10=14 p25=25 p50=30 p75=30 p90=60

같은 물리량 · 같은 분포이므로 하나의 타깃으로 묶을 수 있다.
v3은 `남은기간`을 라벨 산출에서 제외하고 창 길이만 쓴다.

⚠️ `남은기간`을 버린 것이 사용자에게 쓸모없다는 뜻은 아니다. URL을 붙여넣는
   앱 사용자에게 "이 공고 며칠 남았나"는 가장 궁금한 정보다. 앱은 그것을
   **부가 정보로 표시**하되 **점수에는 넣지 않는다**. 점수가 조회 시점에
   의존하면 어제 4점이던 공고가 오늘 5점이 된다.

---------------------------------------------------------------------------
바뀌지 않은 것
---------------------------------------------------------------------------
- 신호 가중치, 등급 경계(to_level), measurable 판정 기준
- 어휘 폴백(score_by_vocabulary) — 2-1에서 "상수보다 MAE가 나쁘다"고
  측정된 그대로다. 고치지 않고 남겨둔 이유는 한계 문서에 적었다.
- '채용 시 마감'(rolling)의 탐지 범위 — 형태 B(saramin)는 v2와 동일한
  `접수기간` 뒤 90자. 형태 A(jobkorea)는 그보다 좁게 잡는다.
  (`parse_application_window` 주석 참조. 처음엔 본문 전체를 뒤졌는데, 같은
   문구를 RX_EARLY_CLOSE가 이미 세고 있어 한 문구로 22점이 되는 이중 계상이었다.)
비교 가능성을 위해 이것들은 v2와 동일하게 유지했다. v2 대비 성능 차이가
위 두 수정에서만 오도록 하기 위해서다.

v2 대비 최종 라벨 변경: 40,348행 중 5,694행(14.1%).

실행: python urgency_rule.py            분포 리포트만
      python urgency_rule.py --write    data/master_merged_v3.json 생성
"""

import datetime
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
SRC_PATH = DATA_DIR / "master_merged.json"
OUT_PATH = DATA_DIR / "master_merged_v3.json"

RULE_VERSION = "v3"


# ---------------------------------------------------------------------------
# 1. 본문 정제
# ---------------------------------------------------------------------------
_JK_TAIL_MARKERS = ['본 채용정보는', '로그인 하고 비슷한 조건의', '관련 태그']
_JK_DISCLAIMER = re.compile(
    r'마감일은\s*기업의\s*사정으로\s*인해\s*조기\s*마감\s*또는\s*변경될\s*수\s*있습니다'
)

# [수정 1] 지원자 현황 통계 블록. '지원자'의 통계이지 '공고'의 내용이 아니다.
# 신호 탐지 전에 들어낸다. 끝 경계를 지원자 수까지로 잡으면 바로 뒤에 오는
# 실제 `모집인원` 값은 살아남는다.
_JK_APPLICANT_STATS = re.compile(
    r'지원자\s*현황\s*통계.{0,80}?지원자\s*수\s*[0-9,]+\s*명', re.S
)


def clean_body(raw_text: str, source: str) -> str:
    """신호 탐지에 쓸 '공고 본문'만 남긴다."""
    if not raw_text:
        return ""
    if source != 'jobkorea':
        return raw_text

    cut = len(raw_text)
    for mk in _JK_TAIL_MARKERS:
        i = raw_text.find(mk)
        if i != -1:
            cut = min(cut, i)
    body = raw_text[:cut]
    body = _JK_DISCLAIMER.sub(' ', body)
    return _JK_APPLICANT_STATS.sub(' ', body)          # [수정 1]


# ---------------------------------------------------------------------------
# 2. 접수 창 길이 — 소스 통합 마감 신호  [수정 2]
# ---------------------------------------------------------------------------
RX_DATE = re.compile(
    r'(20[0-9]{2})\s*[-.년]\s*([0-9]{1,2})\s*[-.월]\s*([0-9]{1,2})\s*일?')
RX_ROLLING = re.compile(r'채용\s*시\s*마감|채용시까지|채용\s*시\s*까지')
RX_ALWAYS_OPEN = re.compile(
    r'마감일\s*상시채용|상시\s*채용|수시\s*채용|상시\s*모집|연중\s*수시')

# 형태 A — jobkorea: "시작일 2026.06.10(수) 마감일 2026.06.23(화)"
RX_START_END = re.compile(
    r'시작일\s*(20\d\d)\.(\d{1,2})\.(\d{1,2})[^0-9]{0,8}마감일\s*(20\d\d)\.(\d{1,2})\.(\d{1,2})')
# 형태 B — saramin: "접수기간 ... 2026. 6. 8(월) ~ 2026. 6. 13(토)"
RX_PERIOD_SEG = re.compile(r'접수\s*기간(.{0,90})', re.S)


def _mkdate(y, m, d):
    try:
        return datetime.date(int(y), int(m), int(d))
    except ValueError:
        return None


def parse_application_window(body: str):
    """(접수창 일수 | None, 채용시마감 여부)

    v2와 달리 jobkorea의 `시작일~마감일`도 같은 창 길이로 읽는다.
    소스가 달라도 돌려주는 값의 의미가 같아진다.

    ---------------------------------------------------------------------
    'rolling'(채용 시 마감) 탐지 범위를 왜 좁게 잡는가
    ---------------------------------------------------------------------
    본문 전체를 뒤지면 안 된다. 같은 문구를 RX_EARLY_CLOSE(+12)가 이미 세고
    있어서 rolling(+10)까지 붙으면 한 문구로 22점이 된다.
    실측: 전체 검색이면 jobkorea 1,335행이 추가로 rolling=True가 되고,
    그중 687행(measurable의 2.25%)의 등급이 한 칸 올라간다.

    범위는 형태별로 다르게 잡는다.
      형태 A — `시작일 … 마감일 …` 매치 구간 + 뒤 40자.
               메타데이터 위치를 정확히 알고 있으므로 거기만 본다.
      형태 B — v2와 동일한 `접수기간` 뒤 90자.

    ⚠️ 형태 A에서 `접수기간` 뒤 90자(형태 B의 창)를 그대로 쓰면 안 된다.
       2-2 앱이 조립하는 헤더는 `접수기간 · 방법 시작일 X 마감일 Y`로 짧아서,
       90자 창이 **본문 앞부분까지 먹는다**. 본문 첫머리에 '채용 시 마감'이
       있으면 접수 조건이 아닌데도 rolling이 켜진다. 학습 데이터의 jobkorea는
       그 자리에 `접수방법`·`지원양식` 같은 메타데이터가 더 있어서 이 문제가
       없었다 — 앱에서만 생기는 train/serve skew라 여기서 막는다."""
    body = body or ''

    m = RX_START_END.search(body)                       # 형태 A
    if m:
        a = _mkdate(*m.group(1, 2, 3))
        b = _mkdate(*m.group(4, 5, 6))
        if a and b:
            d = (b - a).days
            if 0 <= d <= 400:
                seg = body[m.start():m.end() + 40]
                return d, bool(RX_ROLLING.search(seg))

    m = RX_PERIOD_SEG.search(body)                      # 형태 B
    if m:
        seg = m.group(1)
        rolling = bool(RX_ROLLING.search(seg))
        ds = RX_DATE.findall(seg)
        if len(ds) >= 2:
            a, b = _mkdate(*ds[0]), _mkdate(*ds[1])
            if a and b:
                d = (b - a).days
                if 0 <= d <= 400:
                    return d, rolling
        return None, rolling
    return None, False


# ---------------------------------------------------------------------------
# 3. 나머지 신호 (v2와 동일)
# ---------------------------------------------------------------------------
RX_HEADCOUNT = re.compile(r'모집인원\s*([0-9]+)\s*명')
RX_HEADCOUNT_ALT = re.compile(r'([0-9]{1,3})\s*명\s*(?:내외\s*)?(?:모집|채용|선발)')
RX_MANY = re.compile(r'다수\s*(?:모집|채용|선발)|각\s*부문\s*(?:별\s*)?(?:다수|모집)')
RX_URGENT = re.compile(r'급구|긴급\s*채용|긴급채용|시급히|서둘러')
RX_IMMEDIATE = re.compile(r'즉시\s*(?:입사|출근|근무|투입|합류)|바로\s*출근|조속히|즉시\s*채용')
RX_BACKFILL = re.compile(r'결원|충원|대체\s*인력|공석')
RX_EARLY_CLOSE = re.compile(r'조기\s*마감|마감\s*임박|충원\s*시\s*마감|채용\s*시\s*마감')
RX_BONUS = re.compile(r'합격\s*축하금|입사\s*축하금|사이닝\s*보너스|정착\s*지원금')

# 창 길이 -> 가중치. v2의 saramin 매핑을 양 소스에 그대로 적용한다.
_WINDOW_TIERS = [(3, 30, '초단기 모집'), (7, 22, '단기'),
                 (14, 12, ''), (30, 4, ''), (10 ** 9, 0, '장기')]


def extract_signals(body: str):
    """(가중치 합계, 근거 라벨 리스트).

    근거 라벨이 그대로 urgency_reason 문장이 되므로 실제 매치된 것만 담는다.
    v2와 달리 raw_text를 따로 받지 않는다 — 모든 신호를 정제된 body에서만
    읽는다. 그래야 [수정 1]의 제거가 실제로 효과를 낸다."""
    score = 0
    reasons = []

    # --- A. 접수 창 길이 (소스 통합) ---
    win, rolling = parse_application_window(body)
    if win is not None:
        for limit, w, note in _WINDOW_TIERS:
            if win <= limit:
                score += w
                reasons.append(f"접수 {win}일" + (f"({note})" if note else ""))
                break
    if rolling:
        score += 10
        reasons.append("채용 시 마감(충원되면 조기 종료)")
    elif win is None and RX_ALWAYS_OPEN.search(body):
        score += 10
        reasons.append("상시·수시 채용(지속 수요)")

    # --- B. 모집 규모 ---
    n_open = None
    mh = RX_HEADCOUNT.search(body) or RX_HEADCOUNT_ALT.search(body)
    if mh:
        try:
            n_open = int(mh.group(1))
        except ValueError:
            n_open = None
    if n_open:
        if n_open >= 10:
            score += 28
            reasons.append(f"모집인원 {n_open}명(대규모)")
        elif n_open >= 5:
            score += 20
            reasons.append(f"모집인원 {n_open}명")
        elif n_open >= 2:
            score += 11
            reasons.append(f"모집인원 {n_open}명")
    elif RX_MANY.search(body):
        score += 12
        reasons.append("다수 모집")

    # --- C. 즉시성 ---
    if RX_URGENT.search(body):
        score += 32
        reasons.append("급구·긴급 채용 명시")
    if RX_IMMEDIATE.search(body):
        score += 22
        reasons.append("즉시 입사·투입 요구")

    # --- D. 결원 대체 ---
    if RX_BACKFILL.search(body):
        score += 16
        reasons.append("결원·충원 목적")

    # --- E. 마감 압박 문구 ---
    if RX_EARLY_CLOSE.search(body):
        score += 12
        reasons.append("조기 마감 가능성 언급")

    # --- F. 보상 유인 ---
    if RX_BONUS.search(body):
        score += 12
        reasons.append("합격축하금 등 보상 유인")

    return score, reasons


# ---------------------------------------------------------------------------
# 4. 측정 가능 여부 (v2와 동일)
# ---------------------------------------------------------------------------
# "신호 없음"이 낮은 점수인지 측정 불가인지를 구분한다.
# 채용 메타데이터 섹션이 있는데 신호가 없다 -> 실제로 적극성이 낮다(1점).
# 섹션 자체가 없다 -> 관측 대상이 아니다(어휘 폴백, 미검증).
RX_META_SECTION = re.compile(r'접수\s*기간|접수기간|모집\s*인원|모집인원|마감일|남은기간')


def is_measurable(raw_text: str) -> bool:
    return bool(RX_META_SECTION.search(raw_text or ''))


# ---------------------------------------------------------------------------
# 5. 어휘 폴백 (v2와 동일 — 의도적으로 고치지 않았다)
# ---------------------------------------------------------------------------
TIER_A = {
    '급구·긴급채용': r'급구|긴급\s*채용|긴급\s*모집',
    '즉시 합류 요구': r'즉시\s*(?:합류|입사|출근|근무\s*가능)|바로\s*(?:합류|입사|출근)|조속한\s*합류|ASAP|asap',
    '충원·결원': r'충원|결원|공석|대체\s*인력',
    '마감 압박': r'조기\s*마감|채용\s*시\s*마감|상시\s*채용|수시\s*채용',
    '다수 모집': r'다수\s*(?:모집|채용)|여러\s*명\s*(?:모집|채용)|[0-9]{1,2}\s*명\s*(?:모집|채용|충원)',
}
TIER_B = {
    '급성장 조직': r'급성장|고속\s*성장|빠른\s*성장|가파른\s*성장|폭발적\s*성장|스케일\s*?업',
    '조직 신설·초기 멤버': r'신규\s*(?:팀|조직|부서)|초기\s*멤버|창립\s*멤버|파운딩\s*멤버|태스크\s*포스|조직\s*신설|팀\s*빌딩|0\s*to\s*1',
    '증원·조직 확장': r'조직\s*확(?:장|대)|인원\s*(?:확대|충원|증원)|팀\s*확(?:장|대)|채용\s*확대',
}
TIER_C = {
    '빠른 합류·투입': r'빠르게\s*(?:합류|적응|투입)|빠른\s*투입|빠른\s*온보딩',
    '속도 지향 문화': (r'빠른\s*(?:의사\s*결정|의사결정|실행|배포|이터레이션|사이클|대응|피드백|성장)|'
                  r'빠르게\s*(?:성장|실행|배포|개선|대응)|신속(?:한|히)\s*(?:대응|실행|처리|이해)|'
                  r'애자일|스프린트'),
}
_FALLBACK_TIERS = [(TIER_A, 2), (TIER_B, 1), (TIER_C, 1)]


def score_by_vocabulary(raw_text: str):
    """(1~5 점수, 근거 라벨) — 어휘가 하나도 없으면 중립 3점.

    ⚠️ 2-1에서 측정된 결과: 이 폴백은 상수(항상 3점)보다 MAE가 나쁘다
       (1.6610 vs 1.2199). 즉 실질적으로 아무 일도 하지 않는다.
       그럼에도 남겨둔 이유는 대체할 검증된 방법이 없기 때문이다.
       (모델로 대체하는 것도 근거가 없다 — 2-1 README '한계 2' 참조)"""
    text = raw_text or ''
    add, hits = 0, []
    for group, weight in _FALLBACK_TIERS:
        for name, pat in group.items():
            if re.search(pat, text):
                add += weight
                hits.append(name)
    return min(3 + add, 5), hits


# ---------------------------------------------------------------------------
# 6. 등급 · 문장 (v2와 동일)
# ---------------------------------------------------------------------------
def to_level(evidence_score: int) -> int:
    if evidence_score <= 0:
        return 1
    if evidence_score <= 14:
        return 2
    if evidence_score <= 26:
        return 3
    if evidence_score <= 42:
        return 4
    return 5


LEVEL_LABEL = {1: "매우 낮음", 2: "낮음", 3: "보통", 4: "높음", 5: "매우 높음"}


def build_reason(level, reasons, measurable):
    if not measurable:
        if not reasons:
            return (f"적극성 {LEVEL_LABEL[level]}(중립) — 마감·모집 규모 정보가 없고 "
                    f"본문에 긴급 관련 어휘도 없어 기준점 부여")
        return (f"적극성 {LEVEL_LABEL[level]} — 마감 정보는 없으나 본문 어휘에서 "
                f"긴급 정황 감지: " + " · ".join(reasons))
    if not reasons:
        return (f"적극성 {LEVEL_LABEL[level]} — 채용 메타데이터는 있으나 접수 기간·모집 규모·"
                f"즉시 입사·결원 충원 등 적극성 신호가 확인되지 않음")
    return f"적극성 {LEVEL_LABEL[level]} — " + " · ".join(reasons)


# ---------------------------------------------------------------------------
# 7. 단건 채점 — 앱과 학습이 같은 함수를 쓴다
# ---------------------------------------------------------------------------
def score_posting(raw_text: str, source: str = 'unknown') -> dict:
    """공고 하나를 채점한다.

    앱(2-2)과 라벨 생성(아래 main)이 이 함수를 공유한다. 둘이 갈라지면
    "앱이 보여주는 규칙 점수"와 "모델이 배운 라벨"이 서로 다른 것을 가리킨다."""
    raw = raw_text or ''
    body = clean_body(raw, source)
    measurable = is_measurable(raw)
    if measurable:
        evidence, reasons = extract_signals(body)
        level = to_level(evidence)
    else:
        evidence = None
        level, reasons = score_by_vocabulary(raw)
    win, rolling = parse_application_window(body)
    return {
        'urgency_score': level,
        'urgency_reason': build_reason(level, reasons, measurable),
        'evidence_score': evidence,
        'reasons': reasons,
        'measurable': measurable,
        'window_days': win,
        'rolling': rolling,
        'rule_version': RULE_VERSION,
    }


# ---------------------------------------------------------------------------
# 8. 라벨 재생성
# ---------------------------------------------------------------------------
def main(write: bool):
    import json
    from collections import Counter

    sys.stdout.reconfigure(encoding='utf-8')
    with open(SRC_PATH, encoding='utf-8') as f:
        data = json.load(f)

    out = []
    for x in data:
        r = score_posting(x.get('raw_text'), x['source'])
        rec = dict(x)
        rec['urgency_score'] = r['urgency_score']
        rec['urgency_reason'] = r['urgency_reason']
        out.append(rec)

    v2_path = DATA_DIR / "master_merged_v2.json"
    v2 = None
    if v2_path.exists():
        with open(v2_path, encoding='utf-8') as f:
            v2 = {(r['source'], r['job_id']): r['urgency_score'] for r in json.load(f)}

    print("=" * 74)
    print(f"라벨 규칙 {RULE_VERSION} 재산출  (n={len(out):,})")
    print("=" * 74)
    for src in ['jobkorea', 'saramin', 'wanted', None]:
        sub = [r for r in out if src is None or r['source'] == src]
        c = Counter(r['urgency_score'] for r in sub)
        line = '  '.join(f"{lv}:{c.get(lv, 0):>6,}" for lv in range(1, 6))
        print(f"  {src or '전체':<9} n={len(sub):>6,}  {line}")

    if v2:
        chg = Counter()
        for r in out:
            old = v2.get((r['source'], r['job_id']))
            if old is not None and old != r['urgency_score']:
                chg[(old, r['urgency_score'])] += 1
        tot = sum(chg.values())
        print()
        print(f"  v2 대비 등급 변경: {tot:,}행 ({tot / len(out) * 100:.1f}%)")
        for k, v in sorted(chg.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {k[0]}점 -> {k[1]}점 : {v:>6,}")

    if write:
        with open(OUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n저장: {OUT_PATH}  ({len(out):,} rows)")
    else:
        print("\n(--write 를 붙이면 master_merged_v3.json 으로 저장됩니다)")
    return out


if __name__ == '__main__':
    main('--write' in sys.argv)
