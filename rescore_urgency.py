"""
rescore_urgency.py

master_merged.json의 urgency_score / urgency_reason을 raw_text 근거 기반으로 재산출한다.

[왜 재정의했는가]
기존 점수는 84.7%가 3점에 쏠려 사실상 변별력이 없었다. raw_text를 실측해보니
'급구'(0.06%), '충원/결원'(0.62%) 등 전통적인 긴급성 어휘는 거의 존재하지 않아
"시급성"을 그대로 복원하는 것은 불가능했다. 대신 본문에서 실제로 관측 가능한
  - 마감 임박도, 모집 규모, 즉시 입사 요구, 결원 대체, 보상 유인
을 묶어 **채용 적극성(hiring aggressiveness)** 지표로 재정의한다.

[측정 시 주의한 것]
1) jobkorea raw_text는 페이지 전체 덤프라 하단 '추천공고' 목록이 섞여 있다.
   이를 제거하지 않으면 다른 공고의 문구를 이 공고의 신호로 오인한다.
   (예: '합격축하금' 8,457건 중 8,435건이 추천공고 목록의 허수였음)
2) '마감일은 기업의 사정으로 조기 마감될 수 있습니다'는 모든 공고에 붙는
   정형 문구이므로 '조기마감' 신호에서 제외한다.
3) ⚠️ '신입가능/경력무관'은 신호에서 의도적으로 제외했다.
   experience_level과 사실상 동일한 변수라, 대시보드의
   "연차 구간별 urgency 비교" 차트에서 순환논리(spurious correlation)를 만든다.
4) 원본 jobkorea의 '마감일' 컬럼은 이름과 달리 등록일('2일 전 등록')이었다.
   실제 마감 정보는 raw_text의 '남은기간 N일' / '마감일 상시채용'에서 파싱한다.

실행: python rescore_urgency.py [--write]
      --write 없이 실행하면 분포만 리포트하고 파일은 쓰지 않는다.
"""

import datetime
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = Path(__file__).parent / "data"
SRC_PATH = DATA_DIR / "master_merged.json"
OUT_PATH = DATA_DIR / "master_merged_v2.json"


# ---------------------------------------------------------------------------
# 1. 본문 정제
# ---------------------------------------------------------------------------
_JK_TAIL_MARKERS = ['본 채용정보는', '로그인 하고 비슷한 조건의', '관련 태그']
_JK_DISCLAIMER = re.compile(
    r'마감일은\s*기업의\s*사정으로\s*인해\s*조기\s*마감\s*또는\s*변경될\s*수\s*있습니다'
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
    return _JK_DISCLAIMER.sub(' ', body)


# ---------------------------------------------------------------------------
# 2. 신호 추출
# ---------------------------------------------------------------------------
RX_DAYS_LEFT = re.compile(r'남은기간\s*([0-9]+)\s*일')
# saramin은 '남은기간'이 없는 대신 '접수기간 시작~종료'가 있다.
# 접수 창(window) 길이는 기준일 없이도 계산되는 시급성 신호다.
# (3일 안에 닫는 공고 vs 30일 열어두는 공고는 명백히 적극성이 다르다)
RX_PERIOD_SEG = re.compile(r'접수\s*기간(.{0,90})', re.S)
RX_DATE = re.compile(r'(20[0-9]{2})\s*[-.년]\s*([0-9]{1,2})\s*[-.월]\s*([0-9]{1,2})\s*일?')
RX_ROLLING = re.compile(r'채용\s*시\s*마감|채용시까지|채용\s*시\s*까지')


def parse_application_window(text: str):
    """(접수창 일수 or None, 채용시마감 여부)"""
    m = RX_PERIOD_SEG.search(text or '')
    if not m:
        return None, False
    seg = m.group(1)
    rolling = bool(RX_ROLLING.search(seg))
    ds = RX_DATE.findall(seg)
    if len(ds) >= 2:
        try:
            a = datetime.date(int(ds[0][0]), int(ds[0][1]), int(ds[0][2]))
            b = datetime.date(int(ds[1][0]), int(ds[1][1]), int(ds[1][2]))
            d = (b - a).days
            if 0 <= d <= 400:
                return d, rolling
        except ValueError:
            pass
    return None, rolling
RX_ALWAYS_OPEN = re.compile(r'마감일\s*상시채용|상시\s*채용|수시\s*채용|상시\s*모집|연중\s*수시')
RX_HEADCOUNT = re.compile(r'모집인원\s*([0-9]+)\s*명')
RX_HEADCOUNT_ALT = re.compile(r'([0-9]{1,3})\s*명\s*(?:내외\s*)?(?:모집|채용|선발)')
RX_MANY = re.compile(r'다수\s*(?:모집|채용|선발)|각\s*부문\s*(?:별\s*)?(?:다수|모집)')
RX_URGENT = re.compile(r'급구|긴급\s*채용|긴급채용|시급히|서둘러')
RX_IMMEDIATE = re.compile(r'즉시\s*(?:입사|출근|근무|투입|합류)|바로\s*출근|조속히|즉시\s*채용')
RX_BACKFILL = re.compile(r'결원|충원|대체\s*인력|공석')
RX_EARLY_CLOSE = re.compile(r'조기\s*마감|마감\s*임박|충원\s*시\s*마감|채용\s*시\s*마감')
RX_BONUS = re.compile(r'합격\s*축하금|입사\s*축하금|사이닝\s*보너스|정착\s*지원금')


def extract_signals(body: str, raw_text: str):
    """(가중치 합계, 근거 라벨 리스트)를 돌려준다.
    근거 라벨은 그대로 urgency_reason 문장이 되므로, 실제 매치된 것만 담는다."""
    score = 0
    reasons = []

    # --- A. 마감 임박도 (jobkorea만 구조화 데이터 보유) ---
    m = RX_DAYS_LEFT.search(raw_text or "")
    if m:
        d = int(m.group(1))
        if d <= 2:
            score += 45; reasons.append(f"마감 {d}일 전(임박)")
        elif d <= 5:
            score += 35; reasons.append(f"마감 {d}일 전")
        elif d <= 10:
            score += 22; reasons.append(f"마감 {d}일 전")
        elif d <= 20:
            score += 12; reasons.append(f"마감 {d}일 전")
        elif d <= 40:
            score += 5; reasons.append(f"마감 {d}일 전(여유)")
        else:
            reasons.append(f"마감 {d}일 전(장기)")
    else:
        # jobkorea 외 소스: '접수기간 시작~종료'의 창 길이로 대체 측정
        win, rolling = parse_application_window(body)
        if win is not None:
            if win <= 3:
                score += 30; reasons.append(f"접수 {win}일(초단기 모집)")
            elif win <= 7:
                score += 22; reasons.append(f"접수 {win}일(단기)")
            elif win <= 14:
                score += 12; reasons.append(f"접수 {win}일")
            elif win <= 30:
                score += 4; reasons.append(f"접수 {win}일")
            else:
                reasons.append(f"접수 {win}일(장기)")
        if rolling:
            score += 10; reasons.append("채용 시 마감(충원되면 조기 종료)")
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
    if n_open is not None and n_open > 0:
        if n_open >= 10:
            score += 28; reasons.append(f"모집인원 {n_open}명(대규모)")
        elif n_open >= 5:
            score += 20; reasons.append(f"모집인원 {n_open}명")
        elif n_open >= 2:
            score += 11; reasons.append(f"모집인원 {n_open}명")
    elif RX_MANY.search(body):
        score += 12
        reasons.append("다수 모집")

    # --- C. 즉시성 ---
    if RX_URGENT.search(body):
        score += 32; reasons.append("급구·긴급 채용 명시")
    if RX_IMMEDIATE.search(body):
        score += 22; reasons.append("즉시 입사·투입 요구")

    # --- D. 결원 대체 ---
    if RX_BACKFILL.search(body):
        score += 16; reasons.append("결원·충원 목적")

    # --- E. 마감 압박 문구 ---
    if RX_EARLY_CLOSE.search(body):
        score += 12; reasons.append("조기 마감 가능성 언급")

    # --- F. 보상 유인 ---
    if RX_BONUS.search(body):
        score += 12; reasons.append("합격축하금 등 보상 유인")

    return score, reasons


# ---------------------------------------------------------------------------
# 3. 측정 가능 여부 판별
# ---------------------------------------------------------------------------
# ⚠️ 핵심 원칙: "신호 없음"이 낮은 점수인지 측정 불가인지를 구분한다.
#    공고 본문에 '접수기간/모집인원/마감일/지원방법' 같은 채용 메타데이터 섹션이
#    있는데도 적극성 신호가 없다면 -> 실제로 적극성이 낮은 것(=1점).
#    섹션 자체가 없다면 -> 애초에 관측 대상이 아니므로 측정 불가(=NaN).
#
#    ⚠️ 판정 기준은 '마감/접수/모집인원'으로 좁혔다. '근무조건·지원방법'까지 인정하면
#       saramin 87%가 통과하지만, 그중 상당수는 마감 정보가 전혀 없어 시급성을
#       판단할 근거가 실제로는 없다(느슨한 기준 88.3% vs 엄격 79.5%).
#    실측 보유율(엄격): jobkorea 99.1% / saramin 79.5% / wanted 0.0%
#    (wanted의 raw_text는 main_tasks+requirements+preferred 조합이라
#     공고 메타데이터가 구조적으로 존재하지 않는다.)
RX_META_SECTION = re.compile(
    r'접수\s*기간|접수기간|모집\s*인원|모집인원|마감일|남은기간'
)


def is_measurable(raw_text: str) -> bool:
    return bool(RX_META_SECTION.search(raw_text or ''))


# ---------------------------------------------------------------------------
# 3-1. 메타데이터가 없는 공고(주로 wanted)용 어휘 기반 폴백
# ---------------------------------------------------------------------------
# wanted의 raw_text는 직무기술서라 마감일·모집인원이 없다. 대신 본문에서
# '긴급한 느낌'을 주는 어휘를 티어별로 세어 중립 3점에서 가산한다.
#
# ⚠️ 이건 검증된 채용 시급성이 아니라 '어조 프록시'다. reason에 그렇게 명시한다.
# ⚠️ 오탐 제거 이력:
#    - 'T/O' 패턴은 "CTO"에 매칭되어 삭제
#    - '시급'은 "고객의 가장 시급한 과제"처럼 업무 설명이라 채용 문맥 외 삭제
#    - '셋업'은 "기술 셋업을 주도"처럼 업무라 조직/팀 한정으로 축소
#    - 'MVP·런칭'은 제품 용어라 삭제
#
# 점수: 3(중립) + TierA×2 + TierB×1 + TierC×1, 최대 5. 어휘가 없으면 3점.
TIER_A = {   # 채용 긴급성 직결
    '급구·긴급채용': r'급구|긴급\s*채용|긴급\s*모집',
    '즉시 합류 요구': r'즉시\s*(?:합류|입사|출근|근무\s*가능)|바로\s*(?:합류|입사|출근)|조속한\s*합류|ASAP|asap',
    '충원·결원': r'충원|결원|공석|대체\s*인력',
    '마감 압박': r'조기\s*마감|채용\s*시\s*마감|상시\s*채용|수시\s*채용',
    '다수 모집': r'다수\s*(?:모집|채용)|여러\s*명\s*(?:모집|채용)|[0-9]{1,2}\s*명\s*(?:모집|채용|충원)',
}
TIER_B = {   # 조직 확장·신설 정황
    '급성장 조직': r'급성장|고속\s*성장|빠른\s*성장|가파른\s*성장|폭발적\s*성장|스케일\s*?업',
    '조직 신설·초기 멤버': r'신규\s*(?:팀|조직|부서)|초기\s*멤버|창립\s*멤버|파운딩\s*멤버|태스크\s*포스|조직\s*신설|팀\s*빌딩|0\s*to\s*1',
    '증원·조직 확장': r'조직\s*확(?:장|대)|인원\s*(?:확대|충원|증원)|팀\s*확(?:장|대)|채용\s*확대',
}
TIER_C = {   # 속도 지향 어조 (가장 약한 신호)
    '빠른 합류·투입': r'빠르게\s*(?:합류|적응|투입)|빠른\s*투입|빠른\s*온보딩',
    '속도 지향 문화': (r'빠른\s*(?:의사\s*결정|의사결정|실행|배포|이터레이션|사이클|대응|피드백|성장)|'
                  r'빠르게\s*(?:성장|실행|배포|개선|대응)|신속(?:한|히)\s*(?:대응|실행|처리|이해)|'
                  r'애자일|스프린트'),
}
_FALLBACK_TIERS = [(TIER_A, 2), (TIER_B, 1), (TIER_C, 1)]


def score_by_vocabulary(raw_text: str):
    """(1~5 점수, 근거 라벨) — 어휘가 하나도 없으면 중립 3점."""
    text = raw_text or ''
    add = 0
    hits = []
    for group, weight in _FALLBACK_TIERS:
        for name, pat in group.items():
            if re.search(pat, text):
                add += weight
                hits.append(name)
    return min(3 + add, 5), hits


# ---------------------------------------------------------------------------
# 4. 점수 구간 (근거점수 -> 1~5)
#    분위수 강제 배분이 아니라 '절대 근거량' 기준이라, 점수 자체가 의미를 갖는다.
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


def build_reason(level, reasons, measurable, source):
    if not measurable:
        # 어휘 기반 폴백: 근거가 '검증된 마감/규모'가 아니라 '어조'임을 문장에 드러낸다
        if not reasons:
            return (f"적극성 {LEVEL_LABEL[level]}(중립) — 마감·모집 규모 정보가 없고 "
                    f"본문에 긴급 관련 어휘도 없어 기준점 부여")
        return (f"적극성 {LEVEL_LABEL[level]} — 마감 정보는 없으나 본문 어휘에서 "
                f"긴급 정황 감지: " + " · ".join(reasons))

    if not reasons:
        return (f"적극성 {LEVEL_LABEL[level]} — 채용 메타데이터는 있으나 마감 임박·모집 규모·"
                f"즉시 입사·결원 충원 등 적극성 신호가 확인되지 않음")

    return f"적극성 {LEVEL_LABEL[level]} — " + " · ".join(reasons)


# ---------------------------------------------------------------------------
# 5. 실행
# ---------------------------------------------------------------------------
def main(write: bool):
    with open(SRC_PATH, encoding='utf-8') as f:
        data = json.load(f)

    enriched = []
    for x in data:
        body = clean_body(x.get('raw_text') or '', x['source'])
        score, reasons = extract_signals(body, x.get('raw_text') or '')
        enriched.append((x, score, reasons))

    # --- 연속 점수 분포 확인 ---
    print("=" * 74)
    print("STEP 1. 연속 근거점수(0~) 분포")
    print("=" * 74)
    for src in ['wanted', 'jobkorea', 'saramin', None]:
        sub = [s for x, s, _ in enriched if src is None or x['source'] == src]
        vals = sorted(sub)
        n = len(vals)
        q = lambda p: vals[min(int(n * p), n - 1)]
        zero = sum(1 for v in vals if v == 0)
        label = src or '전체'
        print(f"  {label:<9} n={n:>6,}  0점={zero:>6,}({zero/n*100:5.1f}%)  "
              f"p50={q(.50):>3}  p75={q(.75):>3}  p90={q(.90):>3}  max={vals[-1]:>3}")

    print()
    print("  근거점수 히스토그램(전체):")
    hist = Counter(s for _, s, _ in enriched)
    for bucket in [(0, 0), (1, 10), (11, 20), (21, 30), (31, 40), (41, 55), (56, 70), (71, 200)]:
        lo, hi = bucket
        c = sum(v for k, v in hist.items() if lo <= k <= hi)
        bar = '█' * int(c / len(enriched) * 60)
        print(f"    {lo:>3}-{hi:>3}: {c:>6,} ({c/len(enriched)*100:5.1f}%) {bar}")

    # --- 등급 배정 ---
    out = []
    for x, score, reasons in enriched:
        raw = x.get('raw_text') or ''
        measurable = is_measurable(raw)
        if measurable:
            level = to_level(score)
        else:
            # NaN을 남기지 않고 어휘 기반으로 3~5점 부여
            level, reasons = score_by_vocabulary(raw)
        rec = dict(x)
        rec['urgency_score'] = level
        rec['urgency_reason'] = build_reason(level, reasons, measurable, x['source'])
        out.append(rec)

    old_dist = Counter(x.get('urgency_score') for x in data)
    new_dist = Counter(r['urgency_score'] for r in out)

    print()
    print("=" * 74)
    print("STEP 2. 재산출 결과")
    print("=" * 74)
    print(f"  {'등급':<8}{'기존':>10}{'':>4}{'신규':>10}")
    for lv in [1, 2, 3, 4, 5, None]:
        o, n = old_dist.get(lv, 0), new_dist.get(lv, 0)
        name = 'NaN(측정불가)' if lv is None else f'{lv}점'
        print(f"  {name:<12}{o:>8,} ({o/len(data)*100:5.2f}%) "
              f"{n:>8,} ({n/len(out)*100:5.2f}%)")

    print()
    print("  === source별 신규 분포 ===")
    for src in ['wanted', 'jobkorea', 'saramin']:
        sub = [r for r in out if r['source'] == src]
        c = Counter(r['urgency_score'] for r in sub)
        line = '  '.join(f"{lv if lv else 'NaN'}:{c.get(lv,0):>6,}" for lv in [1, 2, 3, 4, 5, None])
        print(f"    {src:<9} (n={len(sub):>6,})  {line}")

    # 측정 가능한 행만 놓고 본 분포 (실제 대시보드에 보이는 모습)
    scored = [r for r in out if r['urgency_score'] is not None]
    print()
    print(f"  === 측정 가능 행만 (n={len(scored):,}, 전체의 {len(scored)/len(out)*100:.1f}%) ===")
    cs = Counter(r['urgency_score'] for r in scored)
    for lv in [1, 2, 3, 4, 5]:
        c = cs.get(lv, 0)
        bar = '█' * int(c / len(scored) * 50)
        print(f"    {lv}점: {c:>6,} ({c/len(scored)*100:5.1f}%) {bar}")

    print()
    print("  === urgency_reason 샘플 ===")
    seen = set()
    for r in out:
        lv = r['urgency_score']
        if lv not in seen:
            seen.add(lv)
            print(f"    [{lv}] {r['source']:<9} {r['urgency_reason'][:96]}")
        if len(seen) == 6:
            break

    if write:
        with open(OUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print()
        print(f"저장 완료: {OUT_PATH}  ({len(out):,} rows)")
    else:
        print()
        print("(--write 를 붙이면 data/master_merged_v2.json 으로 저장됩니다)")

    return out


if __name__ == '__main__':
    main(write='--write' in sys.argv)
