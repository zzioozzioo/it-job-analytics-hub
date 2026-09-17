"""
urgency_rule.py  —  채용 적극성 라벨 규칙 v4

2-1(모델 학습)과 2-2(Streamlit 앱)가 함께 쓰는 단일 출처.

v3에서 v2(`rescore_urgency.py`)의 실측된 결함 두 개를 고쳤고([수정 1][수정 2]),
v4에서 두 개를 더 고쳤다 — 이중 계상([수정 3])과 어휘 폴백 재보정([수정 4]).

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
[수정 3] '채용 시 마감' 한 문구가 두 신호에 이중 계상되던 것  (v4)
---------------------------------------------------------------------------
`채용 시 마감`은 RX_ROLLING(+10)과 RX_EARLY_CLOSE(+12) 양쪽 정규식에 모두
들어 있어서, 한 문구가 22점이 됐다.

v3이 이걸 몰랐던 게 아니다. rolling 탐지 범위를 본문 전체에서 접수 조건
블록으로 좁힌 이유가 바로 이 이중 계상이었다(parse_application_window 주석).
다만 그건 **앵커 밖** 매치만 막았고, 앵커 **안**에 문구가 있는 공고에서는
겹침이 그대로 남았다. 즉 v3의 조치는 절반이었다.

실측(measurable 30,569행):
    rolling과 early_close 동시 발동            2,838행 (9.3%)
      그중 early_close 근거가 그 문구뿐        2,781행 (98.0%)
      그중 '조기 마감' 등 다른 근거도 있음        57행

v4는 **마감 압박을 한 번만 센다.** early_close가 발동하면 rolling은 점수를
얹지 않는다(근거 문장은 rolling 쪽 표현을 쓴다 — 더 구체적이므로).
남길 쪽으로 early_close(+12)를 고른 것은 v3 주석이 "같은 문구를
RX_EARLY_CLOSE가 이미 세고 있다"고 그쪽을 담당으로 전제했기 때문이다.

라벨 영향: measurable 30,569행 중 1,993행(6.5%)의 등급이 내려간다.
    3점 -> 2점  1,551행    4점 -> 3점  280행    5점 -> 4점  162행

⚠️ 이 수정은 소스 간 라벨 정합성 지표를 **악화시킨다.** 숨기지 않고 적는다.

    상위 등급(4+5) 비율    jobkorea   saramin   격차
    v3                        5.4%      2.0%    2.7배
    v4                        4.8%      0.9%    5.2배

v3 README가 "22.9배 -> 2.7배로 줄었다"를 성과로 적었는데, 그 2.7배의 일부는
**양쪽 소스에 똑같이 걸려 있던 이 버그가 떠받치고 있었다.** 잘못된 +10점이
saramin의 한계 공고들을 3점으로 밀어올리고 있었고, 그걸 걷어내니 saramin의
상위 비율이 절반 이하로 내려갔다. 후보 3안(rolling 제거 / 문구 단독일 때만
제거 / early_close에서 문구 제거)을 모두 재봤는데 셋 다 5.2~5.3배로 같으므로,
이건 설계 선택의 부작용이 아니라 버그를 걷어낸 결과다.

정합성 지표가 나빠진 것과 라벨이 나빠진 것은 다르다. 다만 v3 README·02
README의 "2.7배" 서술은 v4 재학습 시점에 함께 고쳐야 한다.

---------------------------------------------------------------------------
[수정 4] 어휘 폴백 재보정 — 기준점과 보폭  (v4)
---------------------------------------------------------------------------
메타데이터가 없는 공고(24.2%)는 규칙이 계산할 수 없어 어휘만 보고 등급을
붙인다. v3까지 그 식이 `min(3 + 가산, 5)`였고, 2-1 README '한계 1'이
**"상수보다 MAE가 나쁘다"** 고 실측해 두었다. 남겨둔 이유는 "대체할 검증된
방법이 없어서"였는데, 그 문장이 v4에서 유효기간이 끝났다.

새 신호도 새 어휘도 추가하지 않는다. 숫자 두 개만 바꿨다.

    v3   level = min(3 + 가산, 5)
    v4   level = clip(2 + 가산 x 0.25, 1, 5)

진단은 v3이 전이 모델을 두고 한 말과 같다 — **"순서는 배웠지만 점 예측
(calibration)이 나쁘다."** QWK가 0이 아니니 순서 신호는 있고, MAE가 상수보다
나쁘니 그 순서를 엉뚱한 눈금 위에 올려놓은 것이다. 기준점이 너무 높고
보폭이 너무 컸다.

**측정 (measurable hold-out 6,056행, `2-1-.../calibrate_fallback.py`)**

    방법                        MAE       QWK
    상수 (2점)                 0.4747    0.0000
    v3 폴백 (3 + 가산)          2.0591    0.0874
    v4 폴백 (2 + 가산 x 0.25)   0.6932    0.2674

v3 폴백을 **두 지표 모두에서** 이긴다(MAE -66%, QWK +0.18). 상수를 MAE로
이기지는 못하지만, '완전히 지배당하는' 위치에서 'MAE를 조금 내주고 순서
정보를 얻는' 위치로 옮겨간다. 남길 근거가 처음으로 생긴 셈이다.

**BASE는 프록시로 고르지 않았다 — 이게 이 수정의 핵심 판단이다**

SCALE(0.25)은 validation이 골랐다. 그런데 같은 규칙을 그대로 적용하면
BASE=1이 뽑히는데, 그건 쓰면 안 된다.

    폴백이 실제로 적용되는 unmeasurable 9,779행 중
    8,144행(83.3%)은 어휘 가산점이 **0**이다.

즉 폴백은 재보정 전에도 후에도 사실상 상수이고, 파라미터는 "그 더미를 몇
점에 쌓을 것인가"를 정할 뿐이다. BASE=1이면 91%가 1점(매우 낮음)에 쌓인다.
그건 2-1 README '한계 2'가 *"1점에 79.8% 쏠리는데, 이는 label shift이지 그
공고들이 덜 급하다는 증거가 아니다"* 라며 이미 기각한 모양이다. 모델로 하면
안 된다고 적어놓고 폴백으로 같은 일을 할 수는 없다.

**BASE는 "관측할 수 없는 공고의 사전 확률"이고, 프록시(measurable)는 정의상
그 질문에 답할 수 없다.** 두 모집단을 가르는 기준이 바로 "메타데이터가
있느냐"이기 때문이다. 그래서 BASE는 measurable train 라벨의 **중앙값(2)**을
쓴다 — v3이 상수 베이스라인을 "항상 3점"에서 "train 중앙값"으로 바꾼 것과
같은 규약이다.

**이 수정이 실제로 고치는 것**

v3 폴백은 `급성장`·`애자일` 같은 단어 몇 개만으로 unmeasurable 877행에
5점(매우 높음), 758행에 4점(높음)을 붙이고 있었다. v4에서는 상위 등급이
사라진다.

    unmeasurable 9,779행     1점     2점     3점    4점   5점
    v3 (3 + 가산)              0      0   8,144   758  877
    v4 (2 + 가산 x 0.25)       0  8,902     874     3    0

⚠️ 그래도 이건 **검증된 라벨이 아니다.** "이 어휘가 있으면 실제로 더 적극적인
   채용인가"는 여전히 확인된 적이 없다. 덜 틀리게 만든 것이지 맞게 만든 것이
   아니다.

---------------------------------------------------------------------------
바뀌지 않은 것
---------------------------------------------------------------------------
- 신호 가중치, 등급 경계(to_level), measurable 판정 기준
- 어휘 사전 자체(TIER_A/B/C)와 가산점 계산 — [수정 4]는 가산점을 등급으로
  옮기는 두 숫자만 건드렸다.
- rolling 탐지 범위 — 형태 B(saramin)는 v2와 동일한 `접수기간` 뒤 90자,
  형태 A(jobkorea)는 그보다 좁게. [수정 3]은 범위가 아니라 **계상 횟수**만
  건드렸다(`parse_application_window` 주석 참조).

라벨 변경 (40,348행 기준, 위치 기준 대조)
    v2 -> v3   5,683행 (14.1%)
    v3 -> v4  11,772행 (29.2%)
                 [수정 3]  1,993행  measurable에서만. 전부 하락
                           3->2 1,551 / 4->3 280 / 5->4 162
                 [수정 4]  9,779행  unmeasurable 전체(기준점이 3->2로 바뀌므로)
                           3->2 8,144 / 5->3 874 / 4->2 758 / 5->4 3
    v2 -> v4  16,906행 (41.9%)

두 수정은 서로 겹치지 않는다 — [수정 3]은 measurable, [수정 4]는 unmeasurable
경로만 건드린다. 그래서 1,993 + 9,779 = 11,772로 정확히 맞는다.

⚠️ v3 README·02 README에 적힌 "v2 대비 5,694행"은 11행 과다 계산이다.
   비교를 (source, job_id) dict로 했는데 이 데이터셋에는 같은 키가
   777종(1,554행) 중복돼 있어 서로 다른 공고가 짝지어졌다. main()의
   load_labels() 주석 참조. 비율(14.1%)은 바뀌지 않는다.

실행: python urgency_rule.py            분포 리포트만
      python urgency_rule.py --write    data/master_merged_v4.json 생성
"""

import datetime
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                            # it-job-analytics-hub/
sys.path.insert(0, str(ROOT))
from common.hf_data import (MASTER, MASTER_V2, MASTER_V3,  # noqa: E402
                            MASTER_V4, fetch as hf_fetch, local_path)

OUT_PATH = local_path(MASTER_V4)      # 경로 규약은 common/hf_data 가 정한다

RULE_VERSION = "v4"


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


# ---------------------------------------------------------------------------
# 2-1. 상시·수시채용 — 앵커 없이 찾으면 안 된다
# ---------------------------------------------------------------------------
# ⚠️ `RX_ALWAYS_OPEN`을 본문 전체에 그냥 `.search()` 하지 말 것.
#    이 파일에는 이미 같은 교훈이 두 번 적혀 있다(RX_ROLLING의 앵커 세 번 시도,
#    `parse_application_window`의 형태 A 40자 창). 그런데 RX_ALWAYS_OPEN에는
#    그 교훈이 적용된 적이 없어서, 이 정규식만 여전히 맨몸으로 노출돼 있었다.
#
# 본문 전체에서 찾으면 **공고 본인의 접수 조건이 아닌 것**이 대량으로 걸린다.
# 40,348행 실측 — `상시채용`류가 본문 어딘가에 등장하는 행:
#
#     source      n        본문 전체    앵커+가드     제거된 오탐
#     jobkorea    10,025      8,202       1,363          6,839
#     saramin     25,950        502         250            252
#     wanted       4,373          0           0              0
#
# jobkorea가 8,202건(81.8%)인 것은 이 공고가 상시채용이라서가 아니다.
# 본문 꼬리의 **추천공고 목록(= 다른 회사 공고)** 을 긁은 것이다:
#
#     … AI추천공고를 확인해 보세요! ㈜퍼플페퍼 [퍼플페퍼] 기술본부 채용 …
#       프론트엔드개발자 상시채용 즉시 지원 웨버씨엔에스 …
#                        ^^^^^^^^ 남의 공고의 접수 조건
#
# saramin은 꼬리가 없지만 다른 경로로 샌다 — **개인정보 활용 동의 문구**다:
#
#     … 개인정보 보호법 … 에 의거하여 당사 상시 채용을 위한 용도로만 사용됩니다 …
#                                        ^^^^^^^^^ 채용 방식이 아니라 약관
#
# 그래서 "접수 조건이 적히는 자리"에서만 찾고, 약관 문맥은 배제한다.
# 앵커 넷은 서로 독립인데 jobkorea에서 SEG와 DLN이 1,263건으로 **정확히 일치**한다
# (v3의 rolling 앵커에서 시도 1과 시도 3이 1,335건으로 일치한 것과 같은 성격의
#  교차검증이다). 무작위 10건 표본 검증에서 오탐 0건.
_AO_ANCHORS = (
    RX_PERIOD_SEG,                                  # 접수기간 + 90자
    re.compile(r'마감일(.{0,20})', re.S),           # jobkorea: "마감일 상시채용"
    re.compile(r'본\s*공고는(.{0,40})', re.S),      # saramin: "본 공고는 수시채용으로…"
)
_AO_HEAD = 150                                       # 제목 영역: "㈜안랩 … 연구소 상시채용"
# 제목 영역의 끝. jobkorea 본문은 `… 상세요강 접수기간∙방법 기업정보 추천공고 …`
# 탭 바로 제목이 끝나고 그 뒤부터 남의 공고가 시작된다. 머리 창을 150자로만
# 자르면 짧은 공고에서 꼬리 첫머리를 먹으므로, 이 표지에서 한 번 더 끊는다.
_AO_HEAD_END = re.compile(r'상세요강|추천공고')
# 약관·동의 문구. 이 문맥의 '상시 채용'은 채용 방식이 아니라 개인정보 이용 목적이다.
_AO_DENY = re.compile(r'개인정보|보호법|신용정보|동의')


def _ao_in(window: str) -> bool:
    return bool(RX_ALWAYS_OPEN.search(window)) and not _AO_DENY.search(window)


def is_always_open(body: str) -> bool:
    """상시·수시채용 공고인가 — **접수 조건 필드 안에서만** 판정한다.

    `RX_ALWAYS_OPEN.search(body)`를 직접 쓰는 대신 이 함수를 쓸 것.
    정규식을 그대로 노출해 두면 호출부가 본문 전체를 긁게 되고, 위 표처럼
    jobkorea에서 6,839건이 남의 공고 때문에 상시채용으로 둔갑한다.

    앵커를 여러 개 두고 **전부 순회**하는 것이 중요하다. jobkorea 본문에는
    `접수기간`이 두 번 나오고(상단 탭 레이블 `상세요강 접수기간∙방법 기업정보`가
    먼저 걸린다) 첫 매치만 보면 진짜 메타데이터 블록을 놓친다 — v3에서 rolling
    앵커가 0행을 내던 것과 같은 함정이다."""
    body = body or ''
    for rx in _AO_ANCHORS:
        for m in rx.finditer(body):
            if _ao_in(m.group(1)):
                return True
    head = body[:_AO_HEAD]
    end = _AO_HEAD_END.search(head)
    return _ao_in(head[:end.start()] if end else head)


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

    # [수정 3] 마감 압박은 한 번만 센다. E가 이 문구를 이미 세는지 먼저 확정하고
    # 들어간다 — A의 rolling과 E의 early_close가 같은 `채용 시 마감`에 둘 다
    # 걸려 한 문구로 22점이 되던 것을 막는다. 상세는 모듈 상단 [수정 3].
    early_close = bool(RX_EARLY_CLOSE.search(body))

    # --- A. 접수 창 길이 (소스 통합) ---
    win, rolling = parse_application_window(body)
    if win is not None:
        for limit, w, note in _WINDOW_TIERS:
            if win <= limit:
                score += w
                reasons.append(f"접수 {win}일" + (f"({note})" if note else ""))
                break
    if rolling:
        # E가 같은 신호를 세지 않을 때만 여기서 센다.
        # (`채용시까지`처럼 RX_EARLY_CLOSE에 없는 표현으로 rolling이 켜진 경우)
        if not early_close:
            score += 10
            reasons.append("채용 시 마감(충원되면 조기 종료)")
    elif win is None and RX_ALWAYS_OPEN.search(body):
        # ⚠️ 알려진 결함 — v5 후보. 여기만 앵커 없이 본문 전체를 본다.
        # 이 분기는 corpus 40,348행 중 2,474행에서 발동하는데, 그중 1,041행
        # (42.1%)은 공고 본인의 접수 조건이 아니다 — is_always_open()의 앵커를
        # 통과하지 못한다(jobkorea 869행/40.6% · saramin 172행/51.8%).
        # 즉 남의 공고나 약관 문구 때문에 +10점이 붙고 있다.
        # 고치려면 `is_always_open(body)`로 바꾸면 된다 — 한 줄이다.
        #
        # 그런데 그 한 줄이 v4 라벨을 바꾼다. 라벨이 바뀌면 models_v4/ 재학습과
        # reference_stats.json 재생성이 따라야 하고(CLAUDE.md의 재현 순서),
        # 02·2-1·2-2 README의 성능 수치가 전부 무효가 된다. 이 저장소는 규칙
        # 변경을 라운드로 묶어 처리해 왔으므로(v3의 [수정 1·2], v4의 [수정 3·4])
        # 여기서 조용히 끼워넣지 않는다. v5 라운드에서 [수정 5]로 다룬다.
        #
        # 03(`job_pool.py`)은 라벨이 아니라 status를 계산하므로 이 제약이 없다.
        # 그쪽은 이미 `is_always_open()`을 쓴다.
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

    # --- E. 마감 압박 문구 ---  ([수정 3] rolling과 합쳐 한 번만 계상)
    if early_close:
        score += 12
        # 근거 문장은 rolling 쪽 표현을 우선한다. 접수 조건 필드에서 읽은
        # 것이므로 "조기 마감 가능성"보다 무엇을 봤는지가 분명하다.
        reasons.append("채용 시 마감(충원되면 조기 종료)" if rolling
                       else "조기 마감 가능성 언급")

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


# [수정 4] 폴백 재보정 파라미터.  level = clip(BASE + 가산 x SCALE, 1, 5)
#
# v3까지는 BASE=3, SCALE=1.0 이었고 그 조합이 상수보다 MAE가 나빴다.
# 새 신호도 새 어휘도 추가하지 않고 이 두 숫자만 바꿨다.
#
#   SCALE  validation이 골랐다 (`2-1-.../calibrate_fallback.py`).
#          "어휘 하나가 등급을 얼마나 움직여야 하는가"는 보폭 문제라
#          프록시가 답할 수 있다.
#   BASE   프록시로 고르지 **않았다.** measurable train 라벨의 중앙값을 쓴다.
#          프록시를 그대로 따르면 BASE=1이 뽑히는데, unmeasurable 9,779행 중
#          83.3%가 어휘 가산 0점이라 그 더미가 통째로 1점(매우 낮음)에 쌓인다.
#          그건 2-1 README '한계 2'가 "label shift이지 그 공고들이 덜 급하다는
#          증거가 아니다"라며 이미 기각한 모양이다. BASE는 '관측할 수 없는
#          공고의 사전 확률'이고 프록시(measurable)는 정의상 그 질문에 답할 수
#          없다. 상세는 calibrate_fallback.py 상단.
FALLBACK_BASE = 2
FALLBACK_SCALE = 0.25


def vocabulary_evidence(raw_text: str):
    """(가산점, 근거 라벨) — 어휘 신호만 뽑는다. 등급 변환은 하지 않는다.

    등급 변환과 분리해 둔 이유: 재보정(BASE·SCALE)을 밖에서 실험하려면
    가산점 자체는 건드리지 않은 채로 꺼내 쓸 수 있어야 한다."""
    text = raw_text or ''
    add, hits = 0, []
    for group, weight in _FALLBACK_TIERS:
        for name, pat in group.items():
            if re.search(pat, text):
                add += weight
                hits.append(name)
    return add, hits


def apply_fallback_scale(add, base=None, scale=None) -> int:
    """가산점 -> 1~5 등급. 반올림은 0.5를 항상 위로 올린다.

    파이썬 내장 round()는 은행가 반올림이라 round(2.5)=2, round(1.5)=2로
    경계에서 방향이 갈린다. 등급 경계가 걸린 값이 실제로 나오므로 고정한다."""
    b = FALLBACK_BASE if base is None else base
    s = FALLBACK_SCALE if scale is None else scale
    return max(1, min(int(math.floor(b + add * s + 0.5)), 5))


def score_by_vocabulary(raw_text: str):
    """(1~5 점수, 근거 라벨) — 메타데이터가 없는 공고(24.2%)의 미검증 폴백.

    ⚠️ 이것은 여전히 **검증된 라벨이 아니다.** 규칙이 계산할 수 없는 공고에
       숫자를 붙이는 임시방편이고, "이 어휘가 있으면 실제로 더 적극적인
       채용인가"는 확인된 적이 없다. v4의 재보정은 그 타당성 문제를 푼 것이
       아니라, 같은 신호를 **덜 틀리게 등급으로 옮기는** 것만 고쳤다.
       (근거: 2-1의 `calibrate_fallback.py`, 02 README '어휘 폴백 재보정')"""
    add, hits = vocabulary_evidence(raw_text)
    return apply_fallback_scale(add), hits


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
        # [수정 4] "(중립)"이라고 쓰지 않는다. 기준점이 3점이던 v3에서는 그 말이
        # 맞았지만 지금은 2점이라 '중립'이 아니다. 그리고 이 경로의 점수는
        # 관측이 아니라 기준점이므로, 그 사실 자체를 문장에 남긴다.
        if not reasons:
            return (f"적극성 {LEVEL_LABEL[level]} — 마감·모집 규모 정보가 없어 "
                    f"측정할 수 없고 본문에 긴급 어휘도 없다. "
                    f"관측이 아니라 기준점({level}점)을 부여한 값이다")
        return (f"적극성 {LEVEL_LABEL[level]} — 마감 정보가 없어 측정 불가. "
                f"본문 어휘로만 추정: " + " · ".join(reasons) +
                " (검증되지 않은 폴백)")
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
    src_path = hf_fetch(MASTER)
    with open(src_path, encoding='utf-8') as f:
        data = json.load(f)

    out = []
    for x in data:
        r = score_posting(x.get('raw_text'), x['source'])
        rec = dict(x)
        rec['urgency_score'] = r['urgency_score']
        rec['urgency_reason'] = r['urgency_reason']
        out.append(rec)

    def load_labels(path):
        """이전 버전 라벨을 원본과 같은 순서의 리스트로 읽는다.

        ⚠️ (source, job_id)를 키로 dict를 만들면 안 된다. 이 데이터셋에는
        같은 키가 777종(1,554행) 중복돼 있어서 dict가 뒤엣것만 남기고,
        서로 다른 공고끼리 짝지어져 비교 결과가 틀린다. 실제로 v3->v4
        비교에서 `1점->5점`처럼 규칙상 불가능한 전이가 나와 발견했다
        (한 문구의 이중 계상을 없앤 수정이라 점수는 내려가기만 한다).

        모든 버전이 master_merged.json을 같은 순서로 훑어 만들어지므로
        위치로 맞추는 것이 정확하다. 정렬이 어긋나면 비교하지 않는다."""
        try:
            with open(path, encoding='utf-8') as f:
                prev = json.load(f)
        except (OSError, ValueError):
            return None
        if len(prev) != len(data):
            return None
        for a, b in zip(prev, data):
            if (a.get('source'), a.get('job_id')) != (b.get('source'), b.get('job_id')):
                return None
        return [r['urgency_score'] for r in prev]

    def baseline(filename):
        """직전 버전 라벨. 로컬에 없으면 hf_data가 data/로 받아온다.
        받을 수도 만들 수도 없으면 그 비교만 건너뛴다(멈추지 않는다)."""
        try:
            return load_labels(hf_fetch(filename))
        except OSError:
            return None

    baselines = [("v3", baseline(MASTER_V3)),
                 ("v2", baseline(MASTER_V2))]

    print("=" * 74)
    print(f"라벨 규칙 {RULE_VERSION} 재산출  (n={len(out):,})")
    print("=" * 74)
    for src in ['jobkorea', 'saramin', 'wanted', None]:
        sub = [r for r in out if src is None or r['source'] == src]
        c = Counter(r['urgency_score'] for r in sub)
        line = '  '.join(f"{lv}:{c.get(lv, 0):>6,}" for lv in range(1, 6))
        print(f"  {src or '전체':<9} n={len(sub):>6,}  {line}")

    for name, prev in baselines:
        if prev is None:
            print(f"\n  ({name} 라벨을 읽을 수 없거나 원본과 정렬이 달라 비교를 건너뜁니다)")
            continue
        chg = Counter()
        for old, r in zip(prev, out):
            if old != r['urgency_score']:
                chg[(old, r['urgency_score'])] += 1
        tot = sum(chg.values())
        print()
        print(f"  {name} 대비 등급 변경: {tot:,}행 ({tot / len(out) * 100:.1f}%)")
        for k, v in sorted(chg.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {k[0]}점 -> {k[1]}점 : {v:>6,}")

    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)   # data/ 폴더 없으면 생성
        with open(OUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n저장: {OUT_PATH}  ({len(out):,} rows)")
    else:
        print("\n(--write 를 붙이면 master_merged_v4.json 으로 저장됩니다)")
    return out

# ---------------------------------------------------------------------------
# 9. 모델용 구조화 피처 (라벨 계산과는 분리 — evidence 합계는 쓰지 않음)
# ---------------------------------------------------------------------------
def structured_features(raw_text: str, source: str = 'unknown') -> dict:
    """TF-IDF 옆에 붙일 구조화 피처.

    extract_signals()의 개별 신호 존재 여부만 가져오고, 가중치 합계(evidence)는
    쓰지 않는다. evidence는 to_level()로 라벨을 직접 만드는 값이라, 그걸
    피처로 넣으면 라벨 유출(label leakage)이 된다."""
    body = clean_body(raw_text or '', source)
    win, rolling = parse_application_window(body)

    n_open = None
    mh = RX_HEADCOUNT.search(body) or RX_HEADCOUNT_ALT.search(body)
    if mh:
        try:
            n_open = int(mh.group(1))
        except ValueError:
            n_open = None

    return {
        'window_days': float(win) if win is not None else -1.0,
        'window_missing': 1.0 if win is None else 0.0,
        'rolling': 1.0 if rolling else 0.0,
        'headcount': float(n_open) if n_open else -1.0,
        'headcount_missing': 0.0 if n_open else 1.0,
        'is_urgent_word': 1.0 if RX_URGENT.search(body) else 0.0,
        'is_immediate': 1.0 if RX_IMMEDIATE.search(body) else 0.0,
        'is_backfill': 1.0 if RX_BACKFILL.search(body) else 0.0,
        'is_early_close': 1.0 if RX_EARLY_CLOSE.search(body) else 0.0,
        'has_bonus': 1.0 if RX_BONUS.search(body) else 0.0,
        'is_many': 1.0 if RX_MANY.search(body) else 0.0,
    }


STRUCT_FEATURE_NAMES = [
    'window_days', 'window_missing', 'rolling', 'headcount', 'headcount_missing',
    'is_urgent_word', 'is_immediate', 'is_backfill', 'is_early_close',
    'has_bonus', 'is_many',
]

if __name__ == '__main__':
    main('--write' in sys.argv)
