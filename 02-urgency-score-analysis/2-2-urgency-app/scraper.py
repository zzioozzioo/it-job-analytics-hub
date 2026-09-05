"""
scraper.py — 채용공고 URL을 학습 데이터와 같은 모양의 raw_text로 바꾼다.

---------------------------------------------------------------------------
왜 그냥 긁어오면 안 되는가
---------------------------------------------------------------------------
모델과 규칙은 크롤링 스냅샷(`master_merged.json`)의 `raw_text` 분포에서
만들어졌다. 사이트를 새로 긁으면 같은 공고라도 텍스트 모양이 다르다.

  - jobkorea는 `남은기간 12일`을 자바스크립트로 그린다. 정적 HTML에는 없다.
  - saramin은 상세 본문이 별도 엔드포인트(view-detail)에 있다.
  - 두 사이트 모두 `<meta name="description">`에 마감일이 들어 있다.

그래서 '긁은 HTML'을 바로 쓰지 않고, **구조화 필드를 뽑아 규칙이 읽는
정규 형식으로 다시 조립한다**(`_canonical_header`). 그래야 규칙과 모델이
학습 때와 같은 문자열을 본다.

⚠️ 이건 재현이지 동일이 아니다. 크롤링 스냅샷과 100% 같은 문자열은 아니고,
   특히 사이트가 마크업을 바꾸면 조용히 어긋난다. 앱은 조립된 raw_text를
   화면에 그대로 펼쳐 보여준다 — 사용자가 무엇을 근거로 채점됐는지 확인할
   수 있어야 오작동을 눈치챌 수 있다.

---------------------------------------------------------------------------
알려진 실패 모드 (앱에서 사용자에게 그대로 알린다)
---------------------------------------------------------------------------
1. 이미지 공고 — saramin·jobkorea 모두 본문 전체를 이미지 한 장으로 올리는
   공고가 흔하다. 텍스트가 거의 없으므로 분석이 불가능하다. OCR은 넣지 않았다.
2. wanted — 공고 메타데이터(마감일·모집인원)가 구조적으로 없다. 규칙이
   측정 불가로 판정하고, 모델도 검증 범위 밖이다.
3. 마감된 공고 — 페이지는 살아 있지만 접수 정보가 사라진다.
4. 사이트 개편 — 셀렉터가 아니라 정규식 기반이라 좀 더 버티지만, 결국 깨진다.
   `diagnostics`에 무엇을 못 찾았는지 남기므로 앱에서 확인할 수 있다.

robots/이용약관: 사용자가 직접 붙여넣은 공고 1건을 조회하는 용도다.
목록 크롤링·대량 수집은 하지 않는다.
"""

import datetime
import html
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

import requests

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36')
HEADERS = {'User-Agent': UA, 'Accept-Language': 'ko-KR,ko;q=0.9',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}
TIMEOUT = 15


class ScrapeError(Exception):
    """사용자에게 그대로 보여줄 수 있는 실패 사유."""


@dataclass
class Posting:
    source: str
    job_id: str
    url: str
    title: str = ''
    company: str = ''
    raw_text: str = ''
    deadline: datetime.date | None = None
    start_date: datetime.date | None = None
    headcount: int | None = None
    always_open: bool = False
    # 앱이 사용자에게 보여줄 진단. 무엇을 못 찾았는지가 중요하다.
    diagnostics: list = field(default_factory=list)

    @property
    def days_left(self):
        """오늘 기준 남은 일수. 부가 정보이지 점수 근거가 아니다.

        점수에 넣지 않는 이유는 urgency_rule.py [수정 2] 참조 —
        조회 시점에 따라 달라지는 값을 점수에 넣으면 어제 4점이던 공고가
        오늘 5점이 된다."""
        if self.deadline is None:
            return None
        return (self.deadline - datetime.date.today()).days


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
_RX_TAG = re.compile(r'<[^>]+>')
_RX_DROP = re.compile(r'<(script|style|noscript)[^>]*>.*?</\1>', re.S | re.I)
_RX_META_DESC = re.compile(
    r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']*)["\']', re.I)
_RX_OG_TITLE = re.compile(
    r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']*)["\']', re.I)
_RX_TITLE = re.compile(r'<title[^>]*>(.*?)</title>', re.S | re.I)


def _text(h: str) -> str:
    """HTML -> 공백 정규화된 평문."""
    t = _RX_DROP.sub(' ', h or '')
    t = _RX_TAG.sub(' ', t)
    return re.sub(r'\s+', ' ', html.unescape(t)).strip()


# ---------------------------------------------------------------------------
# 사이트 크롬(전역 내비게이션 · 푸터) 제거
# ---------------------------------------------------------------------------
# ⚠️ 반드시 필요한 단계다. 학습 데이터의 raw_text에는 사이트 GNB가 없다
#    (jobkorea는 회사명부터, wanted는 직무기술서만 담겨 있었다).
#    긁은 평문을 그대로 쓰면 메뉴 텍스트가 공고 신호로 오인된다.
#
#    실제로 확인된 오탐: 잡코리아 GNB의
#      "회원가입/로그인 기업 서비스 JOB 찾기 합격축하금 공채정보 …"
#    에서 `합격축하금`이 잡혀 '보상 유인' +12점이 붙었다. 모든 잡코리아
#    공고에 똑같이 붙으므로 점수 전체가 한 등급씩 밀린다.
_CHROME_HEAD = {
    # GNB 시작~끝. 공고 본문은 그 뒤부터 시작한다.
    'jobkorea': re.compile(r'^.*?취업톡톡\s*', re.S),
    'wanted': re.compile(r'^.*?포지션\s*상세\s*', re.S),
}
_CHROME_TAIL = {
    'wanted': re.compile(r'(더\s*많은\s*포지션을\s*찾아|저작권자\s*\(주\)원티드랩).*$', re.S),
    # jobkorea 꼬리는 urgency_rule.clean_body가 학습 때와 동일한 기준으로
    # 잘라내므로 여기서 건드리지 않는다.
}


def _strip_chrome(text: str, source: str) -> str:
    t = text or ''
    head = _CHROME_HEAD.get(source)
    if head:
        cut = head.sub('', t, count=1)
        # 크롬 제거가 본문까지 먹었으면(마크업 변경 등) 원본을 되돌린다.
        if len(cut) >= 200:
            t = cut
    tail = _CHROME_TAIL.get(source)
    if tail:
        cut = tail.sub('', t, count=1)
        if len(cut) >= 200:
            t = cut
    return t.strip()


def _get(url: str, referer: str | None = None) -> str:
    hdr = dict(HEADERS)
    if referer:
        hdr['Referer'] = referer
    try:
        r = requests.get(url, headers=hdr, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise ScrapeError(f"접속 실패: {type(e).__name__}") from e
    if r.status_code != 200:
        raise ScrapeError(f"HTTP {r.status_code} — 삭제되었거나 접근이 막힌 공고입니다.")
    r.encoding = r.apparent_encoding or r.encoding
    return r.text


def _meta_desc(h: str) -> str:
    m = _RX_META_DESC.search(h)
    return html.unescape(m.group(1)) if m else ''


def _page_title(h: str) -> str:
    m = _RX_OG_TITLE.search(h) or _RX_TITLE.search(h)
    return re.sub(r'\s+', ' ', html.unescape(m.group(1))).strip() if m else ''


def _date(y, mo, d):
    try:
        return datetime.date(int(y), int(mo), int(d))
    except (ValueError, TypeError):
        return None


_RX_ANY_DATE = re.compile(r'(20\d\d)\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})')


def _find_deadline(text: str):
    """'마감일 : 2026.10.03' / '마감일:2026-10-03' 형태에서 날짜를 뽑는다."""
    m = re.search(r'마감일\s*[:：]?\s*' + _RX_ANY_DATE.pattern, text)
    return _date(*m.group(1, 2, 3)) if m else None


# ---------------------------------------------------------------------------
# 정규 헤더 조립 — 여기가 이 모듈의 핵심
# ---------------------------------------------------------------------------
def _canonical_header(p: Posting) -> str:
    """규칙(urgency_rule)이 읽는 정규 형식으로 메타데이터를 다시 쓴다.

    `시작일 YYYY.MM.DD 마감일 YYYY.MM.DD` 형태로 내보내면 규칙의
    RX_START_END가 접수 창 길이를 계산할 수 있다. 이 형식은 jobkorea
    크롤링 텍스트에 실제로 있던 모양과 같다."""
    parts = []
    if p.start_date and p.deadline:
        parts.append(f"접수기간 · 방법 시작일 {p.start_date:%Y.%m.%d} "
                     f"마감일 {p.deadline:%Y.%m.%d}")
    elif p.deadline:
        # 시작일이 없으면 창 길이를 계산할 수 없다. 마감일만 적어 두면
        # is_measurable은 통과하지만 마감 신호는 0점이 된다 — 의도된 동작이다.
        # (없는 시작일을 오늘로 가정하면 조회 시점이 점수에 섞인다)
        parts.append(f"접수기간 · 방법 마감일 {p.deadline:%Y.%m.%d}")
    elif p.always_open:
        parts.append("접수기간 · 방법 마감일 상시채용")
    if p.headcount is not None:
        parts.append(f"모집인원 {p.headcount} 명")
    return ' '.join(parts)


def _assemble(p: Posting, body: str) -> None:
    header = _canonical_header(p)
    p.raw_text = (header + ' ' + body).strip() if header else body.strip()
    if not header:
        p.diagnostics.append(
            "채용 메타데이터(접수기간·마감일·모집인원)를 찾지 못했습니다. "
            "규칙은 '측정 불가'로 판정합니다.")
    if p.start_date is None and p.deadline is not None:
        p.diagnostics.append(
            "시작일이 없어 접수 창 길이를 계산할 수 없습니다. "
            "마감 관련 가점은 0점 처리됩니다.")
    if len(body) < 300:
        p.diagnostics.append(
            f"본문 텍스트가 {len(body)}자로 매우 짧습니다. 공고 내용이 "
            "이미지로 올라간 경우일 가능성이 높고, 이때 분석 결과는 무의미합니다.")


# ---------------------------------------------------------------------------
# jobkorea
# ---------------------------------------------------------------------------
_RX_JK_START_END = re.compile(
    r'시작일\s*(20\d\d)\.(\d{1,2})\.(\d{1,2})[^0-9]{0,8}마감일\s*(20\d\d)\.(\d{1,2})\.(\d{1,2})')
_RX_JK_HEAD = re.compile(r'모집인원\s*([0-9]+)\s*명')
_RX_JK_ALWAYS = re.compile(r'마감일\s*상시채용|상시\s*채용|수시\s*채용')


def _scrape_jobkorea(job_id: str, url: str) -> Posting:
    h = _get(url)
    t = _text(h)
    p = Posting('jobkorea', job_id, url)

    title = _page_title(h)
    p.title = re.sub(r'\s*\|\s*잡코리아\s*$', '', title)
    m = re.match(r'(.+?)\s*채용\s*-\s*', p.title)
    if m:
        p.company = m.group(1).strip()

    m = _RX_JK_START_END.search(t)           # 렌더된 접수기간 블록 (가장 정확)
    if m:
        p.start_date = _date(*m.group(1, 2, 3))
        p.deadline = _date(*m.group(4, 5, 6))
    else:
        p.deadline = _find_deadline(_meta_desc(h))   # meta 폴백
        if p.deadline:
            p.diagnostics.append(
                "접수기간 블록을 못 찾아 meta 태그의 마감일만 사용했습니다.")

    if _RX_JK_ALWAYS.search(t):
        p.always_open = True
    mh = _RX_JK_HEAD.search(t)
    if mh:
        p.headcount = int(mh.group(1))       # jobkorea는 흔히 `○○`로 가려 둔다

    if '마감되었습니다' in t:
        p.diagnostics.append("이미 마감된 공고입니다.")

    # 본문에서 메타 구간을 지우지 않는다 — 규칙이 조립한 헤더를 먼저 읽고,
    # 본문의 급구/결원 같은 어휘 신호는 그대로 살아 있어야 한다.
    _assemble(p, _strip_chrome(t, 'jobkorea'))
    return p


# ---------------------------------------------------------------------------
# saramin
# ---------------------------------------------------------------------------
_SR_VIEW = 'https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx={}'
_SR_DETAIL = 'https://www.saramin.co.kr/zf_user/jobs/relay/view-detail?rec_idx={}'
_SR_MOBILE = 'https://m.saramin.co.kr/job-search/view?rec_idx={}'
_RX_SR_HEAD = re.compile(r'모집인원\s*[:：]?\s*([0-9]+)\s*명')
# 모바일 페이지에만 있는 접수기간 블록:
#   "시작일 2026.08.31(월) 00:00 마감일 2026.09.07(월) 23:59"
_RX_SR_START_END = re.compile(
    r'시작일\s*(20\d\d)\.(\d{1,2})\.(\d{1,2}).{0,20}?마감일\s*(20\d\d)\.(\d{1,2})\.(\d{1,2})')
_UA_MOBILE = ('Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) '
              'AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1')


def _scrape_saramin(job_id: str, url: str) -> Posting:
    h = _get(_SR_VIEW.format(job_id))
    p = Posting('saramin', job_id, url)
    desc = _meta_desc(h)
    p.title = re.sub(r'\s*-\s*사람인\s*$', '', _page_title(h))
    if ',' in desc:
        p.company = desc.split(',')[0].strip()
    p.deadline = _find_deadline(desc) or _find_deadline(_text(h))

    # 데스크톱 페이지에는 마감일만 있고 시작일이 없다. 시작일이 없으면
    # 접수 창 길이를 계산할 수 없고, 그러면 saramin 공고는 마감 신호가
    # 항상 0점이 된다. 모바일 페이지에는 '시작일 ... 마감일 ...' 블록이
    # 그대로 렌더돼 있어서 여기서 창 길이를 얻는다.
    try:
        mob = _text(requests.get(_SR_MOBILE.format(job_id),
                                 headers={**HEADERS, 'User-Agent': _UA_MOBILE},
                                 timeout=TIMEOUT).text)
        m = _RX_SR_START_END.search(mob)
        if m:
            p.start_date = _date(*m.group(1, 2, 3))
            p.deadline = _date(*m.group(4, 5, 6)) or p.deadline
        else:
            p.diagnostics.append(
                "모바일 페이지에서 접수기간 블록을 찾지 못했습니다(상시채용 공고이거나 "
                "마크업이 바뀐 경우). 접수 창 길이는 계산되지 않습니다.")
    except requests.RequestException:
        p.diagnostics.append(
            "시작일 조회(모바일 페이지)에 실패했습니다. 접수 창 길이 없이 채점합니다.")

    # 상세 본문은 별도 엔드포인트에 있다. 실패해도 요약 정보로 계속 진행한다.
    body = ''
    try:
        body = _text(_get(_SR_DETAIL.format(job_id), referer=_SR_VIEW.format(job_id)))
    except ScrapeError as e:
        p.diagnostics.append(f"상세 본문을 가져오지 못했습니다({e}). 요약 정보만 사용합니다.")
    if not body:
        body = _text(h)

    mh = _RX_SR_HEAD.search(body)
    if mh:
        p.headcount = int(mh.group(1))
    if re.search(r'상시\s*채용|수시\s*채용|채용\s*시\s*마감', body):
        p.always_open = True

    _assemble(p, body)
    return p


# ---------------------------------------------------------------------------
# wanted
# ---------------------------------------------------------------------------
def _scrape_wanted(job_id: str, url: str) -> Posting:
    h = _get(url)
    p = Posting('wanted', job_id, url)
    p.title = re.sub(r'\s*\|\s*원티드\s*$', '', _page_title(h))
    body = _strip_chrome(_text(h), 'wanted')
    # 원티드는 본문 하단 '기술 스택 · 툴 태그 마감일 YYYY.MM.DD'에 마감일이
    # 있는 공고가 있다. 학습 데이터(직무기술서만)에는 없던 정보다.
    p.deadline = _find_deadline(body)
    p.diagnostics.append(
        "원티드 공고에는 마감일·모집인원 같은 채용 메타데이터가 구조적으로 "
        "없습니다. 규칙은 어휘 폴백으로만 채점하며, 이 폴백은 '항상 3점'보다 "
        "정확도가 나쁘다고 측정된 방식입니다(2-1 한계 1).")
    _assemble(p, body)
    return p


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
_ROUTES = [
    (re.compile(r'jobkorea\.co\.kr'),
     re.compile(r'GI_Read/(\d+)|Recruit/GI_Read/(\d+)'), 'jobkorea'),
    (re.compile(r'saramin\.co\.kr'), None, 'saramin'),
    (re.compile(r'wanted\.co\.kr'), re.compile(r'/wd/(\d+)'), 'wanted'),
]

SUPPORTED = {
    'jobkorea': 'https://www.jobkorea.co.kr/Recruit/GI_Read/<번호>',
    'saramin': 'https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx=<번호>',
    'wanted': 'https://www.wanted.co.kr/wd/<번호>',
}


def detect(url: str):
    """(source, job_id). 지원하지 않는 주소면 ScrapeError."""
    u = (url or '').strip()
    if not u.startswith('http'):
        u = 'https://' + u
    host = urlparse(u).netloc.lower()

    if 'jobkorea.co.kr' in host:
        m = re.search(r'GI_Read/(\d+)', u)
        if m:
            return 'jobkorea', m.group(1)
    elif 'saramin.co.kr' in host:
        q = parse_qs(urlparse(u).query)
        if 'rec_idx' in q:
            return 'saramin', q['rec_idx'][0]
        m = re.search(r'rec_idx[=/](\d+)', u)
        if m:
            return 'saramin', m.group(1)
    elif 'wanted.co.kr' in host:
        m = re.search(r'/wd/(\d+)', u)
        if m:
            return 'wanted', m.group(1)
    else:
        raise ScrapeError(
            "지원하지 않는 사이트입니다. 잡코리아 · 사람인 · 원티드만 분석할 수 "
            "있습니다(모델이 그 세 곳의 데이터로만 학습됐습니다).")
    raise ScrapeError("주소에서 공고 번호를 찾지 못했습니다. 공고 상세 페이지 주소인지 확인해 주세요.")


_SCRAPERS = {'jobkorea': _scrape_jobkorea, 'saramin': _scrape_saramin,
             'wanted': _scrape_wanted}


def scrape(url: str) -> Posting:
    source, job_id = detect(url)
    return _SCRAPERS[source](job_id, url.strip())


def from_text(text: str, source: str = 'unknown') -> Posting:
    """스크래핑이 막혔을 때의 우회로 — 사용자가 본문을 직접 붙여넣는다.

    조립된 헤더가 없으므로 붙여넣은 텍스트에 '접수기간'·'모집인원'이
    들어 있어야 규칙이 측정 가능으로 판정한다."""
    p = Posting(source, '', '', raw_text=(text or '').strip())
    p.deadline = _find_deadline(p.raw_text)
    p.diagnostics.append(
        "직접 붙여넣은 텍스트입니다. 접수기간·모집인원 문구가 함께 복사되지 "
        "않았다면 규칙은 '측정 불가'로 판정합니다.")
    return p


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    for u in sys.argv[1:]:
        try:
            p = scrape(u)
            print(f"[{p.source}] {p.title[:60]}")
            print(f"  회사={p.company}  마감={p.deadline}  시작={p.start_date}  "
                  f"인원={p.headcount}  남은일수={p.days_left}")
            print(f"  raw_text {len(p.raw_text):,}자: {p.raw_text[:200]}")
            for d in p.diagnostics:
                print(f"  ! {d}")
        except ScrapeError as e:
            print(f"[실패] {u}\n  {e}")
        print()
