"""
backfill_saramin.py — 사람인 공고의 빠진 메타데이터를 다시 채운다.

---------------------------------------------------------------------------
왜 필요한가
---------------------------------------------------------------------------
master_merged.json 의 사람인 25,950건에는 title / company / location /
experience_level 이 **전부 비어 있다**. 소스별 실측:

    source      n        title  company  location  experience  hard_skills
    jobkorea    10,025    100%     100%      100%        100%          38%
    saramin     25,950      0%       0%        0%          0%          88%
    wanted       4,373    100%     100%      100%        100%         100%

추천 시스템에서 이건 치명적이다.
  - 결과 카드에 "제목 없음"이 뜨면 앱으로 쓸 수 없다
  - 경력/지역 하드 필터를 걸 수 없다 (신입에게 '경력 5년 이상'을 추천하게 된다)

메타데이터와 스킬을 둘 다 갖춘 공고는 지금 8,989건(22.3%)뿐이다.
이 스크립트가 그걸 90%대로 올린다.

---------------------------------------------------------------------------
되는 것이 확인됐다
---------------------------------------------------------------------------
2026-06 수집분을 2026-09에 조회한 결과, 30건 샘플에서 **97%가 복원**됐다.
마감된 공고도 페이지 자체는 살아 있고 <meta> 태그가 남아 있다.

사람인 desktop 페이지의 meta description 이 일정한 형식이다:

    (주)다비오, (주)다비오 PMO 채용, 경력:경력 5~15년,
    학력:대학졸업(2,3년)이상, 면접 후 결정, 마감일:2026-07-02, 홈페이지:...

여기서 company / title / experience / education / deadline 이 나온다.
**지역만 desktop 에 없어서** 모바일 페이지를 한 번 더 봐야 한다(--with-location).

---------------------------------------------------------------------------
설계에서 신경 쓴 것
---------------------------------------------------------------------------
1. **재개 가능** — 2.6만 건이면 몇 시간짜리다. 중간에 끊겨도 이어서 돌 수 있게
   매 CHECKPOINT_EVERY 건마다 결과를 파일에 쓴다. 다시 실행하면 이미 받은 건은
   건너뛴다.
2. **예의** — 요청 간격을 둔다(기본 0.4초). 이미 가진 공고의 메타데이터를 보강하는
   1회성 작업이지 신규 대량 수집이 아니지만, 그래도 몰아치지 않는다.
3. **원본을 덮지 않는다** — master_merged.json 은 건드리지 않고 별도 파일에 쓴다.
   합치는 것은 소비하는 쪽(추천 파이프라인)의 몫이다. 백필이 잘못돼도 원본은 남는다.
4. **실패를 기록한다** — 404/파싱실패를 status 로 남긴다. 나중에 커버리지를
   정직하게 보고할 수 있어야 한다.

실행:
    python backfill_saramin.py --limit 200        # 먼저 소규모로 확인
    python backfill_saramin.py                    # 전체 (약 3시간)
    python backfill_saramin.py --with-location    # 지역까지 (요청 2배, 약 6시간)
    python backfill_saramin.py --report           # 수집 안 하고 현재 커버리지만 출력
"""

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding='utf-8')

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SRC = DATA / "master_merged.json"
OUT = DATA / "saramin_meta_backfill.json"

VIEW = 'https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx={}'
MOBILE = 'https://m.saramin.co.kr/job-search/view?rec_idx={}'
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36')
UA_MOBILE = ('Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) '
             'AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1')
HEADERS = {'User-Agent': UA, 'Accept-Language': 'ko-KR,ko;q=0.9'}

DELAY = 0.4               # 요청 간격(초)
TIMEOUT = 15
RETRY = 2
CHECKPOINT_EVERY = 200


# ---------------------------------------------------------------------------
# 파싱
# ---------------------------------------------------------------------------
RX_DESC = re.compile(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']*)["\']', re.I)
RX_TITLE = re.compile(r'<title[^>]*>(.*?)</title>', re.S | re.I)
RX_TAG = re.compile(r'<[^>]+>')
RX_DROP = re.compile(r'<(script|style|noscript)[^>]*>.*?</\1>', re.S | re.I)
RX_LOC = re.compile(r'지역\s+((?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|'
                    r'전북|전남|경북|경남|제주)[^\s]*(?:\s+[가-힣]+[시군구])?)')


def _plain(h: str) -> str:
    t = RX_DROP.sub(' ', h or '')
    return re.sub(r'\s+', ' ', html.unescape(RX_TAG.sub(' ', t))).strip()


def parse_desc(desc: str) -> dict:
    """meta description 을 필드로 쪼갠다.

    형식: "회사명, 제목, 경력:X, 학력:Y, [급여], 마감일:Z, 홈페이지:W"
    제목에 쉼표가 들어갈 수 있으므로 앞뒤에서 좁혀 들어간다."""
    out = {}
    if not desc:
        return out
    for key, field in [('경력', 'experience_level'), ('학력', 'education')]:
        m = re.search(rf'{key}\s*:\s*([^,]+)', desc)
        if m:
            out[field] = m.group(1).strip()
    m = re.search(r'마감일\s*:\s*(20\d\d-\d{2}-\d{2})', desc)
    if m:
        out['deadline'] = m.group(1)

    head = re.split(r',\s*경력\s*:', desc)[0]        # "회사명, 제목"
    parts = [p.strip() for p in head.split(',')]
    if parts:
        out['company'] = parts[0]
        if len(parts) > 1:
            out['title'] = ', '.join(parts[1:]).strip()
    return out


def parse_title_tag(t: str) -> dict:
    """<title> = "[회사명] 제목 - 사람인". desc 로 못 채운 값의 폴백."""
    t = re.sub(r'\s*-\s*사람인\s*$', '', (t or '').strip())
    m = re.match(r'\[(.+?)\]\s*(.+)$', t)
    if not m:
        return {'title': t} if t else {}
    return {'company': m.group(1).strip(),
            'title': re.sub(r'\s*\(D-\d+\)\s*$', '', m.group(2)).strip()}


# ---------------------------------------------------------------------------
# 수집
# ---------------------------------------------------------------------------
def fetch(url, ua, session):
    for attempt in range(RETRY + 1):
        try:
            r = session.get(url, headers={**HEADERS, 'User-Agent': ua}, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.text
            if r.status_code in (404, 410):
                return None                      # 재시도해도 소용없다
        except requests.RequestException:
            pass
        if attempt < RETRY:
            time.sleep(1.5 * (attempt + 1))
    return None


def scrape_one(rid, session, with_location=False):
    h = fetch(VIEW.format(rid), UA, session)
    if h is None:
        return {'status': 'not_found'}

    m = RX_DESC.search(h)
    rec = parse_desc(html.unescape(m.group(1)) if m else '')
    mt = RX_TITLE.search(h)
    for k, v in parse_title_tag(html.unescape(mt.group(1)) if mt else '').items():
        rec.setdefault(k, v)                     # desc 값을 우선

    if not rec.get('title'):
        return {'status': 'parse_failed'}

    if with_location:
        hm = fetch(MOBILE.format(rid), UA_MOBILE, session)
        if hm:
            lm = RX_LOC.search(_plain(hm))
            if lm:
                rec['location'] = lm.group(1).strip()

    rec['status'] = 'ok'
    return rec


# ---------------------------------------------------------------------------
def load_targets():
    with open(SRC, encoding='utf-8') as f:
        data = json.load(f)
    return [x['job_id'] for x in data
            if x['source'] == 'saramin' and not x.get('title')]


def load_done():
    if OUT.exists():
        with open(OUT, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save(done):
    tmp = OUT.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(done, f, ensure_ascii=False, indent=1)
    tmp.replace(OUT)                             # 원자적 교체 — 중간에 끊겨도 파일이 깨지지 않는다


def report(targets, done):
    from collections import Counter
    c = Counter(v.get('status') for v in done.values())
    n = len(targets)
    print(f"  대상 {n:,}건 / 처리 {len(done):,}건 ({len(done)/n*100:.1f}%)")
    for k, v in c.most_common():
        print(f"    {k or '(없음)':<14}{v:>7,} ({v/max(len(done),1)*100:5.1f}%)")
    ok = [v for v in done.values() if v.get('status') == 'ok']
    if ok:
        print()
        for f in ['title', 'company', 'experience_level', 'education', 'deadline', 'location']:
            got = sum(1 for v in ok if v.get(f))
            print(f"    {f:<18}{got:>7,} / {len(ok):,} ({got/len(ok)*100:5.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='앞에서 N건만 (시험용)')
    ap.add_argument('--with-location', action='store_true', help='지역까지 수집(요청 2배)')
    ap.add_argument('--delay', type=float, default=DELAY)
    ap.add_argument('--report', action='store_true', help='수집 없이 현재 상태만 출력')
    args = ap.parse_args()

    targets = load_targets()
    done = load_done()

    if args.report:
        report(targets, done)
        return

    todo = [r for r in targets if r not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"  대상 {len(targets):,}건 · 완료 {len(done):,}건 · 이번 실행 {len(todo):,}건")
    if not todo:
        print("  받을 것이 없습니다.")
        report(targets, done)
        return
    eta = len(todo) * args.delay * (2 if args.with_location else 1) / 60
    print(f"  예상 소요 약 {eta:.0f}분 (간격 {args.delay}초"
          f"{', 지역 포함' if args.with_location else ''})\n")

    t0 = time.time()
    with requests.Session() as sess:
        for i, rid in enumerate(todo, 1):
            done[rid] = scrape_one(rid, sess, args.with_location)
            if i % CHECKPOINT_EVERY == 0:
                save(done)
                ok = sum(1 for v in done.values() if v.get('status') == 'ok')
                el = time.time() - t0
                print(f"  {i:>6,}/{len(todo):,}  성공 {ok:,}  "
                      f"경과 {el/60:.0f}분  잔여 약 {(len(todo)-i)*el/i/60:.0f}분")
            time.sleep(args.delay)
    save(done)

    print(f"\n  저장: {OUT}")
    report(targets, done)


if __name__ == '__main__':
    main()
