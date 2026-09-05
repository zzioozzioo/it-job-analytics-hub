"""
채용 적극성 분석기 — 공고 URL을 붙여넣으면 규칙과 모델이 각각 채점한다.

화면 설계에서 지킨 원칙 하나: **규칙이 주 결과, 모델은 두 번째 의견.**
라벨이 규칙의 출력이므로 규칙이 계산 가능한 공고에서는 규칙이 정의상 정답이고
모델은 그 근사치다. 모델을 앞에 세우면 근사치를 정답처럼 보이게 만든다.
그래서 규칙 점수를 크게 놓고, 모델은 옆에 대조용으로 둔다.
둘이 갈리면 갈렸다고 그대로 보여준다 — 합치거나 평균 내지 않는다.

실행: streamlit run app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import scraper  # noqa: E402
from urgency_rule import LEVEL_LABEL, clean_body, score_posting  # noqa: E402

# ---------------------------------------------------------
# 브랜드 팔레트 — 1번 프로젝트(.streamlit/config.toml)와 1:1로 맞춘 상수
# ---------------------------------------------------------
PRIMARY = "#2563EB"
PRIMARY_SOFT = "#EFF6FF"
PRIMARY_DARK = "#1E3A8A"
INK = "#0F172A"
MUTED = "#64748B"
SURFACE = "#F1F5F9"
BORDER = "#E2E8F0"
WARN = "#B45309"
WARN_SOFT = "#FFFBEB"

# 등급별 색 — 낮음(차분한 슬레이트) → 높음(경고성 앰버/레드)
LEVEL_COLOR = {1: "#64748B", 2: "#0EA5E9", 3: "#2563EB",
               4: "#F59E0B", 5: "#DC2626"}

st.set_page_config(page_title="채용 적극성 분석기", page_icon="📊",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
  #MainMenu, footer, header {{visibility: hidden;}}
  .block-container {{padding-top: 2.2rem; max-width: 1180px;}}
  .hero h1 {{font-size: 1.9rem; margin: 0 0 .35rem 0; color: {INK};
             letter-spacing: -.02em;}}
  .hero p {{color: {MUTED}; margin: 0; font-size: .95rem; line-height: 1.6;}}
  .card {{background: #fff; border: 1px solid {BORDER}; border-radius: 12px;
          padding: 1.15rem 1.3rem; margin-bottom: .8rem;}}
  .card.soft {{background: {SURFACE}; border-color: {BORDER};}}
  .card.warn {{background: {WARN_SOFT}; border-color: #FDE68A;}}
  .score-num {{font-size: 3.4rem; font-weight: 700; line-height: 1;
               letter-spacing: -.03em;}}
  .score-cap {{font-size: .78rem; color: {MUTED}; text-transform: uppercase;
               letter-spacing: .09em; margin-bottom: .4rem;}}
  .badge {{display:inline-block; padding:.2rem .6rem; border-radius:999px;
           font-size:.75rem; font-weight:600; background:{PRIMARY_SOFT};
           color:{PRIMARY_DARK}; margin-right:.35rem;}}
  .badge.warn {{background:{WARN_SOFT}; color:{WARN};}}
  .badge.mut {{background:{SURFACE}; color:{MUTED};}}
  .reason {{background:{PRIMARY_SOFT}; border-left:3px solid {PRIMARY};
            padding:.5rem .8rem; border-radius:0 6px 6px 0; margin-bottom:.35rem;
            font-size:.9rem; color:{INK};}}
  .kv {{display:flex; justify-content:space-between; padding:.35rem 0;
        border-bottom:1px solid {BORDER}; font-size:.9rem;}}
  .kv:last-child {{border-bottom:none;}}
  .kv span:first-child {{color:{MUTED};}}
  .kv span:last-child {{color:{INK}; font-weight:600;}}
  .note {{color:{MUTED}; font-size:.83rem; line-height:1.6;}}
  .stTabs [data-baseweb="tab"] {{font-size:.93rem;}}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 로더
# =========================================================
@st.cache_resource(show_spinner="모델 로딩 중...")
def load_model():
    """모델이 없어도 앱은 동작해야 한다 — 규칙만으로 채점하고 안내한다."""
    try:
        from urgency_model import UrgencyModel
        return UrgencyModel(), None
    except Exception as e:
        return None, str(e)


@st.cache_data
def load_stats():
    p = HERE / "reference_stats.json"
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


@st.cache_data(show_spinner=False, ttl=600)
def fetch(url: str):
    """같은 URL을 반복 조회하지 않도록 10분 캐시. 사이트에 대한 예의이기도 하다."""
    p = scraper.scrape(url)
    return {k: getattr(p, k) for k in
            ('source', 'job_id', 'url', 'title', 'company', 'raw_text',
             'deadline', 'start_date', 'headcount', 'always_open', 'diagnostics')}


MODEL, MODEL_ERR = load_model()
STATS = load_stats()


# =========================================================
# 렌더 헬퍼
# =========================================================
def score_card(score, caption, sub=''):
    c = LEVEL_COLOR[score]
    return (f'<div class="card" style="border-left:4px solid {c};">'
            f'<div class="score-cap">{caption}</div>'
            f'<div class="score-num" style="color:{c};">{score}<span '
            f'style="font-size:1.1rem;color:{MUTED};font-weight:500;"> / 5</span></div>'
            f'<div style="color:{c};font-weight:600;margin-top:.25rem;">'
            f'{LEVEL_LABEL[score]}</div>'
            f'<div class="note" style="margin-top:.45rem;">{sub}</div></div>')


def percentile_of(score, source):
    """이 점수가 데이터셋에서 어느 위치인지. 절대 점수만으로는 감이 안 온다."""
    if not STATS:
        return None
    dist = STATS['by_source'].get(source, {}).get('dist') or STATS['overall']
    total = sum(dist.values())
    if not total:
        return None
    at_or_below = sum(v for k, v in dist.items() if int(k) <= score)
    return at_or_below / total * 100


# =========================================================
# 헤더
# =========================================================
st.markdown(f"""
<div class="hero">
  <h1>채용 적극성 분석기</h1>
  <p>잡코리아 · 사람인 · 원티드 공고 주소를 넣으면 접수 기간, 모집 규모, 즉시 입사 요구 같은
     <b>본문에서 실제로 관측되는 신호</b>만으로 채용 적극성을 1~5점으로 채점합니다.<br>
     IT 채용공고 40,348건으로 만든 규칙과, 같은 라벨로 학습한 모델이 각각 채점한 결과를 나란히 보여줍니다.</p>
</div>
""", unsafe_allow_html=True)
st.write("")

tab_run, tab_how, tab_limit = st.tabs(
    ["공고 분석", "어떻게 만들었나", "한계와 개선점"])


# =========================================================
# 탭 1 — 공고 분석
# =========================================================
with tab_run:
    mode = st.radio("입력 방식", ["URL 붙여넣기", "본문 직접 붙여넣기"],
                    horizontal=True, label_visibility="collapsed")

    posting = None
    err = None

    if mode == "URL 붙여넣기":
        col_in, col_btn = st.columns([5, 1])
        url = col_in.text_input(
            "공고 주소", placeholder="https://www.jobkorea.co.kr/Recruit/GI_Read/...",
            label_visibility="collapsed")
        go = col_btn.button("분석", type="primary", use_container_width=True)
        st.markdown(
            f'<div class="note">지원: '
            + ' · '.join(f'<code>{v}</code>' for v in scraper.SUPPORTED.values())
            + '</div>', unsafe_allow_html=True)
        if go and url.strip():
            try:
                posting = fetch(url.strip())
            except scraper.ScrapeError as e:
                err = str(e)
            except Exception as e:      # 사이트 개편 등 예상 못한 실패
                err = f"공고를 읽는 중 오류가 발생했습니다: {type(e).__name__}: {e}"
    else:
        st.markdown(
            '<div class="note">사이트가 막히거나 이미지 공고일 때 쓰는 우회로입니다. '
            '<b>접수기간·모집인원 문구까지 함께</b> 복사해야 규칙이 채점할 수 있습니다.</div>',
            unsafe_allow_html=True)
        txt = st.text_area("공고 본문", height=180, label_visibility="collapsed")
        src = st.selectbox("어느 사이트 공고인가요?",
                           ["jobkorea", "saramin", "wanted", "unknown"], index=3)
        if st.button("분석", type="primary") and txt.strip():
            p = scraper.from_text(txt, src)
            posting = {k: getattr(p, k) for k in
                       ('source', 'job_id', 'url', 'title', 'company', 'raw_text',
                        'deadline', 'start_date', 'headcount', 'always_open',
                        'diagnostics')}

    if err:
        st.error(err)

    if posting:
        raw = posting['raw_text']
        source = posting['source']
        result = score_posting(raw, source)

        # ---- 공고 요약 ----
        title = posting['title'] or '(제목 없음)'
        st.markdown(f"### {title}")
        meta_bits = []
        if posting['company']:
            meta_bits.append(f'<span class="badge">{posting["company"]}</span>')
        meta_bits.append(f'<span class="badge mut">{source}</span>')
        if posting['deadline']:
            import datetime
            dl = posting['deadline']
            left = (dl - datetime.date.today()).days
            cls = 'warn' if left is not None and left <= 7 else 'mut'
            tail = (f'D-{left}' if left is not None and left >= 0
                    else '마감됨')
            meta_bits.append(f'<span class="badge {cls}">마감 {dl:%Y.%m.%d} · {tail}</span>')
        st.markdown(' '.join(meta_bits), unsafe_allow_html=True)
        st.write("")

        # ---- 점수 ----
        rule_score = result['urgency_score']
        pct = percentile_of(rule_score, source)
        # "상위 N%"라고 쓰면 동점 처리를 숨기게 된다(2점이 전체의 절반이라
        # 동점이 대량으로 생긴다). 누적 비율을 그대로 말한다.
        sub = (f"같은 사이트 공고 {STATS['by_source'][source]['n']:,}건 중 "
               f"이 점수 이하가 {pct:.0f}%입니다."
               if pct and source in (STATS or {}).get('by_source', {}) else "")
        if not result['measurable']:
            sub = "⚠️ 채용 메타데이터가 없어 <b>어휘 폴백</b>으로 매긴 미검증 점수입니다."
        elif result['window_days'] is None and posting['deadline'] is not None:
            # '창이 길어서 0점'과 '창을 못 재서 0점'은 전혀 다른 상황인데
            # 규칙은 둘을 구분하지 못한다. 화면에서라도 갈라 놓는다.
            sub = ("⚠️ 시작일이 없어 <b>접수 창 길이를 재지 못했습니다</b>. "
                   "가장 큰 배점(최대 30점)이 빠진 점수라, 낮게 나온 것이 "
                   "'덜 급하다'는 뜻이 아닙니다.")
        elif not result['reasons']:
            sub = ("채용 메타데이터는 있는데 적극성 신호가 하나도 없습니다. "
                   "신호가 없다는 것 자체가 관측 결과이므로 1점입니다.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(score_card(rule_score, "규칙 점수 (주 결과)", sub),
                        unsafe_allow_html=True)
        with c2:
            if MODEL is None:
                st.markdown(
                    f'<div class="card soft"><div class="score-cap">모델 점수</div>'
                    f'<div class="note">모델을 불러오지 못했습니다.<br>'
                    f'<code>{MODEL_ERR}</code></div></div>', unsafe_allow_html=True)
            else:
                mp = MODEL.predict_with_scope([raw], [source]).iloc[0]
                m_score, in_scope = int(mp['urgency_score']), bool(mp['in_scope'])
                if not in_scope:
                    msub = ("⚠️ <b>검증 범위 밖</b>입니다. 이 공고에는 채용 메타데이터가 "
                            "없어 모델이 검증된 적이 없습니다. 참고값으로만 보세요.")
                elif m_score == rule_score:
                    msub = "규칙과 일치합니다."
                else:
                    msub = (f"규칙과 {abs(m_score - rule_score)}점 다릅니다. "
                            f"이 공고는 규칙이 계산 가능하므로 <b>규칙 쪽이 정답</b>이고, "
                            f"이 차이는 모델의 근사 오차입니다.")
                st.markdown(score_card(m_score, "모델 점수 (대조용)", msub),
                            unsafe_allow_html=True)

        # ---- 근거 ----
        st.markdown("#### 채점 근거")
        if result['reasons']:
            for r in result['reasons']:
                st.markdown(f'<div class="reason">{r}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="card soft note">본문에서 적극성 신호를 찾지 못했습니다. '
                '채용 메타데이터는 있으므로 <b>신호가 없다는 것 자체가 관측 결과</b>이고, '
                '그래서 1점입니다.</div>', unsafe_allow_html=True)

        # ---- 추출된 사실 ----
        st.markdown("#### 공고에서 뽑은 사실")
        left, right = st.columns(2)
        with left:
            rows = [
                ("접수 시작일", f"{posting['start_date']:%Y.%m.%d}"
                 if posting['start_date'] else "—"),
                ("마감일", f"{posting['deadline']:%Y.%m.%d}"
                 if posting['deadline'] else "—"),
                ("접수 창 길이", f"{result['window_days']}일"
                 if result['window_days'] is not None else "계산 불가"),
                ("모집인원", f"{posting['headcount']}명"
                 if posting['headcount'] is not None else "미공개"),
            ]
            st.markdown('<div class="card">' + ''.join(
                f'<div class="kv"><span>{k}</span><span>{v}</span></div>'
                for k, v in rows) + '</div>', unsafe_allow_html=True)
        with right:
            wp = (STATS or {}).get('window_percentiles', {})
            note = ""
            if result['window_days'] is not None and wp:
                w = result['window_days']
                note = (f"데이터셋 {STATS['window_n']:,}건의 접수 창 길이는 "
                        f"중앙값 {wp['50']}일, 하위 10%가 {wp['10']}일입니다. "
                        f"이 공고는 {w}일이라 "
                        + ("<b>평균보다 짧습니다</b>." if w < int(wp['50'])
                           else "평균 이상입니다."))
            st.markdown(
                f'<div class="card soft"><div class="note">'
                f'<b>근거 점수</b> {result["evidence_score"]}점 '
                f'(0점→1등급, 43점 이상→5등급)<br><br>{note}</div></div>',
                unsafe_allow_html=True)

        # ---- 진단 ----
        if posting['diagnostics']:
            st.markdown("#### 이 분석에서 주의할 점")
            for d in posting['diagnostics']:
                st.markdown(f'<div class="card warn note">{d}</div>',
                            unsafe_allow_html=True)

        # ---- 원문 ----
        with st.expander(f"규칙이 실제로 읽은 텍스트 보기 ({len(raw):,}자)"):
            st.markdown(
                '<div class="note">사이트에서 긁은 HTML을 그대로 쓰지 않고, '
                '구조화 필드(시작일·마감일·모집인원)를 <b>학습 데이터와 같은 형식으로 '
                '다시 조립</b>한 결과입니다. 앞부분의 <code>접수기간 · 방법 시작일 … '
                '마감일 …</code>이 조립된 헤더입니다.<br>'
                '점수가 이상하면 여기서 확인하세요 — 사이트 개편으로 엉뚱한 값이 '
                '들어가면 여기에 드러납니다.</div>', unsafe_allow_html=True)
            st.code(clean_body(raw, source)[:6000] or "(비어 있음)")


# =========================================================
# 탭 2 — 어떻게 만들었나
# =========================================================
with tab_how:
    st.markdown("""
### 1. 무엇을 재고 있나

원래 데이터에는 `urgency_score`(시급성)가 있었지만 **84.7%가 3점**이라 변별력이
없었습니다. 본문을 실측해보니 `급구`(0.06%)·`충원/결원`(0.62%) 같은 전통적인
긴급성 어휘가 거의 없어서, "시급성"을 그대로 복원하는 건 불가능했습니다.

그래서 본문에서 **실제로 관측 가능한 것**만 묶어 **채용 적극성(hiring
aggressiveness)** 으로 다시 정의했습니다.

| 신호 | 근거 |
|---|---|
| 접수 창 길이 | 3일 안에 닫는 공고와 30일 열어두는 공고는 적극성이 다르다 |
| 모집 규모 | 모집인원, 다수 모집 |
| 즉시성 | 급구·긴급채용, 즉시 입사·투입 요구 |
| 결원 대체 | 결원·충원·공석 |
| 마감 압박 | 조기 마감·채용 시 마감 |
| 보상 유인 | 합격축하금·사이닝 보너스 |

의도적으로 **뺀** 것도 있습니다. `신입가능/경력무관`은 `experience_level`과
사실상 같은 변수라, 넣으면 "연차별 적극성" 분석이 순환논리가 됩니다.

---

### 2. 모델은 왜 만들었고, 왜 두 번째 자리인가

라벨 `urgency_score`는 규칙이 본문에서 정규식으로 만든 **결정론적 값**입니다.
즉 `라벨 = f(본문)`이고, 같은 본문을 주면 규칙을 돌려서 정확도 100%를 얻습니다.
**규칙이 계산 가능한 공고에서 모델은 정의상 규칙의 근사치**입니다.

그래서 이 앱은 규칙 점수를 주 결과로 놓고 모델을 대조용으로 둡니다.
모델이 존재 이유를 갖는 자리는 하나뿐입니다 — **규칙이 계산할 수 없는 공고**
(전체의 24.2%, 마감일·모집인원이 아예 없는 공고). 다만 그 자리에서 모델이
믿을 만한지는 아직 증명되지 않았습니다(→ 한계 탭).

---

### 3. 처음 점수의 79%는 거품이었다

첫 모델은 Macro F1 **0.8595**를 기록했지만, 검증 설계를 뜯어보니 상당 부분이
평가 설계에서 나온 거품이었습니다. 통제를 하나씩 넣으면서 점수가 어떻게
움직였는지가 이 프로젝트의 실제 내용입니다.

| 단계 | Macro F1 | 빠진 것 |
|---|---|---|
| v1 (중복 미제거 · 랜덤 split) | 0.8595 | — |
| \\+ 중복 공고 그룹 분리 | 0.8459 | −0.0136 · 같은 공고를 train/test 양쪽에서 봄 |
| \\+ 라벨 산출 구간 마스킹 | 0.7158 | −0.1301 · 정규식 역공학 |
| \\+ 사이트 템플릿 제거 | 0.6620 | −0.0538 · 사이트 지문 |

**0.8595 중 0.1975(23%)가 거품**이었습니다.

- **중복** — 정규화 후 전문 일치 6.97%, 앞 600자 일치 11.2%. 라벨이 텍스트의
  결정론적 함수라 이게 train/test에 갈리면 암기로 맞힙니다. `StratifiedGroupKFold`로 묶었습니다.
- **라벨 누출** — 라벨 산출에 쓴 구간을 마스킹하고 재학습해 누출 규모를 쟀습니다.
  모델이 강할수록 의존도가 컸습니다: LinearSVC 7.3% / LogReg 6.8% / XGBoost 10.0%.
- **사이트 템플릿** — v1의 중요 피처 TOP 20이 전부 `등록일`·`잡코리아 이력서` 같은
  템플릿이었습니다. 특히 **jobkorea 본문의 72%가 꼬리(추천공고 목록 = 다른 공고의 내용)**
  였습니다. `max_df=0.9`로는 안 걸립니다 — 전체 코퍼스 DF가 0.247밖에 안 되기 때문에,
  소스 *내부* DF를 봐야 잡힙니다.
- **베이스라인** — Dummy → source-only → LinearSVC → XGBoost 사다리를 놓으니
  LinearSVC가 3초에 XGBoost(32분)의 대부분을 따라잡았습니다.

---

### 4. 라벨을 두 번 고쳤다 (v3)

모델을 아무리 손봐도 안 되는 지점이 있었습니다. **한 소스로 배워 다른 소스를
예측하면 QWK 0.04 — 사실상 무작위**였습니다. 원인은 모델이 아니라 라벨이었습니다.

**[수정 1] 지원자 수를 모집인원으로 오인**

jobkorea 본문 꼬리의 `지원자 현황 통계 지원자 수 8 명 모집인원 ○○ 명`에서
`8 명 모집`이 잡혀 "모집인원 8명"으로 읽혔습니다. jobkorea는 모집인원을 `○○`로
가리기 때문에 정식 패턴이 비고 바로 옆 지원자 수가 대신 걸립니다.
**의미가 정반대**입니다 — 지원자가 많으면 덜 급한 건데 +20점이 붙었습니다.

실측 4,165건에서 발생, measurable 30,569행 중 **3,486행(11.4%)의 등급**이 바뀝니다.
그것도 `4점→2점` 1,487건처럼 2등급짜리 오차입니다.

**[수정 2] 소스마다 다른 물리량을 재고 있었다**

| | v2가 잰 것 |
|---|---|
| jobkorea | `남은기간 N일` = 마감일 − **크롤링한 날** |
| saramin | `접수기간 시작~종료` = 마감일 − 시작일 |

같은 라벨 이름을 붙였지만 **다른 양**입니다. 게다가 `남은기간`은 공고의 속성이
아니라 *내가 언제 봤는가*의 속성입니다 — 하루 뒤에 크롤링하면 라벨이 달라집니다.

접수 창 길이는 양쪽에서 뽑히고 분포도 거의 같습니다
(jobkorea p50=30일 / saramin p50=30일). 그래서 v3은 **창 길이 하나로 통일**했습니다.

결과 — 같은 개념의 소스 간 분포가 실제로 맞춰졌습니다.
(measurable 30,569행 = 모델이 학습하는 모집단 기준)

| 상위 등급(4+5점) 비율 | jobkorea | saramin | 격차 |
|---|---|---|---|
| v2 | 45.2% | 2.0% | **22.9배** |
| v3 | 5.4% | 2.0% | **2.7배** |

잔여 격차 2.7배는 접수 창 정보 보유율 차이(jobkorea 76.3% / saramin 68.3%)에서
옵니다. 라벨 정의가 아니라 데이터 커버리지 문제입니다.

**곁가지 — 앵커를 어디에 두느냐**

`채용 시 마감`(+10)을 어디서 찾을지 세 번 바꿨습니다.

| 시도 | 범위 | 검출 | 판정 |
|---|---|---|---|
| 1 | 본문 전체 | 1,335행 | 과탐 |
| 2 | `접수기간` 뒤 90자 | 0행 | **미탐 — 앵커가 틀림** |
| 3 | `시작일…마감일` 매치 + 40자 | 1,335행 | 맞음 |

2번이 0행인 이유가 핵심입니다. jobkorea 본문에는 `접수기간`이 두 번 나오는데,
상단 탭 레이블(`상세요강 접수기간∙방법 기업정보`)이 먼저 걸려서 90자 창이 실제
메타데이터가 아니라 탭 바를 보고 있었습니다. 실제 블록은 이렇게 생겼습니다.

```
시작일 2026.06.09(화) 마감일 2026.06.15(월) 채용 시 마감 접수방법 잡코리아 즉시지원
```

`채용 시 마감`은 문장이 아니라 **접수 조건 필드의 값**입니다.
라벨 작업에서 규칙만큼이나 앵커가 중요하다는 사례라 남깁니다.

---

### 5. 이 앱이 URL을 읽는 방법

사이트를 긁어서 나온 HTML을 그대로 쓰지 않습니다. 학습 데이터의 `raw_text`와
모양이 다르기 때문입니다.

| 사이트 | 문제 | 해법 |
|---|---|---|
| jobkorea | `남은기간`을 JS로 그림 | 렌더된 `시작일/마감일` 블록 파싱, 실패 시 `<meta>` 폴백 |
| saramin | 데스크톱에 시작일 없음, 본문은 별도 엔드포인트 | 시작일은 **모바일 페이지**, 본문은 `view-detail` |
| wanted | 마감일·모집인원이 구조적으로 없음 | 측정 불가로 판정하고 그렇게 표시 |

뽑은 구조화 필드는 **학습 데이터와 같은 형식으로 다시 조립**해서
(`접수기간 · 방법 시작일 YYYY.MM.DD 마감일 YYYY.MM.DD 모집인원 N 명`)
규칙과 모델이 학습 때와 같은 문자열을 보게 합니다.
그 조립 결과는 분석 탭의 "규칙이 실제로 읽은 텍스트 보기"에서 전부 확인할 수 있습니다.
""")


# =========================================================
# 탭 3 — 한계와 개선점
# =========================================================
with tab_limit:
    if MODEL is not None:
        m = MODEL.test_metrics
        t = MODEL.transfer_summary
        if m:
            cols = st.columns(5)
            for col, (k, label) in zip(cols, [
                    ('qwk', 'QWK (주 지표)'), ('mae', 'MAE'),
                    ('off_by_1', '±1점 이내'), ('macro_f1', 'Macro F1'),
                    ('accuracy', 'Accuracy')]):
                col.metric(label, f"{m[k]:.4f}")
            st.markdown(
                f'<div class="note">measurable hold-out {MODEL.meta["split"]["test"]:,}행 기준. '
                f'라벨이 1~5 순서형이므로 QWK와 MAE가 주 지표입니다 — '
                f'"1점을 5점으로" 틀린 것과 "3점을 4점으로" 틀린 것은 같은 오류가 아닙니다.'
                f'</div>', unsafe_allow_html=True)
            st.write("")
        if t:
            st.markdown(
                f"**cross-source 전이 QWK(평균)** — v2 라벨 `{t['mean_qwk_v2']:.4f}` → "
                f"v3 라벨 `{t['mean_qwk_v3']:.4f}` "
                f"(`{t['mean_qwk_v3'] - t['mean_qwk_v2']:+.4f}`). "
                f"라벨 수정의 효과는 확인됐지만 **여전히 쓸 수 있는 수준은 아닙니다** "
                f"— 아래 한계 3 참조.")
            st.write("")

    st.markdown("""
### 확인된 한계

**1. 규칙은 규칙이지, 검증된 진실이 아니다**

라벨은 사람이 붙인 정답이 아니라 정규식의 출력입니다. 지금 측정할 수 있는 건
"규칙을 얼마나 잘 복제하는가"뿐이고, **규칙 자체가 맞는지는 확인할 방법이 없습니다.**
접수 창이 짧은 공고가 실제로 더 적극적인지는 아무도 검증하지 않았습니다.

이 프로젝트에서 가장 큰 개선은 모델이 아니라 **사람이 붙인 소량 정답 세트
(300~500건)** 입니다. 그게 없으면 위의 QWK 숫자는 "규칙 복제율"이라고 읽어야 합니다.

**2. 메타데이터 없는 공고(24.2%)의 라벨은 지금도 근거가 없다**

마감일·모집인원이 없는 공고는 어휘를 세어 3점에서 가산하는 폴백으로 라벨링됩니다.
같은 정답 위에서 재보면:

| 방법 | QWK | MAE |
|---|---|---|
| 상수 (항상 3점) | 0.0000 | 1.1924 |
| 현행 어휘 폴백 | 0.1071 | **2.0093** |

**폴백이 상수보다 MAE가 나쁩니다.** 어휘 가산이 실질적으로 아무 일도 하지 않습니다.
v2 라벨에서도(1.6610 vs 1.2199), v3 라벨에서도 같은 결론입니다.
데이터셋의 24.2%가 이 방식으로 라벨링돼 있고, 원티드 공고는 전부 여기 해당합니다.

**3. 그런데 모델이 그 자리를 대신할 수도 없다**

v3의 라벨 수정으로 cross-source 전이 QWK가 `0.0486 → 0.1309`(2.7배)로 올랐습니다.
라벨 진단이 맞았다는 증거입니다. **하지만 여기서 멈추면 과장입니다.**

QWK도 MAE도 라벨 분포에 의존하는데 v2와 v3은 분포가 다릅니다. 그래서 각 라벨
세트에 **자기 상수 베이스라인**(train 라벨 중앙값을 항상 찍기)을 깔고 다시 쟀습니다.

| 라벨 | 상수 대비 MAE 개선 | QWK |
|---|---|---|
| v2 | +16.6% | 0.0529 |
| v3 | **−30.1%** | 0.1307 |

**전이 모델은 "항상 2점 찍기"보다 MAE가 나쁩니다.** `class_weight` 탓인가 싶어
가중치 없는 모델도 돌렸지만 −26.3%로 같았습니다.

모순이 아닙니다 — 모델은 **순서(ranking)는 배웠지만 점 예측(calibration)이 나쁩니다.**
QWK는 순서를 보상하고, MAE는 한 클래스에 몰린 분포에서 예측을 퍼뜨리는 것을 벌합니다.

결론: 라벨 수정은 전이를 "사실상 무작위"에서 "약한 신호는 있음"으로 옮겼을 뿐,
**"쓸 수 있음"으로는 옮기지 못했습니다.** 학습에 없던 사이트에 이 모델을 쓰는 것은
여전히 근거가 없고, 앱이 `in_scope=False`를 크게 표시하는 이유입니다.

**4. 스크래핑은 조용히 깨진다**

- **이미지 공고** — 사람인·잡코리아 모두 본문 전체를 이미지 한 장으로 올리는 공고가
  흔합니다. 텍스트가 거의 없으므로 분석이 무의미합니다. OCR은 넣지 않았습니다.
  (본문 300자 미만이면 경고를 띄웁니다)
- **사이트 개편** — 정규식 기반이라 셀렉터보다는 버티지만 결국 깨집니다.
  깨졌을 때 알아챌 수 있도록 "규칙이 실제로 읽은 텍스트"를 전부 노출합니다.
- **재현이지 동일이 아님** — 조립한 `raw_text`는 학습 데이터의 크롤링 스냅샷과
  100% 같은 문자열이 아닙니다.

**5. 측정 방법 자체의 제약**

- 단일 hold-out입니다. 교차검증 기반 신뢰구간이 없습니다.
- 한국어 형태소 분석을 쓰지 않았습니다. 어절 단위라 `학력은`/`학력이`가 별개
  피처가 됩니다.
- 학습·평가 모두 데이터 스냅샷 기준입니다. 공고가 갱신되면 재학습이 필요합니다.

---

### 다음에 할 일 (효과가 큰 순서)

| 순위 | 할 일 | 왜 |
|---|---|---|
| 1 | **사람이 붙인 정답 300~500건** | 지금 측정 가능한 것이 "규칙 복제율"뿐인 근본 원인. 이게 있어야 규칙이 맞는지 처음으로 검증됨 |
| 2 | **어휘 폴백 폐기** | 상수보다 나쁘다는 것이 이미 측정됨. 24.2%를 `null`로 두는 편이 정직함 |
| 3 | 신호 간 이중 계상 정리 | `채용 시 마감`이 rolling(+10)과 조기마감(+12)에 모두 걸려 한 문구로 22점. v2도 그랬고, 가중치를 바꾸면 v2 대비 비교가 오염돼 이번엔 두었음 |
| 4 | 사람인 시작일 커버리지 확대 | 모바일 페이지 의존이라 상시채용 공고는 여전히 창 길이 없음 |
| 5 | Kiwi·Mecab 형태소 분석 | 어휘 크기와 성능 모두 개선 여지 |
| 6 | 교차검증 신뢰구간 | 지금 숫자에 오차 범위가 없음 |
| 7 | 이미지 공고 OCR | 커버리지 문제이지 정확도 문제가 아님 |

---

<div class="note">
이 앱의 점수는 <b>공고 본문에서 관측되는 신호</b>의 요약이지, 그 회사가 실제로
급하게 뽑고 있다는 증거가 아닙니다. 지원 여부를 이 점수로 결정하지 마세요.
</div>
""", unsafe_allow_html=True)
