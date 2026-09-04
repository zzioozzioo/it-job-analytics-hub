import streamlit as st
import pandas as pd
import json
import os
from collections import Counter

import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# =========================================================
# 0. 페이지 설정
# =========================================================
st.set_page_config(
    page_title="IT 채용공고 분석 대시보드",
    layout="wide",
    page_icon="🧭",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# 0-1. 브랜드 팔레트
#      ⚠️ .streamlit/config.toml 의 [theme] 값과 1:1로 맞춰둔 상수.
#         차트 / 지도 / 워드클라우드가 전부 여기서 색을 가져가므로,
#         테마를 바꿀 때는 config.toml과 이 블록만 고치면 전체가 따라온다.
# ---------------------------------------------------------
PRIMARY = "#2563EB"        # config.toml primaryColor (Blue 600)
PRIMARY_LIGHT = "#93C5FD"  # Blue 300
PRIMARY_SOFT = "#EFF6FF"   # Blue 50  — 활성 탭/뱃지 배경
PRIMARY_DARK = "#1E3A8A"   # Blue 900
ACCENT = "#F59E0B"         # Amber 500 — 보조 강조(지역 상세 등)
INK = "#0F172A"            # config.toml textColor (Slate 900)
MUTED = "#64748B"          # Slate 500 — 보조 텍스트/축
SURFACE = "#F1F5F9"        # config.toml secondaryBackgroundColor (Slate 100)
BORDER = "#E2E8F0"         # Slate 200

# 워드클라우드용 브랜드 그라데이션 (밝은 블루 → 딥 네이비)
BRAND_CMAP = LinearSegmentedColormap.from_list(
    "brand_blue", [PRIMARY_LIGHT, "#3B82F6", PRIMARY, PRIMARY_DARK]
)

# ---------------------------------------------------------
# 0-2. Matplotlib 전역 스타일
#      💡 Malgun Gothic(Windows 전용) 대신 NanumGothic 사용
#         배포 환경(Streamlit Cloud 등, Linux)에서는 packages.txt의 fonts-nanum이 설치해줌
#      배경을 투명으로 두어 Streamlit 카드 배경 위에 자연스럽게 얹히도록 함
# ---------------------------------------------------------
plt.rcParams.update({
    'font.family': 'NanumGothic',
    'axes.unicode_minus': False,
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'savefig.facecolor': 'none',
    'savefig.transparent': True,
    'text.color': INK,
    'axes.labelcolor': MUTED,
    'axes.edgecolor': BORDER,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'font.size': 10.5,
    'axes.titlesize': 12,
    'axes.titleweight': 'semibold',
    'axes.titlecolor': INK,
    'axes.titlepad': 14,
})

current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, '..', '.env')
load_dotenv(dotenv_path)

KAKAO_API_KEY = os.environ.get("KAKAO_API_KEY", "")


# =========================================================
# 0-3. 커스텀 CSS 주입
# =========================================================
st.markdown(
    f"""
    <style>
    /* ---------- 기본 UI 정리 ---------- */
    /* 우측 상단 햄버거 메뉴 + 하단 워터마크 숨김
       (Streamlit 버전에 따라 셀렉터가 다르므로 구/신 셀렉터를 함께 지정) */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    [data-testid="stToolbar"] {{display: none !important;}}
    [data-testid="stDecoration"] {{display: none !important;}}
    [data-testid="stStatusWidget"] {{display: none !important;}}

    /* ⚠️ stHeader는 position:fixed로 상단 약 3.75rem을 계속 점유한다.
       툴바만 숨기고 이 바를 남겨두면, block-container의 padding-top을 줄였을 때
       첫 요소(헤더 타이틀)가 이 바 아래로 파고들어 윗부분이 잘린다.
       -> 바 자체를 0 높이로 접어서 상단 공간을 실제로 회수한다. */
    [data-testid="stHeader"],
    .stAppHeader {{
        height: 0 !important;
        min-height: 0 !important;
        background: transparent !important;
    }}

    /* ---------- 레이아웃 여백 ---------- */
    .block-container {{
        /* stHeader를 접었으므로 이 값이 곧 실제 상단 여백이 된다 */
        padding-top: 3rem;
        padding-bottom: 4rem;
        max-width: 1480px;
    }}

    /* ---------- 헤더 (Hero) ---------- */
    .app-header {{
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px 20px;
        padding: 0 0 18px 0;
        margin-bottom: 26px;
        border-bottom: 1px solid {BORDER};
        overflow: visible;
    }}
    /* 좁아지면 배지가 아래로 내려가고 타이틀이 폭을 온전히 쓰도록 */
    .app-header__left {{
        flex: 1 1 340px;
        min-width: 0;   /* flex 기본값(auto)이면 콘텐츠보다 못 줄어들어 넘친다 */
    }}
    .app-header__title {{
        /* 고정 크기 대신 clamp로 유동 -> 좁은 뷰포트에서도 넘치지 않음 */
        font-size: clamp(1.3rem, 2.4vw, 1.8rem);
        /* 800은 한글 폰트에 실제 웨이트가 없어 합성 볼드로 번지며 잘릴 수 있음 */
        font-weight: 700;
        letter-spacing: -0.01em;
        color: {INK};
        /* 한글 폰트의 ascent+descent가 1.25em을 넘겨 위아래가 잘리는 것 방지 */
        line-height: 1.45;
        margin: 0;
        padding-bottom: 2px;
        word-break: keep-all;      /* 한글은 어절 단위로 줄바꿈 */
        overflow-wrap: break-word;
        overflow: visible;
    }}
    .app-header__sub {{
        font-size: 0.88rem;
        color: {MUTED};
        margin-top: 4px;
        line-height: 1.5;
        word-break: keep-all;
    }}
    .app-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        flex: 0 1 auto;
        min-width: 0;
        max-width: 100%;
        background: {PRIMARY_SOFT};
        color: {PRIMARY};
        border: 1px solid {PRIMARY_LIGHT};
        border-radius: 999px;
        padding: 6px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        /* 레포명이 길어도 타이틀을 밀어내지 않고 자기 쪽에서 말줄임 */
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    @media (max-width: 820px) {{
        .app-header {{ align-items: flex-start; }}
        .app-header__left {{ flex: 1 1 100%; }}
    }}

    /* ---------- 지표 카드 (st.metric) ---------- */
    [data-testid="stMetric"],
    [data-testid="metric-container"] {{
        background: #FFFFFF;
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 18px 20px 16px 20px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, .04),
                    0 1px 3px rgba(15, 23, 42, .06);
        transition: transform .18s ease,
                    box-shadow .18s ease,
                    border-color .18s ease;
    }}
    /* hover 시 살짝 떠오르는 인터랙션 */
    [data-testid="stMetric"]:hover,
    [data-testid="metric-container"]:hover {{
        transform: translateY(-4px);
        border-color: {PRIMARY_LIGHT};
        box-shadow: 0 12px 24px rgba(37, 99, 235, .10),
                    0 4px 8px rgba(15, 23, 42, .06);
    }}
    [data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: .01em;
    }}
    [data-testid="stMetricValue"] {{
        color: {INK} !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }}

    /* ---------- 탭 ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        border-bottom: 1px solid {BORDER};
        margin-bottom: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 46px;
        padding: 0 20px;
        border-radius: 10px 10px 0 0;
        background: transparent;
        color: {MUTED};
        font-weight: 600;
        font-size: 0.94rem;
        transition: background .15s ease, color .15s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background: {SURFACE};
        color: {PRIMARY};
    }}
    /* 활성 탭: 배경 + 텍스트 + 하단 인디케이터 3중으로 구분 */
    .stTabs [aria-selected="true"] {{
        background: {PRIMARY_SOFT} !important;
        color: {PRIMARY} !important;
        box-shadow: inset 0 -3px 0 {PRIMARY};
    }}
    /* baseweb 기본 밑줄 하이라이트 제거 (위 inset shadow로 대체) */
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {{
        display: none;
    }}
    .stTabs [data-baseweb="tab-panel"] {{
        padding-top: 22px;
    }}

    /* ---------- 컨테이너(패널) ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 14px;
        border-color: {BORDER};
        background: #FFFFFF;
    }}

    /* ---------- 타이포그래피 위계 ---------- */
    /* 섹션 제목 (st.subheader) */
    h3 {{
        font-size: 1.18rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
        color: {INK};
        padding-top: 0 !important;
    }}
    /* 패널 내부 소제목 (st.markdown "###") */
    h4 {{
        font-size: 0.97rem !important;
        font-weight: 700 !important;
        color: {INK};
        margin-bottom: 4px !important;
    }}
    [data-testid="stCaptionContainer"] p {{
        color: {MUTED};
        font-size: 0.84rem;
        line-height: 1.55;
    }}

    /* ---------- 위젯 다듬기 ---------- */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        overflow: hidden;
    }}
    div[data-baseweb="select"] > div {{
        border-radius: 9px;
        border-color: {BORDER};
    }}
    /* 지도 iframe 모서리 정리 */
    iframe[title="streamlit_folium.st_folium"] {{
        border-radius: 12px;
        border: 1px solid {BORDER};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _nanum_font_path():
    """WordCloud에 넘겨줄 NanumGothic ttf 실제 경로를 찾는다.
    fonts-nanum(packages.txt)이 설치돼 있어야 정상적으로 찾아짐."""
    try:
        return fm.findfont('NanumGothic', fallback_to_default=False)
    except Exception:
        return None


NANUM_FONT_PATH = _nanum_font_path()


# =========================================================
# 1. 허깅페이스 데이터셋 로딩 (data/master_merged.json 대체)
# =========================================================
HF_REPO_ID = "data-craftee/korean-it-recruit-dataset"
HF_FILENAME = "master_merged_v2.json"


def _get_hf_token():
    """Private 레포 대비 토큰 조회. Public 레포면 None이어도 정상 동작."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        return st.secrets.get("HF_TOKEN", None)
    except Exception:
        return None


@st.cache_data(show_spinner="허깅페이스에서 master_merged.json 다운로드 중...")
def load_master_df():
    token = _get_hf_token()
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        repo_type="dataset",
        token=token,
    )
    with open(local_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    df = pd.DataFrame(records)

    # final_techs가 리스트가 아닌 값(NaN 등)이 섞여 있을 수 있으니 방어적으로 정리
    df['final_techs'] = df['final_techs'].apply(lambda v: v if isinstance(v, list) else [])
    return df


try:
    df_master = load_master_df()
except Exception as e:
    st.error(f"허깅페이스 데이터셋을 불러오지 못했습니다: {e}")
    st.stop()

EXPERIENCE_ORDER = ["신입", "1~3년차", "3~5년차", "5년이상", "경력무관"]
EXPERIENCE_LABELS = {
    "신입": "🌱 신입",
    "1~3년차": "🚀 1~3년차",
    "3~5년차": "🔥 3~5년차",
    "5년이상": "👑 5년이상",
    "경력무관": "🌀 경력무관",
}

SOURCES = ["wanted", "jobkorea", "saramin"]

# 💡 EXCLUDE_TECH (불용어) — streamlit.py에서 그대로 가져옴
#    final_techs에 남아 있는 직무/일반 키워드는 연관 분석 시 노이즈가 커서 제외 옵션으로 제공
EXCLUDE_TECH = {
    '소프트웨어개발', '솔루션', 'SI', '시스템', '네트워크', '서버', '정보보안', 'Sm', '데이터', 'erp',
    '문서작성', '클라이언트', '유지보수', '방화벽', 'Ms office', '기술지원', '영어', '검증', '모델링',
    '전략기획', '회로설계', '재고관리', '아키텍처', '매출관리', '인터페이스', 'GUI', 'PPT', 'PM', '회계', '고객관리',
    '핀테크', '모바일앱개발', '문서관리', '보안관제', 'HTTP', '반응형웹', '포토샵'
}


# =========================================================
# 2. [렉 해결] 카카오 지오코딩: 파일 기반 영구 캐싱 + 병렬 호출
#    (location_kakao_preprocessing.py 로직 그대로 재사용)
# =========================================================
COORDS_CACHE_FILE = os.path.join(current_dir, 'kakao_coords_cache.json')


def _load_coords_cache():
    """디스크에 저장된 좌표 캐시를 읽어온다. 앱을 껐다 켜도 유지됨."""
    if os.path.exists(COORDS_CACHE_FILE):
        try:
            with open(COORDS_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_coords_cache(cache):
    try:
        with open(COORDS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except IOError as e:
        st.warning(f"좌표 캐시 저장 실패: {e}")


def _fetch_single_coordinate(loc, api_key):
    """지역 1개에 대한 카카오 API 호출 (스레드에서 병렬 실행됨)"""
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    try:
        response = requests.get(url, headers=headers, params={'query': loc}, timeout=5)
        if response.status_code != 200:
            return loc, None, f"API 에러 {response.status_code}"
        result = response.json()
        if result.get('documents'):
            y = float(result['documents'][0]['y'])
            x = float(result['documents'][0]['x'])
            return loc, [y, x], None
        return loc, None, None
    except Exception as e:
        return loc, None, str(e)


@st.cache_data
def get_coordinates_dict(locations, api_key):
    cache = _load_coords_cache()

    # 이미 캐시(파일)에 있는 지역은 API 재호출 없이 그대로 사용
    to_fetch = [loc for loc in locations if loc and loc != '원격근무' and loc not in cache]

    if to_fetch and api_key:
        # 순차 호출 -> 병렬 호출 (최대 10개 동시 요청)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(_fetch_single_coordinate, loc, api_key): loc for loc in to_fetch}
            for future in as_completed(futures):
                loc, coords, error = future.result()
                if error:
                    st.warning(f"'{loc}' 좌표 변환 실패: {error}")
                elif coords:
                    cache[loc] = coords

        # 새로 받아온 좌표를 파일에 반영 -> 다음 실행부터는 API 호출 자체가 생략됨
        _save_coords_cache(cache)
    elif to_fetch and not api_key:
        st.info(f"KAKAO_API_KEY가 없어 캐시에 없는 {len(to_fetch)}개 지역은 지도에 표시되지 않습니다. (기존 캐시는 정상 사용)")

    # ⚠️ 캐시 조회는 API 키 유무와 무관하게 항상 수행 (캐시 파일만으로도 대부분 커버됨)
    return {loc: cache[loc] for loc in locations if loc in cache}


def build_map(region_group):
    """공고 수에 따라 마커 크기/농도를 달리하는 지도.
    마커 색은 브랜드 PRIMARY 계열로 통일해 차트와 톤을 맞춘다."""
    m = folium.Map(location=[36.5, 127.5], zoom_start=7, tiles='CartoDB positron')

    max_count = region_group['job_count'].max() or 1

    for _, row in region_group.iterrows():
        radius = min(max(row['job_count'] * 0.2, 5), 30)
        # 공고가 많은 지역일수록 진하게 (0.35 ~ 0.75)
        opacity = 0.35 + 0.40 * (row['job_count'] / max_count)

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius,
            color=PRIMARY,
            weight=1.5,
            fill=True,
            fill_color=PRIMARY,
            fill_opacity=opacity,
            opacity=0.9,
            tooltip=f"{row['location']} · 공고 {row['job_count']}건"
        ).add_child(folium.Popup(row['location'])).add_to(m)

    return m


# =========================================================
# 3. 공통 유틸
# =========================================================
def tech_counter(df, col='final_techs'):
    all_techs = []
    for cell in df[col]:
        if isinstance(cell, list):
            all_techs.extend(cell)
    return Counter(all_techs)


@st.cache_data(show_spinner="연관 기술 인덱스 계산 중...")
def build_cooccurrence_index(sources: tuple, exclude_stopwords: bool):
    """공고별 기술 스택 집합 리스트 + 전체 빈도 카운터를 만든다.
    streamlit.py의 processed_jobs / all_raw_techs 구성 로직과 동일하다.
    (df_master는 세션 내에서 불변이므로 sources/불용어 옵션만 캐시 키로 사용)"""
    df = df_master[df_master['source'].isin(sources)]

    jobs = []
    counter = Counter()
    for cell in df['final_techs']:
        if not isinstance(cell, list):
            continue
        techs = set(cell)  # 한 공고 내 중복 제거
        if exclude_stopwords:
            techs = {t for t in techs if t not in EXCLUDE_TECH}
        if techs:
            jobs.append(techs)
            counter.update(techs)
    return jobs, counter


def compute_related_techs(jobs, base_tech, universe):
    """base_tech가 포함된 공고를 훑어 함께 등장한 기술의 빈도를 센다.
    universe(상위 기술 집합) 안의 기술만 집계해 꼬리 노이즈를 걷어낸다."""
    related = Counter()
    match_count = 0
    for techs in jobs:
        if base_tech in techs:
            match_count += 1
            for t in techs:
                if t != base_tech and t in universe:
                    related[t] += 1
    return match_count, related


def _style_axes(ax):
    """차트 잡음 제거: 위/오른쪽 축선을 없애고 옅은 세로 그리드만 남긴다."""
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(BORDER)
    ax.grid(axis='x', color=BORDER, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def bar_chart_top_techs(counter: Counter, top_n: int, title: str, color: str = PRIMARY):
    top = counter.most_common(top_n)
    if not top:
        st.info("표시할 기술 스택 데이터가 없습니다.")
        return
    df_top = pd.DataFrame(top, columns=['기술 스택', '빈도수'])
    df_top.index = df_top.index + 1

    fig, ax = plt.subplots(figsize=(7.4, max(3.6, len(df_top) * 0.34)))
    plot_df = df_top.sort_values(by='빈도수', ascending=True)
    ax.barh(plot_df['기술 스택'], plot_df['빈도수'], color=color, height=0.68, zorder=3)

    ax.set_title(title, loc='left')
    ax.set_xlabel('출현 빈도 (건)')
    _style_axes(ax)
    fig.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close(fig)

    return df_top


def rank_table(df_top, extra_pct_col=None):
    """순위 테이블. 빈도수는 막대(ProgressColumn)로 표현해 스캔하기 쉽게 만든다."""
    col_cfg = {
        "빈도수": st.column_config.ProgressColumn(
            "빈도수",
            format="%d",
            min_value=0,
            max_value=int(df_top['빈도수'].max()),
        )
    }
    if extra_pct_col:
        col_cfg[extra_pct_col] = st.column_config.NumberColumn(extra_pct_col, format="%.1f%%")

    st.dataframe(df_top, width='stretch', column_config=col_cfg)


def render_wordcloud(counter: Counter, title: str, colormap=BRAND_CMAP):
    if not counter:
        st.info("워드클라우드를 그릴 데이터가 없습니다.")
        return
    if NANUM_FONT_PATH is None:
        st.info("한글 폰트가 없어 워드클라우드는 생략합니다. (packages.txt의 fonts-nanum 설치 필요)")
        return

    wc = WordCloud(
        font_path=NANUM_FONT_PATH,
        background_color=None,   # 카드 배경과 자연스럽게 합성되도록 투명 처리
        mode='RGBA',
        width=800,
        height=560,
        colormap=colormap,
        prefer_horizontal=0.92,
        margin=4,
    ).generate_from_frequencies(dict(counter))

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, loc='left')
    fig.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close(fig)


# =========================================================
# 4. 헤더
# =========================================================
st.markdown(
    f"""
    <div class="app-header">
      <div class="app-header__left">
        <div class="app-header__title">IT 채용공고 기술 스택 대시보드</div>
        <div class="app-header__sub">
          wanted · jobkorea · saramin 통합 분석 — 총 <b>{len(df_master):,}</b>건의 채용공고
        </div>
      </div>
      <div class="app-badge">🤗 {HF_REPO_ID}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if NANUM_FONT_PATH is None:
    st.warning("⚠️ NanumGothic 폰트를 찾지 못했습니다. packages.txt(fonts-nanum) 설치 여부를 확인해주세요.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 기술스택 통계", "🔗 연관 기술 분석", "🗺️ 지역별 통계", "📈 연차별 통계"]
)

# ---------------------------------------------------------
# TAB 1. 기술스택 통계
# ---------------------------------------------------------
with tab1:
    st.subheader("전체 기술 스택 빈도")
    st.caption("세 개 채용 플랫폼에서 수집한 공고의 `final_techs`를 합산한 기술 수요 분포입니다.")

    with st.container(border=True):
        selected_sources = st.multiselect(
            "소스 필터",
            options=SOURCES,
            default=SOURCES,
            key="tab1_source_filter",
        )

    if not selected_sources:
        st.warning("최소 1개 이상의 소스를 선택해주세요.")
    else:
        df_tab1 = df_master[df_master['source'].isin(selected_sources)]
        counter1 = tech_counter(df_tab1)

        st.write("")
        m1, m2, m3 = st.columns(3, gap="large")
        m1.metric("선택된 공고 수", f"{len(df_tab1):,} 건")
        m2.metric("고유 기술 스택 수", f"{len(counter1):,} 개")
        m3.metric("포함된 소스", f"{len(selected_sources)} / {len(SOURCES)}")

        st.write("")
        top_n = st.slider("표시할 상위 기술 개수", min_value=5, max_value=50, value=20, step=5, key="tab1_top_n")

        st.write("")
        col_bar, col_wc = st.columns([1.35, 1], gap="large")

        with col_bar:
            with st.container(border=True):
                st.markdown(f"#### 상위 기술 스택 TOP {top_n}")
                df_top = bar_chart_top_techs(counter1, top_n, f"기술 스택 TOP {top_n}")
                if df_top is not None:
                    rank_table(df_top)

        with col_wc:
            with st.container(border=True):
                st.markdown("#### 워드클라우드")
                render_wordcloud(Counter(dict(counter1.most_common(100))), "기술 수요 분포")

# ---------------------------------------------------------
# TAB 2. 연관 기술 분석 (동시 출현)
# ---------------------------------------------------------
with tab2:
    st.subheader("기술 스택 연관성 분석")
    st.caption("선택한 기술을 포함한 공고에서 **함께 요구된 기술**의 순위를 보여줍니다.")

    with st.container(border=True):
        col_ctl_a, col_ctl_b = st.columns([1.7, 1], gap="large", vertical_alignment="bottom")
        with col_ctl_a:
            rel_sources = st.multiselect(
                "소스 필터",
                options=SOURCES,
                default=SOURCES,
                key="tab2_source_filter",
            )
        with col_ctl_b:
            exclude_stopwords = st.checkbox(
                "불용어 제외",
                value=True,
                key="tab2_exclude",
                help="'시스템', '네트워크', 'SI' 등 기술 스택으로 보기 어려운 일반 키워드를 제외합니다.",
            )

    if not rel_sources:
        st.warning("최소 1개 이상의 소스를 선택해주세요.")
    else:
        jobs, rel_counter = build_cooccurrence_index(tuple(rel_sources), exclude_stopwords)
        top_skills = [t for t, _ in rel_counter.most_common(50)]

        if not top_skills:
            st.info("분석할 기술 스택이 없습니다.")
        else:
            st.write("")
            base_tech = st.selectbox(
                "기준 기술 스택 (상위 50개)",
                options=top_skills,
                key="tab2_base_tech",
            )

            match_count, related = compute_related_techs(jobs, base_tech, set(top_skills))

            st.write("")
            r1, r2, r3 = st.columns(3, gap="large")
            r1.metric(f"'{base_tech}' 포함 공고", f"{match_count:,} 건")
            r2.metric("함께 등장한 기술", f"{len(related):,} 개")
            r3.metric(
                "공고당 평균 동반 기술",
                f"{sum(related.values()) / match_count:.1f} 개" if match_count else "N/A",
            )

            if not related:
                st.info(f"'{base_tech}'와 동시에 출현한 다른 기술 스택 데이터가 없습니다.")
            else:
                st.write("")
                top_n_rel = st.slider(
                    "표시할 연관 기술 개수", min_value=5, max_value=30, value=15, step=5, key="tab2_top_n"
                )

                st.write("")
                col_rel_bar, col_rel_wc = st.columns([1.35, 1], gap="large")

                with col_rel_bar:
                    with st.container(border=True):
                        st.markdown(f"#### '{base_tech}' 연관 기술 TOP {top_n_rel}")
                        df_rel = bar_chart_top_techs(
                            related, top_n_rel, f"{base_tech}와 함께 요구된 기술"
                        )
                        if df_rel is not None:
                            # 동시 출현율 = 해당 기술을 요구한 공고 중 이 기술도 함께 요구한 비율
                            df_rel['동시 출현율'] = (df_rel['빈도수'] / match_count * 100).round(1)
                            rank_table(df_rel, extra_pct_col='동시 출현율')

                with col_rel_wc:
                    with st.container(border=True):
                        st.markdown("#### 연관 스택 워드클라우드")
                        render_wordcloud(related, f"{base_tech} 연관 스택")

# ---------------------------------------------------------
# TAB 3. 지역별 통계
# ---------------------------------------------------------
with tab3:
    st.subheader("지역별 기술 스택 분포")

    df_loc_all = df_master.copy()
    excluded_count = df_loc_all['location'].isna().sum()
    df_map_src = df_loc_all.dropna(subset=['location']).copy()

    if not KAKAO_API_KEY:
        st.warning("KAKAO_API_KEY가 설정되어 있지 않습니다. 캐시에 없는 신규 지역은 지도에 표시되지 않습니다.")

    if df_map_src.empty:
        st.info("지도에 표시할 지역 데이터가 없습니다.")
    else:
        unique_locations = tuple(df_map_src['location'].unique())
        dynamic_coords = get_coordinates_dict(unique_locations, KAKAO_API_KEY)

        df_map_src['lat'] = df_map_src['location'].apply(lambda x: dynamic_coords.get(x, [None, None])[0])
        df_map_src['lon'] = df_map_src['location'].apply(lambda x: dynamic_coords.get(x, [None, None])[1])

        df_map = df_map_src.dropna(subset=['lat', 'lon'])

        if df_map.empty:
            st.info("좌표를 확인할 수 있는 지역이 없습니다.")
        else:
            region_group = df_map.groupby('location').agg({
                'final_techs': 'sum',
                'lat': 'first',
                'lon': 'first'
            }).reset_index()
            region_group['job_count'] = df_map.groupby('location').size().values

            st.write("")
            g1, g2, g3 = st.columns(3, gap="large")
            g1.metric("지도 표시 공고 수", f"{len(df_map):,} 건")
            g2.metric("집계된 지역 수", f"{len(region_group):,} 곳")
            g3.metric("최다 공고 지역", f"{region_group.loc[region_group['job_count'].idxmax(), 'location']}")

            st.write("")
            st.caption("지도의 **파란색 마커**를 클릭하면 해당 지역의 핵심 기술 스택이 우측에 나타납니다.")

            col_map, col_details = st.columns([1.5, 1], gap="large")

            if 'selected_region' not in st.session_state:
                st.session_state['selected_region'] = None

            with col_map:
                with st.container(border=True):
                    m = build_map(region_group)
                    map_data = st_folium(m, height=600, use_container_width=True, key="main_map")

                    if map_data.get('last_object_clicked_popup') is not None:
                        st.session_state['selected_region'] = map_data['last_object_clicked_popup']

            with col_details:
                with st.container(border=True):
                    clicked_region = st.session_state['selected_region']

                    if clicked_region:
                        st.markdown(f"#### {clicked_region} 상세 분석")
                        region_data = region_group[region_group['location'] == clicked_region]

                        if not region_data.empty:
                            techs_list = region_data.iloc[0]['final_techs']
                            job_count = region_data.iloc[0]['job_count']

                            st.metric("해당 지역 공고 수", f"{job_count:,} 건")

                            tech_cnt = Counter(techs_list)
                            filtered_tech_cnt = Counter({t: c for t, c in tech_cnt.items() if c >= 2})

                            if filtered_tech_cnt:
                                st.write("")
                                # 지역 상세는 액센트 컬러로 구분해 지도(파랑)와 역할을 나눔
                                df_top_techs = bar_chart_top_techs(
                                    filtered_tech_cnt, 10, f"{clicked_region} TOP 10 요구 스택", color=ACCENT
                                )
                                if df_top_techs is not None:
                                    rank_table(df_top_techs)
                            else:
                                st.info("이 지역에는 2회 이상 추출된 기술 스택이 없습니다.")
                    else:
                        st.markdown("#### 전국 통합 핵심 스택")
                        st.caption("👈 지도에서 지역 마커를 클릭하면 해당 지역의 상세 순위로 바뀝니다.")

                        all_techs = df_map['final_techs'].sum()
                        tech_counts = Counter(all_techs)
                        filtered_techs = Counter({t: c for t, c in tech_counts.items() if c >= 3})

                        st.write("")
                        df_nation = bar_chart_top_techs(filtered_techs, 10, "전국 TOP 10 요구 스택")
                        if df_nation is not None:
                            rank_table(df_nation)

# ---------------------------------------------------------
# TAB 4. 연차별 통계
# ---------------------------------------------------------
with tab4:
    st.subheader("연차별 기술 스택 & 채용 적극성")

    df_exp_all = df_master.copy()
    no_exp_count = df_exp_all['experience_level'].isna().sum()

    df_exp = df_exp_all.dropna(subset=['experience_level'])
    available_levels = [lvl for lvl in EXPERIENCE_ORDER if lvl in df_exp['experience_level'].unique()]

    with st.container(border=True):
        selected_level = st.radio(
            "조회할 연차 구간",
            options=available_levels,
            format_func=lambda x: EXPERIENCE_LABELS.get(x, x),
            horizontal=True,
            key="tab4_level",
        )

    df_level = df_exp[df_exp['experience_level'] == selected_level]

    st.write("")
    y1, y2, y3 = st.columns(3, gap="large")
    y1.metric("해당 구간 공고 수", f"{len(df_level):,} 건")
    y2.metric("평균 채용 적극성", f"{df_level['urgency_score'].mean():.2f}" if len(df_level) else "N/A")
    y3.metric("전체 대비 비율", f"{len(df_level) / len(df_exp) * 100:.1f} %" if len(df_exp) else "N/A")

    st.write("")
    top_n_level = st.slider("표시할 상위 기술 개수", min_value=5, max_value=30, value=15, step=5, key="tab4_top_n")

    st.write("")
    col_left, col_right = st.columns([1.35, 1], gap="large")

    with col_left:
        with st.container(border=True):
            st.markdown(f"#### {EXPERIENCE_LABELS.get(selected_level, selected_level)} — Top 기술 스택")
            counter_level = tech_counter(df_level)
            df_top_level = bar_chart_top_techs(
                counter_level, top_n_level, f"{selected_level} 요구 기술 TOP {top_n_level}"
            )  # matplotlib 제목엔 이모지 제외(폰트에 글리프 없음)
            if df_top_level is not None:
                rank_table(df_top_level)

    with col_right:
        with st.container(border=True):
            st.markdown("#### 채용 적극성 분포")
            st.caption("1(낮음) ~ 5(높음) 구간별 공고 수")
            if len(df_level):
                urgency_counts = df_level['urgency_score'].value_counts().sort_index()
                st.bar_chart(urgency_counts, color=PRIMARY, height=380)
            else:
                st.info("데이터가 없습니다.")

    st.write("")
    st.markdown("---")
    st.subheader("연차 구간별 채용 적극성 비교")
    st.caption("연차가 낮을수록 채용 적극성이 높은 공고가 많은지 등을 비교할 수 있습니다.")

    summary = df_exp.groupby('experience_level')['urgency_score'].agg(['mean', 'count']).reindex(available_levels)
    summary.columns = ['평균 채용 적극성', '공고 수']

    st.write("")
    col_sum_chart, col_sum_table = st.columns([1.35, 1], gap="large")
    with col_sum_chart:
        with st.container(border=True):
            st.markdown("#### 구간별 평균 채용 적극성")
            st.bar_chart(summary['평균 채용 적극성'], color=PRIMARY, height=340)
    with col_sum_table:
        with st.container(border=True):
            st.markdown("#### 요약 테이블")
            st.dataframe(
                summary,
                width='stretch',
                column_config={
                    "평균 채용 적극성": st.column_config.NumberColumn("평균 채용 적극성", format="%.2f"),
                    "공고 수": st.column_config.NumberColumn("공고 수", format="%d"),
                },
            )
