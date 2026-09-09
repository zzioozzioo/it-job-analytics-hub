"""
boilerplate.py

사이트 템플릿(보일러플레이트) 제거. 개선 2번.

[문제]
v1 모델의 중요 피처 TOP 20이 '등록일 / 기업정보 / 잡코리아 이력서 / 유의사항 /
24시 제출서류 / 지원방법' 이었다. 전부 채용 시급성과 무관한 사이트 템플릿이다.
라벨이 source별로 다른 규칙에서 나오기 때문에(jobkorea=마감일 파싱,
saramin=접수창 길이, wanted=어휘 폴백) source를 알아맞히는 것만으로
QWK 0.4를 얻는다. 템플릿은 source를 100% 알려주는 지문(fingerprint)이다.

[실측 근거]  (master_merged_v2.json 9,000행 표본, 1~3gram)
  jobkorea: DF 0.996짜리 저작권 고지가 전 공고에 붙음
            ('본 채용정보는 잡코리아의 동의없이 무단전재 또는 재배포할 수 없으며...')
            raw_text 평균 3,020자 중 2,176자(72.1%)가 꼬리 = 추천공고 목록 + 고지
            -> 꼬리에 '다른 공고'의 문구가 들어 있어 이 공고의 신호로 오인된다
  saramin : DF 0.70~0.82짜리 정형 고지문
            ('입사지원 서류에 허위사실이 발견될 경우, 채용확정 이후라도...')
  wanted  : 전용 템플릿이 사실상 없음. 상위 n-gram이 전부 실제 기술 용어
            (설계/기반/경험/api/aws). 직무기술서라 템플릿이 붙지 않는다.

[왜 TF-IDF의 max_df로는 안 잡히는가]
max_df=0.9는 전체 코퍼스 기준이다. jobkorea 저작권 고지의 전체 DF는
0.996 x 0.248(jobkorea 비중) = 0.247 이라 가볍게 통과한다.
source '안에서의' DF를 봐야 잡힌다. 그래서 fit_source_stopwords()를 둔다.

[두 층위]
  1) strip_template()      : 구조적 제거. 꼬리 절단 + 정형 고지문 삭제.
                             텍스트 자체를 줄이므로 다른 신호도 함께 보존된다.
  2) fit_source_stopwords(): 규칙이 놓친 잔여 템플릿을 통계로 잡는다.
                             반드시 train에서만 fit 한다(test 누출 방지).

[주의: masked와의 역할 분리]
라벨 산출 구간 제거(masked)와 템플릿 제거(clean)는 다른 통제다.
일부 문장은 둘 다에 해당한다(예: '본 채용은 수시 진행으로 채용 시 마감될 수
있습니다'). 이 겹침을 숨기지 않으려고 overlap_report()로 양을 따로 보고한다.
완전 통제 조건은 masked+clean 이다.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# rescore_urgency.clean_body 의 꼬리 마커를 단일 출처로 재사용한다.
_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))
try:
    from rescore_urgency import _JK_TAIL_MARKERS, _JK_DISCLAIMER
except ImportError:  # 리포지토리 밖에서 임포트될 때의 폴백
    _JK_TAIL_MARKERS = ['본 채용정보는', '로그인 하고 비슷한 조건의', '관련 태그']
    _JK_DISCLAIMER = re.compile(
        r'마감일은\s*기업의\s*사정으로\s*인해\s*조기\s*마감\s*또는\s*변경될\s*수\s*있습니다')


# ---------------------------------------------------------------------------
# 1. 구조적 제거
# ---------------------------------------------------------------------------
# jobkorea 헤더 네비게이션: 본문 앞에 붙는 탭/링크 이름들.
_JK_NAV = re.compile(
    r'상세요강\s*접수기간[^\s]*방법\s*기업정보\s*추천공고'
    r'|채용정보에\s*잘못된\s*내용이\s*있을\s*경우\s*문의\s*해주세요\.?'
    r'|지도보기|스크랩|즉시\s*지원|TOP\s*궁금해요'
)

# saramin 정형 고지문. 실측 빈도 순(4,000행 표본에서 0.5% 이상 등장한 것 위주).
# 채용 내용이 아니라 '모든 공고에 붙는 안내문'만 넣는다.
_SR_TEMPLATE = re.compile(
    r'입사지원\s*서류에\s*허위사실이\s*발견될\s*경우[^.]*?취소될\s*수\s*있습니다\.?'
    r'|지원서\s*및\s*기타\s*제출자료\s*내용에\s*허위사실을\s*포함할\s*경우[^.]*?있습니다\.?'
    r'|모집분야별로\s*마감일이\s*상이할\s*수\s*있으니\s*유의하시길\s*바랍니다\.?'
    r'|학력,?\s*성별,?\s*연령을\s*보지\s*않는\s*블라인드\s*채용입니다\.?'
    r'|채용절차의\s*공정화에\s*관한\s*법률[^.]*?반환하며[^.]*?있습니다\.?'
    r'|채용서류\s*반환\s*청구\s*안내'
    r'|정보\s*수정이\s*필요할\s*경우\s*고객센터[^.]*?문의해\s*주세요\.?'
    r'|정확한\s*상세요강은\s*반드시\s*채용\s*홈페이지에서[^.]*?바랍니다\.?'
    r'|서류전형\s*결과는\s*합격자에\s*한하여\s*개별\s*통보됩니다\.?'
    r'|(?:국가등록장애인|보훈대상자|취업보호대상자)[^.]*?우대(?:합니다|됩니다)\.?'
    r'|이력서양식\s*:\s*사람인\s*온라인\s*이력서'
    r'|접수방법\s*:\s*사람인\s*(?:온라인\s*)?입사지원'
    r'|좋은\s*포지션을\s*추천드립니다\.?'
    r'|필요\s*시\s*레퍼런스\s*체크,?\s*인성검사가\s*진행\s*될\s*수\s*있습니다\.?'
)

# 잡코리아/사람인 브랜드명 자체도 source 지문이다.
_BRAND = re.compile(r'잡코리아|사람인|saramin|jobkorea|원티드|wanted', re.I)


def strip_template(text, source):
    """source별 템플릿을 제거한 본문을 돌려준다.

    jobkorea는 페이지 전체 덤프라 꼬리 72%가 추천공고 목록(=다른 공고 내용)이다.
    이걸 남겨두면 이 공고와 무관한 문구를 이 공고의 신호로 학습한다."""
    t = text or ''
    if not t:
        return ''

    if source == 'jobkorea':
        cut = len(t)
        for mk in _JK_TAIL_MARKERS:
            i = t.find(mk)
            if i != -1:
                cut = min(cut, i)
        t = t[:cut]
        t = _JK_DISCLAIMER.sub(' ', t)
        t = _JK_NAV.sub(' ', t)
    elif source == 'saramin':
        t = _SR_TEMPLATE.sub(' ', t)

    t = _BRAND.sub(' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def strip_series(texts, sources):
    """pandas Series 쌍에 strip_template을 적용한다."""
    return pd.Series([strip_template(t, s) for t, s in zip(texts, sources)],
                     index=texts.index)


# ---------------------------------------------------------------------------
# 2. 통계 기반 잔여 템플릿 제거 (source-wise DF)
# ---------------------------------------------------------------------------
def _label_signal_tokens():
    """라벨 산출에 쓰인 정규식에서 한글 리터럴을 뽑아 '보호 대상' 어휘를 만든다.

    첫 실험에서 stopword 목록에 '모집인원 / 접수기간 / 고용형태'가 들어가면서
    urgency QWK가 0.7960 -> 0.7421로 떨어졌다. 템플릿이 아니라 라벨 신호를
    지운 것이다. clean(템플릿 제거)이 masked(라벨 구간 제거)의 일을 대신하면
    두 조건을 분리해서 볼 수 없게 되므로, 여기서 명시적으로 보호한다.

    LABEL_SOURCE_PATTERNS에서 자동으로 뽑기 때문에 패턴이 바뀌면 같이 따라간다."""
    from check_label_leakage import LABEL_SOURCE_PATTERNS
    toks = set()
    for p in LABEL_SOURCE_PATTERNS:
        toks.update(re.findall(r'[가-힣]{2,}', p))
    return toks


def fit_source_stopwords(texts, sources, min_df=0.5, min_gap=0.4, max_terms=3000):
    """어떤 source 안에서 DF가 높으면서 source 간 편차가 큰 단어를 stopword로.

    판정: max_source_DF >= min_df  AND  (max_source_DF - min_source_DF) >= min_gap
      - 앞 조건: 그 source에서 거의 모든 공고에 등장 (= 템플릿)
      - 뒤 조건: 다른 source에는 거의 없음 (= source 지문)
    둘 다 만족해야 자른다. '개발'처럼 모든 source에 흔한 일반어는 gap이 작아 남는다.

    ⚠️ 반드시 train 텍스트로만 호출할 것. test로 fit 하면 그 자체가 누출이다."""
    cv = CountVectorizer(binary=True, min_df=10, max_features=80000)
    X = cv.fit_transform(texts)
    names = np.array(cv.get_feature_names_out())
    src = np.asarray(sources)

    per_source = []
    for s in pd.unique(src):
        m = src == s
        if m.sum() == 0:
            continue
        per_source.append(np.asarray(X[m].sum(axis=0)).ravel() / m.sum())
    if len(per_source) < 2:
        return []

    P = np.vstack(per_source)
    hi, lo = P.max(axis=0), P.min(axis=0)
    sel = (hi >= min_df) & ((hi - lo) >= min_gap)
    idx = np.argsort(-(hi - lo)[sel])
    cand = list(names[sel][idx])

    # 라벨 신호 어휘는 stopword에서 뺀다. 그건 masked 조건이 담당한다.
    protect = _label_signal_tokens()
    kept = [t for t in cand if not any(p in t for p in protect)]
    return kept[:max_terms]


def overlap_report(texts, sources, label_mask_fn):
    """clean이 제거한 구간 중 '라벨 산출 구간'과 겹치는 양을 보고한다.

    clean(템플릿 제거)과 masked(라벨 구간 제거)는 다른 통제인데 일부 문장이
    양쪽에 해당한다. 이 겹침을 숨기면 두 조건의 비교가 오염된다."""
    n_removed = n_label_in_removed = 0
    for t, s in zip(texts, sources):
        kept = strip_template(t, s)
        removed = len(t) - len(kept)
        if removed <= 0:
            continue
        n_removed += removed
        # 제거 전후로 라벨 패턴 매치 수가 얼마나 줄었는지로 근사한다
        before = label_mask_fn(t).count('<MASK>')
        after = label_mask_fn(kept).count('<MASK>')
        n_label_in_removed += max(0, before - after)
    return {'removed_chars': int(n_removed),
            'label_matches_removed': int(n_label_in_removed)}
