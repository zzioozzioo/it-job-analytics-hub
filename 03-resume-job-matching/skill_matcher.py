"""
skill_matcher.py — 이력서 텍스트를 받아 공고를 추천한다.

담당: 심지우 · 참고 문서: TASK-skill-matching.md, HANDOFF-skill-matching.md

---------------------------------------------------------------------------
지금 이 파일의 상태 — 5단계 완료 + 팀 피드백 반영 (2026-09-22)
---------------------------------------------------------------------------
    STEP 3  IDF 가중 합산                                          [완료]
    STEP 4  동점 처리·정규화 — 5개 후보 구현, length_norm이 팀 채택 확정  [완료]
    STEP 5  설계 결정 4개 플래그의 실제 분기 로직                     [완료]

김민석 리뷰(2026-09-22)에서 나온 세 가지를 반영했다:
    1. 모든 스코어러의 sum()을 정렬된 순서로 고정 — 파이썬 문자열 해시가
       프로세스마다 무작위화돼(PEP 456) set을 그냥 순회하면 부동소수점
       덧셈 순서가 실행마다 바뀐다. 동점이 지배적인 이 데이터에서는 그
       흔들림이 순위를 뒤집는다. eval_proxy에서 실제로 겪은 버그와 동일
       (Recall@10이 65.0/65.5/66.0%로 실행마다 달라졌었음).
    2. recommend() 결과에서 job_id 중복 제거 — (source, job_id) 중복
       777건 때문에 서로 다른 공고가 같은 job_id를 가질 수 있다.
    3. 기준선 수치를 2026-09-17 HANDOFF 갱신판으로 교체 (이전
       68.0%/95.0%대 값은 위 1번과 같은 종류의 버그로 무효화됨). 새 기준:
       단순 IDF 65.0%, length_norm 92.5% (Recall@10, 100% 보유 조건).

팀이 eval_proxy `--compare-norms`(6종 비교)로 이미 "sqrt(=length_norm) 유지"를
결론 내렸다 — DEFAULT_SCORE_MODE는 그 결론과 일치한다. compare_score_modes()는
그 결론을 다시 검증하고 싶을 때 쓰는 용도로 남겨둔다.

이 파일 자체는 마지막에 지워질 코드가 아니라 계속 커지는 파일이다 — 뼈대 단계라고
부실하게 짜면 위 단계에서 계속 다시 손대게 되니, "함수 하나만 비어있는" 게 아니라
데이터 로딩·캐싱·시그니처는 최종 형태로 잡아두고 점수 계산 부분만 단순하게 뒀다.

---------------------------------------------------------------------------
왜 job_pool 함수만 쓰는가
---------------------------------------------------------------------------
job_pool.load_corpus() / load_pool() / build_idf() 를 거치지 않고 데이터를 직접
읽으면, eval_proxy.py가 만드는 모집단과 여기서 보는 모집단이 어긋난다. 그러면
Recall@10 숫자를 서로 다른 실험 사이에 비교할 수 없게 된다 — HANDOFF 문서가
job_pool을 "03이 쓰는 공고 데이터의 단일 출처"라고 못박은 이유가 이것이다.

---------------------------------------------------------------------------
리소스를 모듈 레벨에서 한 번만 로드하는 이유
---------------------------------------------------------------------------
recommend()는 요청(질의)마다 호출된다. 매 호출마다 40,348건을 다시 읽고 IDF를
다시 계산하면 eval_proxy로 200~400개 질의를 돌릴 때마다 몇 분씩 걸린다. 그래서
코퍼스·IDF·풀은 지연 로딩 + 모듈 전역 캐시로 한 번만 만든다.

⚠️ 이 캐시 때문에 풀이 바뀌었는데(예: 백필 파일이 새로 생김) 같은 프로세스를
   계속 쓰면 반영이 안 된다. 새 데이터로 다시 재보려면 프로세스를 재시작하거나
   `reset_cache()`를 부른다.
"""

import math
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))

from tech_normalize import extract_techs          # noqa: E402  이력서 텍스트 -> 표준 기술명
import job_pool                                    # noqa: E402  데이터 단일 출처


# ---------------------------------------------------------------------------
# 설계 결정 4개 — STEP 5에서 분기 로직을 채운다. 지금은 기본값만.
# ---------------------------------------------------------------------------
# 회의에서 고르지 않고 "플래그로 두고 프록시로 고른다"로 정해진 항목들
# (TASK-skill-matching.md 안건 3 / HANDOFF-skill-matching.md 3번).
# 최종 비교는 eval_proxy로 김민석이 돌린다 — 여기서는 끄고 켤 수 있게만 만든다.
DEFAULT_FLAGS = {
    # 상위 개념 함의: 이력서 Spring Boot -> Spring 요구 공고도 일치로 볼지
    "hypernym_match": False,
    # 스킬 0개 이력서: 빈 결과 vs 텍스트 유사도 폴백
    # TODO: 이력서에 기술 스택이 0개인 경우 - 추천 X vs 비슷한 무언가라도 보여줌
    "empty_resume_fallback": False,
    # compute_related_techs 식 암묵 확장: Django만 있어도 Python 추가할지
    "implicit_related": False,
    # 비사전 스킬(엑셀·정보처리기사 등)을 매칭 키에 포함할지
    # 참고: 김민석 실측 65.7% vs 68.7%로 오히려 더 나빴음 — 기본 False 유지
    "use_skills_extra": False,
}


# ---------------------------------------------------------------------------
# STEP 5-1 — 상위 개념 함의 (hypernym_match)
# ---------------------------------------------------------------------------
# 팀 결정: 이력서에 "Spring Boot"가 있으면 "Spring"을 요구하는 공고도 일치로
# 본다 (README: 데이터셋에 Spring 2,505건 · Spring Boot 126건이 따로 존재).
#
# ⚠️ 이 표는 팀이 확정한 한 쌍만 시드로 들어있다. 더 추가하려면 실제 158종
#    캐논 목록(예: 실행 중 idf.keys() 출력)을 보고 팀이 판단해야 한다 —
#    "React Native -> React"처럼 그럴듯해 보여도 이 저장소의 캐논 사전이
#    실제로 그 두 이름을 어떻게 표준화하는지 확인 없이는 추측해서 넣지 않는다.
HYPERNYM_MAP = {
    "Spring Boot": "Spring",
}


def _expand_hypernyms(techs: set) -> set:
    expanded = set(techs)
    for specific, general in HYPERNYM_MAP.items():
        if specific in techs:
            expanded.add(general)
    return expanded


# ---------------------------------------------------------------------------
# STEP 5-2 — compute_related_techs 식 암묵 확장 (implicit_related)
# ---------------------------------------------------------------------------
# 01-tech-stack-wordcloud/app.py:503의 compute_related_techs()를 그대로
# 옮겨왔다 — 새 지표를 만들지 않고 검증된 걸 재사용한다.
IMPLICIT_MIN_RATIO = 0.5   # base_tech 포함 공고 중 related_tech도 있는 비율
IMPLICIT_TOP_K = 2         # 기술 하나당 최대 몇 개까지 확장할지


def compute_related_techs(jobs, base_tech, universe):
    """base_tech가 포함된 공고를 훑어 함께 등장한 기술의 빈도를 센다.
    universe(상위 기술 집합) 안의 기술만 집계해 꼬리 노이즈를 걷어낸다.
    (01-tech-stack-wordcloud/app.py:503 원본 그대로)"""
    related = Counter()
    match_count = 0
    for techs in jobs:
        if base_tech in techs:
            match_count += 1
            for t in techs:
                if t != base_tech and t in universe:
                    related[t] += 1
    return match_count, related


def _build_related_map(corpus, idf):
    """{기술: [강하게 동반되는 기술들]}. IMPLICIT_MIN_RATIO 이상만 남긴다.

    universe를 idf 어휘(158종)로 제한하는 이유는 01번과 같다 — 꼬리 노이즈
    제거. 계산은 한 번만 하고 _CACHE에 저장한다(158 * 40,348건 스캔은
    몇 초 걸리지만 매 recommend() 호출마다 할 일은 아니다)."""
    universe = set(idf)
    jobs = [canon for canon, _ in job_pool.techs_series(corpus)]
    related_map = {}
    for base in universe:
        match_count, related = compute_related_techs(jobs, base, universe)
        if not match_count:
            continue
        strong = [(t, c) for t, c in related.items()
                  if c / match_count >= IMPLICIT_MIN_RATIO]
        strong.sort(key=lambda x: -x[1])
        if strong:
            related_map[base] = [t for t, _ in strong[:IMPLICIT_TOP_K]]
    return related_map


def _expand_related(techs: set, related_map: dict) -> set:
    expanded = set(techs)
    for t in techs:
        expanded.update(related_map.get(t, ()))
    return expanded


# ---------------------------------------------------------------------------
# STEP 5-3 — 스킬 0개 이력서 폴백 (empty_resume_fallback)
# ---------------------------------------------------------------------------
# ⚠️ 이 플래그의 채택 여부는 아직 팀 결정이 안 났다 — "빈 결과로 둘지 텍스트
#    유사도 폴백을 둘지"가 논의 중이고, 폴백을 두면 4단계 임베딩과 역할이
#    겹친다는 우려가 있었다. 결정에 쓸 판단 자료가 있어야 하니 최소한의
#    버전(단어 겹침 Jaccard)만 만들어 둔다 — 이게 최종안이라는 뜻이 아니다.
#    진짜 텍스트 유사도(임베딩)가 들어오면 이 함수는 교체될 가능성이 높다.
_WORD_RE = re.compile(r"[A-Za-z가-힣0-9]+")


def _tokenize(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text or "") if len(w) >= 2}


def _text_fallback(resume_text, pool, filters, top_k):
    mine_words = _tokenize(resume_text)
    if not mine_words:
        return []
    scored = []
    for _, row in pool.iterrows():
        if not _passes_filters(row, filters):
            continue
        job_words = _tokenize(row.get("raw_text") or "")
        if not job_words:
            continue
        union = mine_words | job_words
        sc = len(mine_words & job_words) / len(union) if union else 0.0
        if sc > 0:
            scored.append((sc, str(row["job_id"])))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [
        (job_id, sc, {"matched_techs": [], "matched_count": 0,
                      "fallback": "text_overlap"})
        for sc, job_id in scored[:top_k]
    ]


# ---------------------------------------------------------------------------
# STEP 5-4 — 비사전 스킬 포함 여부 (use_skills_extra)
# ---------------------------------------------------------------------------
# 이력서 쪽에서 "엑셀"·"정보처리기사" 같은 비사전 스킬을 뽑으려면, job_pool의
# split_skills()처럼 구조화된 목록이 없으니(이력서는 자유 텍스트) 코퍼스에
# 나오는 비사전 스킬 어휘를 미리 모아두고 부분 문자열로 찾는다.
EXTRA_VOCAB_MIN_DF = 5


def _build_extras_vocab(corpus):
    df = Counter()
    for _, extra in job_pool.techs_series(corpus):
        df.update(extra)
    return {t for t, c in df.items() if c >= EXTRA_VOCAB_MIN_DF}


def _extract_extras(resume_text, vocab):
    text = (resume_text or "").lower()
    return {t for t in vocab if t in text}


# ---------------------------------------------------------------------------
# STEP 4 — 동점 처리·정규화 후보들
# ---------------------------------------------------------------------------
# HANDOFF 문서의 핵심 발견: IDF 단순 합산은 어휘가 158종뿐이라 겹침 패턴이
# 금방 반복돼 동점이 대량 발생한다(질의 200건 중 88건에서 동점 11건 이상).
# 아래는 그 동점을 줄이는 후보 정규화 방식들이다. eval_proxy로 비교해서
# 실제로 쓸 것을 SCORE_MODE에서 고른다 — "이번 라운드의 핵심 설계 결정".
#
# 각 함수는 (hit, mine, job_techs, idf) -> float 을 받는다.
#   hit        mine ∩ job_techs (겹치는 기술)
#   mine       이력서에서 뽑은 기술 집합
#   job_techs  그 공고가 요구하는 기술 전체 집합 (겹치지 않는 것 포함)
#   idf        {기술명: idf}
def _score_raw_idf(hit, mine, job_techs, idf):
    """STEP 3과 동일 — 정규화 없음. 비교 대조군.

    ⚠️ sorted(hit)로 순서를 고정한다. 파이썬 문자열 해시는 프로세스마다
    무작위화돼(PEP 456) set을 그냥 순회하면 덧셈 순서가 실행마다 바뀌고,
    부동소수점 덧셈은 결합법칙이 안 맞아 마지막 비트가 흔들린다. 동점이
    수십~수백 건씩 몰리는 이 데이터에서는 그 흔들림이 순위를 뒤집는다 —
    김민석이 eval_proxy에서 실제로 겪은 버그(Recall@10이 실행마다
    65.0/65.5/66.0%로 달라짐)와 같은 원인이다. 아래 다른 스코어러들도 동일."""
    return sum(idf.get(t, 0.0) for t in sorted(hit))


def _score_length_norm(hit, mine, job_techs, idf):
    """HANDOFF 문서에서 이미 검증된 방식이자, 팀이 실제로 채택을 확정한
    방식이다(eval_proxy `--compare-norms`, 6종 비교 결론: "현행 sqrt 유지").
    겹침 점수를 공고 전체 요구사항의 "규모"로 나눈다. 공고가 기술을 많이
    요구할수록(=이력서가 그중 일부만 맞혀도 얻는 점수가) 나눠지므로, "적게
    요구하는데 정확히 맞는 공고"가 "많이 요구하는데 일부만 맞는 공고"보다
    위로 올라온다.

    실측(HANDOFF, 2026-09-17 갱신 — 이전 68.0%/95.0%대 판은 프록시 자체의
    비결정성 버그로 무효화됐다): 100% 보유 조건 Recall@10 65.0% -> 92.5%,
    MRR 0.461 -> 0.728."""
    denom = math.sqrt(sum(idf.get(t, 0.0) for t in sorted(job_techs))) or 1.0
    return sum(idf.get(t, 0.0) for t in sorted(hit)) / denom


def _score_jaccard(hit, mine, job_techs, idf):
    """자카드 유사도 — IDF를 아예 안 쓰고 집합 겹침 비율만 본다.
    이력서 스킬이 많을수록(mine이 커질수록), 공고가 요구하는 게 적을수록
    유리해진다. IDF 계열과 성격이 달라 비교 기준으로 넣어둔다."""
    union = mine | job_techs
    return len(hit) / len(union) if union else 0.0


def _score_cosine(hit, mine, job_techs, idf):
    """IDF를 벡터 가중치로 쓴 코사인 유사도. length_norm과 달리 이력서 쪽
    벡터의 크기(mine의 희귀도 총합)도 분모에 들어간다 — 희귀 기술을 많이
    아는 이력서일수록 어지간한 겹침으로는 상대적으로 점수가 덜 오른다."""
    dot = sum(idf.get(t, 0.0) ** 2 for t in sorted(hit))
    mine_norm = math.sqrt(sum(idf.get(t, 0.0) ** 2 for t in sorted(mine))) or 1.0
    job_norm = math.sqrt(sum(idf.get(t, 0.0) ** 2 for t in sorted(job_techs))) or 1.0
    return dot / (mine_norm * job_norm)


def _score_per_skill_count(hit, mine, job_techs, idf):
    """length_norm의 단순한 버전 — IDF가 아니라 그냥 "요구 기술 개수"로
    나눈다. 계산이 더 단순해 length_norm과 비교했을 때 IDF 기반 정규화가
    실제로 추가 이득이 있는지 확인하는 대조군 역할."""
    return sum(idf.get(t, 0.0) for t in sorted(hit)) / len(job_techs) if job_techs else 0.0


SCORERS = {
    "raw_idf": _score_raw_idf,
    "length_norm": _score_length_norm,     # 기본값 — HANDOFF에서 이미 검증됨
    "jaccard": _score_jaccard,
    "cosine": _score_cosine,
    "per_skill_count": _score_per_skill_count,
}
DEFAULT_SCORE_MODE = "length_norm"


# ---------------------------------------------------------------------------
# 리소스 캐시 — 지연 로딩, 모듈 전역에 한 번만
# ---------------------------------------------------------------------------
_CACHE = {}


def _load(as_of="2026-06-20", version="v4"):
    """corpus/idf/pool을 한 번만 만들어 전역에 쌓아둔다.

    as_of가 다르면 pool의 status(open/closed/rolling/unknown)가 달라지므로
    as_of별로 따로 캐시한다. corpus/idf는 as_of와 무관해 한 번만 만든다.
    """
    if "corpus" not in _CACHE:
        _CACHE["corpus"] = job_pool.load_corpus(version)
        _CACHE["idf"] = job_pool.build_idf(_CACHE["corpus"])

    key = ("pool", as_of, version)
    if key not in _CACHE:
        _CACHE[key] = job_pool.load_pool(
            as_of=as_of, corpus=_CACHE["corpus"], version=version
        )
    return _CACHE["corpus"], _CACHE["idf"], _CACHE[key]


def reset_cache():
    """데이터가 바뀐 뒤(백필 도착 등) 다시 로드하고 싶을 때 호출."""
    _CACHE.clear()


def _get_related_map(corpus, idf):
    if "related_map" not in _CACHE:
        _CACHE["related_map"] = _build_related_map(corpus, idf)
    return _CACHE["related_map"]


def _get_extras_vocab(corpus):
    if "extras_vocab" not in _CACHE:
        _CACHE["extras_vocab"] = _build_extras_vocab(corpus)
    return _CACHE["extras_vocab"]


# ---------------------------------------------------------------------------
# 필터 — "값 없으면 통과" (TASK-skill-matching.md 안건 2에서 확정된 설계)
# ---------------------------------------------------------------------------
def _passes_filters(row, filters):
    if not filters:
        return True
    for field in ("experience_level", "location"):
        want = filters.get(field)
        have = row.get(field)
        if want and have and want != have:      # 둘 다 값이 있을 때만 비교
            return False
    return True


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------
def recommend(resume_text: str, top_k: int = 10, filters: dict = None,
              flags: dict = None, as_of: str = "2026-06-20",
              score_mode: str = DEFAULT_SCORE_MODE) -> list:
    """이력서 텍스트 -> [(job_id, score, evidence), ...]  점수 내림차순.

    score_mode  SCORERS 중 하나. 기본값 "length_norm"은 팀이 eval_proxy
                `--compare-norms`(6종 비교)로 채택을 확정한 방식이다.
                실측(2026-09-17 갱신판): Recall@10 65.0%->92.5%(100% 보유
                조건). 다른 방식과 비교하려면 compare_score_modes()를 쓴다.
    """
    if score_mode not in SCORERS:
        raise ValueError(f"알 수 없는 score_mode: {score_mode!r}. "
                         f"가능한 값: {', '.join(SCORERS)}")
    scorer = SCORERS[score_mode]

    flags = {**DEFAULT_FLAGS, **(flags or {})}
    corpus, idf, pool = _load(as_of=as_of)

    # --- 이력서에서 기술 뽑기 + STEP 5 플래그 적용 ---
    mine = extract_techs(resume_text)

    if flags["hypernym_match"]:
        mine = _expand_hypernyms(mine)

    if flags["implicit_related"]:
        related_map = _get_related_map(corpus, idf)
        mine = _expand_related(mine, related_map)

    mine_extra = set()
    if flags["use_skills_extra"]:
        extras_vocab = _get_extras_vocab(corpus)
        mine_extra = _extract_extras(resume_text, extras_vocab)

    eff_mine = mine | mine_extra

    if not eff_mine:
        if flags["empty_resume_fallback"]:
            return _text_fallback(resume_text, pool, filters, top_k)
        return []

    # --- 채점 ---
    scored = []
    for _, row in pool.iterrows():
        job_techs = row["techs"]
        if flags["use_skills_extra"]:
            job_techs = job_techs | row["skills_extra"]
        hit = eff_mine & job_techs
        if not hit:
            continue
        if not _passes_filters(row, filters):
            continue
        sc = scorer(hit, eff_mine, job_techs, idf)
        scored.append((sc, str(row["job_id"]), sorted(hit)))

    # job_id를 2차 키로 둬서 완전 동점이어도 실행할 때마다 순서가 안 바뀌게
    # 한다. 이게 동점 "해결"은 아니다 — 그건 score_mode가 하는 일이고,
    # 이 정렬은 그러고도 남는 동점의 순서를 재현 가능하게만 만든다.
    scored.sort(key=lambda x: (-x[0], x[1]))

    # job_pool 리포트에 적힌 (source, job_id) 중복 777건 때문에, 서로 다른
    # 공고인데 job_id 문자열이 같은 경우가 있다. 여기서 완전히 구분할 방법이
    # 없으니(HANDOFF의 job_id 시그니처를 그대로 유지하는 한), 점수가 높은
    # 쪽만 남기고 스킵한다 — top_k 자리가 같은 id로 중복 소모되는 걸 막는
    # 최소한의 조치다. 근본 해결은 job_pool이 진짜 고유 식별자를 주는 것.
    seen = set()
    result = []
    for sc, job_id, hit in scored:
        if job_id in seen:
            continue
        seen.add(job_id)
        result.append((job_id, sc, {"matched_techs": hit, "matched_count": len(hit)}))
        if len(result) >= top_k:
            break
    return result


def compare_score_modes(as_of="2026-06-20", n=200, k=10, ratios=(1.0, 0.6, 0.4)):
    """SCORERS의 모든 정규화 방식을 eval_proxy로 나란히 채점한다.

    브리프의 "플래그로 두고 프록시로 고른다" 방식 그대로 — 어떤 정규화가
    나은지 여기서 정하지 않고, 표를 뽑아서 판단 근거로 남긴다.
    """
    from eval_proxy import evaluate  # noqa: E402  지연 import, CLI 전용 경로

    results = {}
    for mode in SCORERS:
        print(f"\n===== score_mode = {mode} =====")

        def _bound(resume_text, top_k=10, filters=None, _mode=mode):
            return recommend(resume_text, top_k=top_k, filters=filters,
                             as_of=as_of, score_mode=_mode)

        results[mode] = evaluate(_bound, n=n, k=k, ratios=ratios, as_of=as_of)
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-modes", action="store_true",
                    help="SCORERS 5개를 eval_proxy로 나란히 채점")
    ap.add_argument("--as-of", default="2026-06-20")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()

    if a.compare_modes:
        compare_score_modes(as_of=a.as_of, n=a.n)
    else:
        # 수동 스모크 테스트. 실제 채점은 eval_proxy.evaluate(recommend)로 한다.
        sample = "파이썬과 장고로 결제 API를 개발했고 AWS에 배포했습니다."
        for job_id, sc, ev in recommend(sample, top_k=5, as_of=a.as_of):
            print(f"  {job_id}  score={sc:.3f}  {ev}")