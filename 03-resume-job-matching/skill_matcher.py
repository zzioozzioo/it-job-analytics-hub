"""
skill_matcher.py — 이력서 텍스트를 받아 공고를 추천한다.

담당: 심지우 · 참고 문서: TASK-skill-matching.md, HANDOFF-skill-matching.md

---------------------------------------------------------------------------
지금 이 파일의 상태 — 4단계(동점 처리·정규화) 완료
---------------------------------------------------------------------------
    STEP 3  IDF 가중 합산                                          [완료]
    STEP 4  동점 처리·정규화 — 5개 후보 구현, 기본값은 length_norm    [완료]
    STEP 5  설계 결정 4개 플래그의 실제 분기 로직 (지금은 값만 받고 안 씀) [TODO]

STEP 4에서 만든 것: raw_idf(대조군) · length_norm(HANDOFF 검증 승자) ·
jaccard · cosine · per_skill_count. `score_mode`로 고르고, 어느 게 나은지는
`compare_score_modes()`로 eval_proxy를 돌려 표로 확인한다 — 코드가 대신
고르지 않는다("플래그로 두고 프록시로 고른다"는 이 저장소의 방식 그대로).

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
import sys
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
    "hypernym_match": False,          # TODO STEP 5: True일 때 분기 구현
    # 스킬 0개 이력서: 빈 결과 vs 텍스트 유사도 폴백
    "empty_resume_fallback": False,   # TODO STEP 5: True일 때 폴백 경로 구현
    # compute_related_techs 식 암묵 확장: Django만 있어도 Python 추가할지
    "implicit_related": False,        # TODO STEP 5: True일 때 01/app.py:503 로직 연결
    # 비사전 스킬(엑셀·정보처리기사 등)을 매칭 키에 포함할지
    # 참고: 김민석 실측 65.7% vs 68.7%로 오히려 더 나빴음 — 기본 False 유지
    "use_skills_extra": False,
}


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
    """STEP 3과 동일 — 정규화 없음. 비교 대조군."""
    return sum(idf.get(t, 0.0) for t in hit)


def _score_length_norm(hit, mine, job_techs, idf):
    """HANDOFF 문서에서 이미 검증된 방식: 겹침 점수를 공고 전체 요구사항의
    "규모"로 나눈다. 공고가 기술을 많이 요구할수록(=이력서가 그중 일부만
    맞혀도 얻는 점수가) 나눠지므로, "적게 요구하는데 정확히 맞는 공고"가
    "많이 요구하는데 일부만 맞는 공고"보다 위로 올라온다.
    실측(HANDOFF): Recall@10 68.0% -> 95.0% (요구 스킬 100% 보유 조건)."""
    denom = math.sqrt(sum(idf.get(t, 0.0) for t in job_techs)) or 1.0
    return sum(idf.get(t, 0.0) for t in hit) / denom


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
    dot = sum(idf.get(t, 0.0) ** 2 for t in hit)
    mine_norm = math.sqrt(sum(idf.get(t, 0.0) ** 2 for t in mine)) or 1.0
    job_norm = math.sqrt(sum(idf.get(t, 0.0) ** 2 for t in job_techs)) or 1.0
    return dot / (mine_norm * job_norm)


def _score_per_skill_count(hit, mine, job_techs, idf):
    """length_norm의 단순한 버전 — IDF가 아니라 그냥 "요구 기술 개수"로
    나눈다. 계산이 더 단순해 length_norm과 비교했을 때 IDF 기반 정규화가
    실제로 추가 이득이 있는지 확인하는 대조군 역할."""
    return sum(idf.get(t, 0.0) for t in hit) / len(job_techs) if job_techs else 0.0


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

    score_mode  SCORERS 중 하나. 기본값 "length_norm"은 HANDOFF 문서에서
                이미 Recall@10 68.0%->95.0%로 확인된 방식이다. 다른 방식과
                비교하려면 compare_score_modes()를 쓴다.
    """
    if score_mode not in SCORERS:
        raise ValueError(f"알 수 없는 score_mode: {score_mode!r}. "
                         f"가능한 값: {', '.join(SCORERS)}")
    scorer = SCORERS[score_mode]

    flags = {**DEFAULT_FLAGS, **(flags or {})}
    _, idf, pool = _load(as_of=as_of)

    mine = extract_techs(resume_text)
    if flags["use_skills_extra"]:
        # TODO STEP 5: skills_extra 어휘와도 매칭시키려면 여기서 이력서 쪽
        # 비사전 스킬도 뽑아야 한다 (지금은 표준 기술명만 뽑음).
        pass

    if not mine:
        if flags["empty_resume_fallback"]:
            pass  # TODO STEP 5: 텍스트 유사도 폴백 (4단계 임베딩과 역할 겹침 — 논의 중)
        return []

    if flags["implicit_related"]:
        pass  # TODO STEP 5: compute_related_techs로 mine을 확장

    scored = []
    for _, row in pool.iterrows():
        techs = row["techs"]
        if flags["use_skills_extra"]:
            techs = techs | row["skills_extra"]
        hit = mine & techs
        if not hit:
            continue
        if not _passes_filters(row, filters):
            continue
        sc = scorer(hit, mine, techs, idf)
        scored.append((sc, str(row["job_id"]), sorted(hit)))

    # job_id를 2차 키로 둬서 완전 동점이어도 실행할 때마다 순서가 안 바뀌게
    # 한다. 이게 동점 "해결"은 아니다 — 그건 score_mode가 하는 일이고,
    # 이 정렬은 그러고도 남는 동점의 순서를 재현 가능하게만 만든다.
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [
        (job_id, sc, {"matched_techs": hit, "matched_count": len(hit)})
        for sc, job_id, hit in scored[:top_k]
    ]


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