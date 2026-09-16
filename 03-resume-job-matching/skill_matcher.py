"""
skill_matcher.py — 이력서 텍스트를 받아 공고를 추천한다.

담당: 심지우 · 참고 문서: TASK-skill-matching.md, HANDOFF-skill-matching.md

---------------------------------------------------------------------------
지금 이 파일의 상태 — 2단계(뼈대)
---------------------------------------------------------------------------
아래 항목은 아직 안 채워져 있다. 각 TODO 자리에서 다음 단계 작업을 이어간다.

    STEP 3  IDF 가중 합산 (지금은 "겹치는 기술 개수"만 씀 — 기준점용 임시 버전)
    STEP 4  동점 처리·정규화 (지금은 아예 없음 — 동점이 대량 발생할 것)
    STEP 5  설계 결정 4개 플래그의 실제 분기 로직 (지금은 값만 받고 안 씀)

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
              flags: dict = None, as_of: str = "2026-06-20") -> list:
    """이력서 텍스트 -> [(job_id, score, evidence), ...]  점수 내림차순.

    지금 점수 계산은 "겹치는 표준 기술명 개수"다 — IDF 가중이 아니다.
    STEP 3에서 이 부분을 idf 합산으로 바꾼다. 지금은 배선(로딩·캐싱·필터·
    evidence 형식)이 맞는지 확인하는 용도의 임시 점수다.

    ⚠️ eval_proxy.py의 완료 기준에 "IDF 가중이 들어감(단순 개수 아님)"이
       명시돼 있다 — 이 버전은 아직 그 기준을 통과 못 한다. STEP 3 전까지는
       중간 산출물로만 쓸 것.
    """
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

    # TODO STEP 4: 동점 처리·정규화. 지금은 정규화가 없어 겹침 개수가 같은
    # 공고끼리 순서가 임의적이다 (HANDOFF 문서: 200건 중 88건에서 동점 11건+).
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
        raw_score = sum(idf.get(t, 0.0) for t in hit) # IDF 가중 합산 기본 버전
        scored.append((raw_score, row["job_id"], sorted(hit)))

    scored.sort(key=lambda x: -x[0])
    return [
        (job_id, sc, {"matched_techs": hit, "matched_count": len(hit)})
        for sc, job_id, hit in scored[:top_k]
    ]


if __name__ == "__main__":
    # 수동 스모크 테스트. 실제 채점은 eval_proxy.evaluate(recommend)로 한다.
    sample = "파이썬과 장고로 결제 API를 개발했고 AWS에 배포했습니다."
    for job_id, sc, ev in recommend(sample, top_k=5):
        print(f"  {job_id}  score={sc}  {ev}")