"""
eval_proxy.py — 정답 라벨 없이 추천기를 채점한다.

---------------------------------------------------------------------------
어떻게 라벨 없이 점수가 나오는가
---------------------------------------------------------------------------
공고 하나를 골라 **그 공고의 요구 스킬로 가짜 이력서를 만든다.** 그 이력서를
추천기에 넣으면, 원래 그 공고가 상위에 나와야 한다. 안 나오면 추천기가 고장난
것이다. 정답(어떤 공고가 이 사람에게 맞는가)을 사람이 붙이지 않아도 된다.

    공고 #1234 요구 스킬  {Python, Django, AWS, Redis, Kubernetes}
       -> 60%만 남김      {Python, Django, AWS}
       -> 이력서 텍스트    "Python · Django · AWS"
       -> recommend(...)  상위 10개 안에 #1234 가 있는가?  몇 번째인가?

---------------------------------------------------------------------------
⚠️ 이 숫자가 증명하는 것과 증명하지 못하는 것
---------------------------------------------------------------------------
가짜 이력서는 공고의 스킬을 그대로 베낀 것이라 **진짜 이력서보다 훨씬 쉽다.**
사람이 쓴 이력서에는 표기가 다르고, 경력 서술이 섞이고, 요구 스킬의 절반은
아예 없다. 그래서 이 프록시는

    ✅ "망가졌다"를 잡아낸다 — 튜닝 후 점수가 떨어지면 뭔가 깨진 것이다
    ❌ "좋은 추천이다"를 증명하지 못한다

02에서 Macro F1 0.8595 중 0.1975가 평가 설계에서 나온 거품이었던 것과 같은
자리다. 이 경고를 지우지 말 것. `evaluate()`도 결과에 함께 실어 돌려준다.

---------------------------------------------------------------------------
설계에서 신경 쓴 것
---------------------------------------------------------------------------
1. **표준명만 쓴다.** 가짜 이력서를 텍스트로 렌더한 뒤 다시 뽑았을 때 원래
   스킬이 되돌아와야, 프록시가 '매칭 성능'을 재지 '사전 커버리지'를 재지 않는다.
   실측: 원문 스킬을 그대로 이어붙이면 왕복 재현율이 **48.5%**밖에 안 된다
   (`엑셀`·`정보처리기사`·`JAVA` 같은 비사전 표기 때문). `job_pool`이 나눠준
   표준명(`techs`)만 쓰면 **99.4%**가 된다.

2. **중복 job_id 는 질의에서 뺀다.** `recommend()`가 `job_id`만 돌려주는데
   이 데이터셋에는 같은 `(source, job_id)`가 777종 중복돼 있다. 그대로 채점하면
   "다른 공고를 맞혔는데 정답 처리"가 된다. 질의 대상에서만 빼고 후보 풀에는
   남긴다(빼면 모집단이 달라져 다른 실험과 비교가 안 된다).

3. **모집단을 인자로 받는다.** Recall@k 는 후보가 몇 건인지에 직접 의존한다.
   `job_pool.load_pool()`이 돌려준 것을 그대로 넣어야 하한선과 비교가 된다.

4. **무작위 베이스라인을 함께 낸다.** 브리프의 0.03%가 그 자리다. 이게 없으면
   "92.5%"가 높은 숫자인지 알 수 없다.

---------------------------------------------------------------------------
사용법
---------------------------------------------------------------------------
    from eval_proxy import evaluate
    evaluate(recommend)                 # -> {'recall@10': .., 'mrr': .., ...}

    python eval_proxy.py --self-test    # 내장 IDF 베이스라인으로 하한선 재현
"""
import argparse
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))

from tech_normalize import extract_techs          # noqa: E402
import job_pool                                    # noqa: E402

DEFAULT_N = 400          # 질의 수. 400이면 Recall 표준오차가 ±2%p 수준이다
DEFAULT_K = 10
DEFAULT_RATIOS = (1.0, 0.6, 0.4)
HEADLINE_RATIO = 0.4     # 브리프가 하한선으로 적어둔 조건(가장 어려운 것)
MIN_TECHS = 3            # 이보다 스킬이 적으면 60%/40% 조건이 의미가 없다


# ---------------------------------------------------------------------------
def render_resume(techs) -> str:
    """스킬 집합 -> 이력서 텍스트.

    가장 단순한 렌더링이다. 일부러 그렇게 뒀다 — 문장을 그럴듯하게 지어내면
    그 문장의 어휘가 점수에 섞여, 무엇을 재는 실험인지 흐려진다.
    """
    return ' · '.join(sorted(techs))


def make_queries(pool, n=DEFAULT_N, keep_ratio=1.0, seed=0, min_techs=MIN_TECHS):
    """[(job_id, resume_text, 남긴 스킬)] — 재현 가능하도록 시드를 고정한다."""
    dup = pool['job_id'].duplicated(keep=False)
    ok = pool[(pool['techs'].map(len) >= min_techs) & (~dup)]
    if ok.empty:
        raise ValueError("질의로 쓸 공고가 없다 (techs >= %d 인 행이 없음)" % min_techs)

    rng = random.Random(seed)
    idx = list(ok.index)
    rng.shuffle(idx)
    out = []
    for i in idx[:n]:
        techs = sorted(pool.at[i, 'techs'])
        k = max(1, round(len(techs) * keep_ratio))
        kept = set(random.Random(seed + i).sample(techs, k))
        out.append((pool.at[i, 'job_id'], render_resume(kept), kept))
    return out


def _rank_of(target_id, results, k):
    """추천 결과에서 정답의 순위(1부터). 없으면 None."""
    for rank, row in enumerate(results[:k], 1):
        jid = row[0] if isinstance(row, (tuple, list)) else row
        if str(jid) == str(target_id):
            return rank
    return None


def score(recommend, queries, k=DEFAULT_K):
    """{'recall@k', 'mrr', 'n'} — 정답이 1건뿐이라 Recall@k = 적중률이다."""
    hit = 0
    rr = 0.0
    for target_id, resume_text, _ in queries:
        results = recommend(resume_text, top_k=k) or []
        r = _rank_of(target_id, results, k)
        if r:
            hit += 1
            rr += 1.0 / r
    n = len(queries)
    return {f'recall@{k}': hit / n, 'mrr': rr / n, 'n': n}


def random_baseline(pool, queries, k=DEFAULT_K, seed=0):
    """무작위 추천. 브리프의 0.03% 자리 — 없으면 92.5%가 높은지 알 수 없다."""
    ids = pool['job_id'].tolist()
    rng = random.Random(seed)
    hit = rr = 0
    for target_id, _, _ in queries:
        pick = rng.sample(ids, min(k, len(ids)))
        r = _rank_of(target_id, [(x,) for x in pick], k)
        if r:
            hit += 1
            rr += 1.0 / r
    n = len(queries)
    return {f'recall@{k}': hit / n, 'mrr': rr / n, 'n': n}


def evaluate(recommend, pool=None, n=DEFAULT_N, k=DEFAULT_K,
             ratios=DEFAULT_RATIOS, seed=0, as_of='2026-06-20', verbose=True):
    """추천 함수를 채점한다.

    recommend  (resume_text, top_k=..) -> [(job_id, score, evidence), ...]
    pool       job_pool.load_pool() 결과. None 이면 기본 조건으로 로드한다
    as_of      pool 을 직접 만들 때의 기준일. 오늘로 두면 open 이 2건뿐이라
               필터를 켠 추천기는 아무것도 못 돌려준다(데이터가 2026.06~07 수집)

    반환: 헤드라인(가장 어려운 조건)과 조건별 표를 함께 돌려준다.
    """
    if pool is None:
        pool = job_pool.load_pool(as_of=as_of)

    out = {'pool_size': len(pool), 'by_ratio': {}}
    for ratio in ratios:
        q = make_queries(pool, n=n, keep_ratio=ratio, seed=seed)
        out['by_ratio'][ratio] = score(recommend, q, k)
    q = make_queries(pool, n=n, keep_ratio=HEADLINE_RATIO, seed=seed)
    out['random'] = random_baseline(pool, q, k, seed)
    out.update(out['by_ratio'].get(HEADLINE_RATIO, out['by_ratio'][ratios[-1]]))
    out['caveat'] = ("가짜 이력서는 공고 스킬을 그대로 베낀 것이라 진짜 이력서보다 "
                     "쉽다. 이 숫자는 '망가졌다'를 잡아낼 뿐 '좋다'를 증명하지 못한다.")
    if verbose:
        _print(out, k)
    return out


def _print(res, k):
    print(f"\n  후보 풀 {res['pool_size']:,}건 · 질의 {res['n']}건\n")
    print(f"    {'가짜 이력서 조건':22}{'Recall@%d' % k:>11}{'MRR':>9}")
    print('    ' + '-' * 42)
    for ratio, s in res['by_ratio'].items():
        print(f"    요구 스킬 {ratio:.0%} 보유{'':6}{s[f'recall@{k}']:>10.1%}{s['mrr']:>9.3f}")
    r = res['random']
    print(f"    {'(무작위)':22}{r[f'recall@{k}']:>10.2%}{r['mrr']:>9.3f}")
    print(f"\n  ⚠️ {res['caveat']}\n")


# ---------------------------------------------------------------------------
# 내장 베이스라인 — 프록시 자체를 검증하고, 하한선을 재현 가능하게 만든다
# ---------------------------------------------------------------------------
def idf_baseline(pool, idf, include_extra=False, normalize=False):
    """IDF 가중 스킬 매칭. 브리프가 '하한선'으로 적어둔 바로 그 방식.

    이걸 여기 둔 이유는 심지우의 `skill_matcher.py`를 대신하려는 게 아니라,
    **프록시가 제대로 동작하는지 확인할 대조군**이 필요해서다. 브리프에 적힌
    Recall@10 92.5% / MRR 0.790 을 재현하는 스크립트가 저장소에 없었다.
    (`reference_stats.json`이 생성 스크립트 없이 놓여 있던 것과 같은 문제다)
    """
    if include_extra:
        jobs = list(zip(pool['job_id'],
                        [a | b for a, b in zip(pool['techs'], pool['skills_extra'])]))
    else:
        jobs = list(zip(pool['job_id'], pool['techs']))

    # 길이 정규화 `/√Σidf` — 공고가 요구하는 스킬 전체의 IDF 합으로 나눈다.
    # 스킬을 많이 나열한 공고가 우연히 겹칠 확률이 높다는 편향을 눌러준다.
    # 여기 두는 이유: HANDOFF가 이 방식의 수치(Recall@10 94.0%)를 **하한선으로
    # 제시**하는데 그걸 재현하는 코드가 저장소에 없었다. 이 저장소가 같은 일로
    # 두 번 데였다 — `reference_stats.json`(생성 스크립트 없음)과 브리프의 92.5%.
    norm = {}
    if normalize:
        for jid, techs in jobs:
            s = math.sqrt(sum(idf.get(t, 0.0) for t in sorted(techs)))
            norm[jid] = s if s > 0 else 1.0

    def recommend(resume_text, top_k=10, filters=None):
        mine = extract_techs(resume_text)
        if include_extra:
            low = resume_text.lower()
            mine = mine | {x for x in _EXTRA_VOCAB if x in low}
        if not mine:
            return []
        scored = []
        for jid, techs in jobs:
            hit = mine & techs
            if hit:
                # ⚠️ `for t in hit`로 더하면 **실행마다 점수가 달라진다.**
                # `hit`는 집합이고 파이썬 문자열 해시는 프로세스마다 무작위화되므로
                # (PEP 456) 순회 순서가 매번 바뀐다. 부동소수점 덧셈은 결합법칙을
                # 만족하지 않아서, 같아야 할 두 점수가 마지막 비트에서 갈린다.
                #
                # 보통은 무시할 오차지만 여기서는 순위가 뒤집힌다 — 이 데이터는
                # **동점이 지배 요인**이기 때문이다(질의당 동점 평균 94건,
                # HANDOFF 2번 절). 그 결과 self-test가 재현되지 않았다:
                #     같은 코드 3회 실행 -> Recall@10  65.0% / 65.5% / 66.0%
                # 하한선으로 쓰라고 배포한 숫자가 ±1%p 흔들리면 "점수가 내려가면
                # 뭔가 깨진 것"이라는 사용법이 성립하지 않는다.
                #
                # 정렬된 순서로 더하면 순서가 고정되고 결과가 재현된다.
                h = sorted(hit)
                sc = sum(idf.get(t, 0.0) for t in h)
                if normalize:
                    sc /= norm[jid]
                scored.append((sc, jid, h))
        # 안정 정렬이므로 동점은 pool 행 순서를 따른다. 동점 처리 정책 자체는
        # 심지우가 고를 설계 결정이고(HANDOFF 2번), 여기서는 대조군이 흔들리지
        # 않게 고정하는 것까지만 한다.
        scored.sort(key=lambda x: -x[0])
        return [(jid, sc, {'matched_techs': h, 'matched_count': len(h)})
                for sc, jid, h in scored[:top_k]]

    return recommend


_EXTRA_VOCAB = set()


def self_test(as_of='2026-06-20', n=DEFAULT_N, k=DEFAULT_K, seed=0,
              include_extra=False, normalize=False):
    global _EXTRA_VOCAB
    corpus = job_pool.load_corpus()
    pool = job_pool.load_pool(as_of=as_of, corpus=corpus)
    idf = job_pool.build_idf(corpus, include_extra=include_extra)
    if include_extra:
        _EXTRA_VOCAB = {x for s_ in pool['skills_extra'] for x in s_ if len(x) >= 2}
    print(f"  IDF 사전 {len(idf):,}종 · 기준일 {as_of}"
          f"{' · 비사전 스킬 포함' if include_extra else ''}"
          f"{' · 길이 정규화 /√Σidf' if normalize else ''}")
    return evaluate(idf_baseline(pool, idf, include_extra, normalize), pool=pool,
                    n=n, k=k, seed=seed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true',
                    help='내장 IDF 베이스라인으로 하한선을 재현한다')
    ap.add_argument('--as-of', default='2026-06-20')
    ap.add_argument('--n', type=int, default=DEFAULT_N)
    ap.add_argument('--k', type=int, default=DEFAULT_K)
    ap.add_argument('--include-extra', action='store_true',
                    help='비사전 스킬(엑셀·정보처리기사…)까지 매칭에 쓴다')
    ap.add_argument('--norm', action='store_true',
                    help='길이 정규화 /√Σidf 를 건다 (HANDOFF 표의 아랫줄)')
    a = ap.parse_args()
    self_test(as_of=a.as_of, n=a.n, k=a.k, include_extra=a.include_extra,
              normalize=a.norm)
