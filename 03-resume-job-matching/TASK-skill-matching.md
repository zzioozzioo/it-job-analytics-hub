# [작업 요청] 03번 — 스킬 매칭 베이스라인

> ⚠️ **이 문서만 읽고 착수하지 마세요. 먼저 [`HANDOFF-skill-matching.md`](HANDOFF-skill-matching.md)를 보세요.**
> 이 브리프의 설계 방향은 유효하지만 **수치 일부는 폐기됐습니다.** 특히 아래
> '스스로 채점하는 법' 절의 **Recall@10 92.5% / MRR 0.790 표는 재현되지 않습니다 —
> 목표로 쓰지 마세요.** 현재 기준선과 그 재현 명령은 HANDOFF에 있습니다.

**담당**: 심지우
**요청자**: 김민석
**예상 분량**: 파일 1개(`skill_matcher.py`) + 짧은 리포트

---

## 왜 이 작업이 심지우 담당인가

01번에서 만든 게 그대로 재료가 됩니다. 새로 배울 게 거의 없습니다.

| 01번 자산 | 03번에서의 쓰임 |
|---|---|
| `MASTER_TECH_DICT` (`임시/test2.ipynb`) | **이력서의 한글 기술명을 공고와 같은 표기로 맞춤** — 이게 없으면 매칭 자체가 안 됨 |
| `build_cooccurrence_index` (`01/app.py:483`) | 반환값 `(공고별 기술 집합, 빈도)`가 IDF 가중치 재료 그 자체 |
| `EXCLUDE_TECH` (`01/app.py:365`) | `'시스템'`·`'데이터'`처럼 아무 데나 붙는 단어 제거 |
| `compute_related_techs` (`01/app.py:503`) | (선택) 이력서 스킬 확장 — `Django`만 써도 `Python` 추론 |

**사전은 이미 모듈로 빼뒀습니다**: `common/tech_normalize.py` (아래 참조)

---

## 만들 것

`03-resume-job-matching/skill_matcher.py`

```python
def recommend(resume_text: str, top_k: int = 10, filters: dict = None) -> list:
    """이력서 텍스트 -> 추천 공고 리스트.

    returns: [(job_id, score, evidence), ...]  점수 내림차순
             evidence = {'matched_techs': [...], 'matched_count': int, ...}
    """
```

**반환 형식은 꼭 지켜주세요.** 나중에 임베딩 결과와 RRF로 합칠 때 같은 모양이어야 합니다.

---

## 왜 단순 "겹치는 개수"로 세면 안 되나

| 기술 | 이 기술을 요구하는 공고 | IDF |
|---|---|---|
| Python | 4,747건 (25.3%) | 1.38 |
| AWS | 3,375건 (18.0%) | 1.72 |
| Kubernetes | 1,748건 (9.3%) | **2.37** |
| Rust | 573건 (3.0%) | **3.49** |

Python은 4명 중 1명이 갖고 있고 Rust는 33명 중 1명입니다. 개수로만 세면 **Python·Git·SQL만 겹치는 공고가 상위**로 올라옵니다. 희귀한 일치에 가중치를 줘야 합니다.

IDF는 `build_cooccurrence_index`가 돌려주는 값에서 **바로 나옵니다**:

```python
jobs, counter = build_cooccurrence_index(...)
N = len(jobs)
idf = {t: math.log(N / counter[t]) for t in counter}
```

`counter.update(techs)`에서 `techs`가 `set`이라 **`counter[t]`가 이미 "그 기술을 요구하는 공고 수"**(문서 빈도)입니다. 추가 계산이 필요 없습니다.

---

## 쓸 재료

### 1) 정규화 모듈 — `common/tech_normalize.py`

`임시/test2.ipynb`의 `MASTER_TECH_DICT`를 꺼내 모듈로 만들어 뒀습니다.

```python
import sys; sys.path.insert(0, '../common')
from tech_normalize import extract_techs, normalize_list, EXCLUDE_TECH

extract_techs("스프링부트와 JPA로 커머스 API를 만들었고 도커로 배포")
# -> {'Spring Boot', 'JPA', 'Docker'}
```

- 별칭 298개 → 표준명 158종, **한글 별칭 91개**
- 데이터셋에 실제로 등장하는 156종을 **100% 커버** (확인함)
- 공고 `raw_text`에서 재추출 시 기존 `final_techs` **83% 재현** + 건당 2.1개 추가 발견

> **원본에서 고친 것 하나**: 사전이 `springboot`(공백 없음)만 별칭으로 갖고 있어서
> `"Spring Boot"`가 짧은 `spring`에 먼저 걸려 `Spring`으로 뭉개졌습니다. 데이터셋에
> `Spring`(2,505건)과 `Spring Boot`(126건)가 둘 다 있어 실제 손실이라, 표준명의
> 공백형을 별칭으로 되먹였습니다(19종). 자세한 건 모듈 주석에 있습니다.

### 2) 공고 스킬 필드 — 어느 걸 쓸까

| 필드 | 채움 비율 | 비고 |
|---|---|---|
| `final_techs` | 46.6% | 01에서 쓰던 것. 깔끔하지만 절반 이하 |
| `hard_skills` | 77.0% | 사람인이 88%로 특히 잘 채워져 있음 |
| **둘 다 합집합** | **79.1%** | **이걸 권합니다** |

`build_cooccurrence_index`는 `final_techs`만 봅니다. 이 한 줄을 바꿔야 합니다.

```python
for cell in df['final_techs']:                 # 현재 (46.6%)
# ↓
for cell in df['final_techs'] + df['hard_skills']:   # 03에서 (79.1%)
```

---

## 하드 필터 (유사도가 아니라 조건)

신입에게 "경력 5년 이상"을 추천하면 안 됩니다. **점수를 깎는 게 아니라 후보에서 빼야** 합니다.

```python
filters = {'experience': '신입', 'location': '서울'}
```

⚠️ **지금은 이 필드가 35.7%만 채워져 있습니다.** 사람인 25,950건에 `title`·`company`·`location`·`experience_level`이 전부 비어 있어서요.

김민석이 백필 작업(`backfill_saramin.py`) 중이고 **90%대로 올라갈 예정**입니다. 30건 샘플에서 97% 복원을 확인했습니다.

**그때까지 기다리지 마시고**, 필터는 "값이 있으면 적용, 없으면 통과" 로 짜두시면 백필 완료 후 자동으로 켜집니다.

---

## 스스로 채점하는 법

김민석이 **평가 프록시**(`eval_proxy.py`)를 만들어 전달할 예정입니다.

```python
from eval_proxy import evaluate
evaluate(recommend)   # -> {'recall@10': 0.925, 'mrr': 0.790}
```

공고의 요구 스킬을 "가짜 이력서"로 써서 그 공고를 상위에 회수하는지 재는 방식입니다. **정답 라벨 없이도 숫자가 나옵니다.**

참고로 아주 단순한 IDF 가중 매칭만으로도 이 정도가 나옵니다:

| 가짜 이력서 조건 | Recall@10 | MRR |
|---|---|---|
| 요구 스킬 100% 보유 | 100.0% | 0.977 |
| 요구 스킬 60% 보유 | 100.0% | 0.954 |
| 요구 스킬 40% 보유 | 92.5% | 0.790 |
| (무작위) | 0.03% | — |

**이 숫자를 넘기는 게 목표가 아닙니다.** 이건 하한선입니다. 튜닝할 때 점수가 내려가지 않는지 확인하는 용도로 쓰세요.

> ⚠️ 프록시가 높다고 좋은 추천이라는 뜻은 아닙니다. 가짜 이력서는 공고 스킬을 그대로
> 쓴 거라 진짜 이력서보다 훨씬 쉽습니다. **"망가졌다"를 잡아낼 뿐 "좋다"를 증명하지
> 못합니다.** 02번에서 F1 0.8595가 실제로는 0.66이었던 것과 같은 이야기입니다.

---

## 판단이 필요한 것 (정해서 알려주세요)

1. **상위 개념 함의** — 이력서에 `Spring Boot`가 있으면 `Spring`을 요구하는 공고도 맞다고 볼 것인가? 사전으로 자동 처리하지 않고 남겨뒀습니다. 매칭 설계 결정이라서요.
2. **스킬 0개 이력서** — 기술명이 하나도 안 잡히는 이력서는 어떻게 할지 (빈 결과 vs 텍스트 유사도 폴백)
3. **`compute_related_techs` 활용 여부** — `Django`만 있는 이력서에 `Python`을 암묵 추가할지. 재현율은 오르지만 정밀도가 떨어질 수 있습니다.

---

## 완료 기준

- [ ] `recommend()` 가 위 시그니처대로 동작
- [ ] IDF 가중이 들어감 (단순 개수 아님)
- [ ] `evidence`에 **어떤 기술이 일치했는지** 담김 — 화면에 "Python·Django·AWS 3개 일치"로 보여줄 것
- [ ] 필터가 "값 없으면 통과"로 동작 (백필 전에도 깨지지 않게)
- [ ] 평가 프록시 점수를 리포트에 기록

---

## 안 해도 되는 것

- **벡터 DB** — 4만 건이면 numpy 브루트포스가 2.8ms입니다. 인덱스가 필요 없습니다.
- **임베딩** — 김민석이 4단계에서 붙입니다. 지금은 스킬 매칭만.
- **UI** — 매칭이 검증된 다음에.

---

## 참고 파일

```
common/tech_normalize.py                     정규화 모듈 (이번에 새로 만듦)
01-tech-stack-wordcloud/app.py:365           EXCLUDE_TECH
01-tech-stack-wordcloud/app.py:483           build_cooccurrence_index
01-tech-stack-wordcloud/app.py:503           compute_related_techs
임시/test2.ipynb                             MASTER_TECH_DICT 원본
03-resume-job-matching/backfill_saramin.py   메타데이터 백필 (김민석)
```
