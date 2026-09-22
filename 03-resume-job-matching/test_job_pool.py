"""job_pool.py 회귀 테스트 — 네트워크도 데이터 파일도 필요 없다.

이 파일이 생긴 이유는 실제로 당한 사고 하나다.

`job_pool.py`의 docstring은 status를 네 가지로 약속하고 있었다.

    open     마감일이 as_of 이후
    closed   마감일이 지남
    rolling  상시·수시채용 (마감일 개념이 없음)      <- 이 줄
    unknown  마감일을 못 읽음

그런데 `parse_deadline()`은 `RX_ROLLING`(= `채용 시 마감`류)만 보고
**상시·수시채용은 아예 보지 않았다.** 문서가 약속한 것의 한쪽을 코드가
구현하지 않은 상태였고, 예외가 나지 않으니 아무도 몰랐다. 결과는
`only_open=True`에서 진짜 상시채용 공고 1,264건이 조용히 사라지는 것이었다
(jobkorea rolling 18건 -> 1,281건).

02가 `reference_stats.json`에서 당한 것과 같은 종류다 — **틀려도 예외가
나지 않는 결함**은 테스트로 고정하지 않으면 재발한다.

pytest 없이 그냥 실행된다:

    python test_job_pool.py
"""
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_pool import parse_deadline, status_of  # noqa: E402

AS_OF = datetime.date(2026, 6, 20)


def _status(text):
    deadline, rolling = parse_deadline(text)
    return status_of(deadline, rolling, AS_OF)


# ---------------------------------------------------------------------------
# 이 파일이 생긴 계기 — 상시채용이 unknown으로 떨어지던 버그
# ---------------------------------------------------------------------------
def test_always_open_is_rolling_not_unknown():
    """docstring이 'rolling = 상시·수시채용'이라고 약속한 것을 실제로 지킨다.

    이 케이스가 바로 버그 신고자가 넣었던 합성 공고다. 전에는 unknown이 나왔다."""
    assert _status("접수기간 상시채용 접수방법 온라인 입사지원") == 'rolling'


def test_jobkorea_deadline_field_is_rolling():
    """jobkorea의 실제 형태 — `마감일`이 날짜가 아니라 `상시채용`이다.

    이 형태가 jobkorea에서 1,263건이고, 전부 unknown으로 떨어지고 있었다."""
    assert _status("접수기간 · 방법 시작일 2026.06.08(월) 마감일 상시채용 "
                   "접수방법 잡코리아 즉시지원") == 'rolling'


def test_rolling_phrase_still_works():
    """원래 보던 `채용 시 마감`이 죽으면 안 된다 — 갈래를 더한 것이지 바꾼 게 아니다."""
    assert _status("접수기간 채용 시 마감 접수방법 온라인") == 'rolling'


def test_recommendation_tail_does_not_fake_rolling():
    """꼬리의 `상시채용`은 남의 공고 것이다. 이걸 주우면 마감된 공고가 살아난다.

    앵커 없이 고쳤다면 jobkorea에서 6,839건이 가짜 rolling이 됐을 자리다."""
    text = ("㈜우리회사 백엔드 개발자 모집 상세요강 접수기간∙방법 기업정보 추천공고 "
            + "본문입니다 " * 40 +
            "AI추천공고를 확인해 보세요! ㈜다른회사 프론트엔드 채용 상시채용 즉시 지원")
    assert _status(text) == 'unknown'


def test_privacy_boilerplate_does_not_fake_rolling():
    """saramin 약관 문구의 '상시 채용'은 개인정보 이용 목적이지 채용 방식이 아니다."""
    text = ("모집분야 백엔드 " + "본문 " * 40 +
            "* 본 입사지원서 상의 개인정보는 [개인정보 보호법] 제15조에 의거하여 "
            "당사 상시 채용을 위한 용도로만 사용됩니다")
    assert _status(text) == 'unknown'


# ---------------------------------------------------------------------------
# status 네 가지가 실제로 전부 도달 가능한가 — docstring과의 계약
# ---------------------------------------------------------------------------
# 위 버그의 정체는 "문서에 적힌 상태 중 하나가 사실상 죽어 있었다"는 것이다.
# 네 상태가 모두 살아 있는지 한 번에 확인한다.
def test_all_four_statuses_are_reachable():
    cases = {
        'open':    "접수기간 · 방법 시작일 2026.06.01(월) 마감일 2026.07.09(목)",
        'closed':  "접수기간 · 방법 시작일 2026.04.01(수) 마감일 2026.05.09(토)",
        'rolling': "접수기간 상시채용 접수방법 온라인",
        'unknown': "회사 소개와 복지만 적혀 있고 접수 조건이 없는 공고",
    }
    got = {name: _status(text) for name, text in cases.items()}
    assert got == {k: k for k in cases}, got


def test_unknown_is_not_treated_as_open():
    """`unknown`을 open에 섞지 않는다 — '열려 있다'는 뜻이 아니다."""
    assert _status("접수 조건이 없는 본문") == 'unknown'


def test_deadline_is_read_from_the_field():
    """마감일이 실제로 파싱되는지 (rolling 갈래가 날짜를 가리면 안 된다)."""
    deadline, _ = parse_deadline(
        "접수기간 · 방법 시작일 2026.06.01(월) 마감일 2026.07.09(목)")
    assert deadline == datetime.date(2026, 7, 9)


def test_rolling_does_not_erase_a_real_deadline():
    """마감일과 `채용 시 마감`이 함께 있으면 마감일이 이긴다.

    status_of는 deadline이 있으면 open/closed를 먼저 돌려준다. 날짜가 있는데
    rolling으로 뭉개면 `as_of` 계산이 무의미해진다."""
    deadline, rolling = parse_deadline(
        "접수기간 · 방법 시작일 2026.06.01(월) 마감일 2026.07.09(목) 채용 시 마감")
    assert deadline == datetime.date(2026, 7, 9)
    assert rolling is True
    assert status_of(deadline, rolling, AS_OF) == 'open'


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
