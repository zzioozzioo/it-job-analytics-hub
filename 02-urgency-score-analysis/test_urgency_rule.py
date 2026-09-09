"""urgency_rule.py 회귀 테스트 — 입력 문자열로 고정한다.

`urgency_rule.py`는 라벨 생성(2-1)과 앱(2-2)이 공유하는 단일 출처인데
지금까지 테스트가 없었다. README가 적어둔 함정들은 전부 **조용히 재발하는**
종류다 — 점수가 틀려도 예외가 나지 않으므로 눈치채지 못한다.

여기 있는 케이스는 전부 실제로 한 번씩 틀렸던 것이다. 새로 만든 게 아니라
모듈 주석과 README에 근거가 적혀 있는 것을 문자열로 옮겼을 뿐이다.

pytest 없이 그냥 실행된다:

    python test_urgency_rule.py

(환경에 pytest가 들어오면 test_* 함수를 그대로 수집한다)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from urgency_rule import (FALLBACK_BASE, FALLBACK_SCALE,  # noqa: E402
                          RX_EARLY_CLOSE, RX_ROLLING, apply_fallback_scale,
                          clean_body, extract_signals, parse_application_window,
                          score_by_vocabulary, score_posting,
                          vocabulary_evidence)

# 접수 창 30일 = +4점. 아래 케이스들의 공통 배경값이다.
WIN30_A = "접수기간 · 방법 시작일 2026.06.09(화) 마감일 2026.07.09(목)"
WIN30_B = "접수기간 2026. 6. 9(화) ~ 2026. 7. 9(목)"


def _ev(text):
    return extract_signals(text)[0]


# ---------------------------------------------------------------------------
# [수정 3] '채용 시 마감'은 한 번만 센다  (v4)
# ---------------------------------------------------------------------------
# 이 문구가 RX_ROLLING(+10)과 RX_EARLY_CLOSE(+12) 양쪽 정규식에 들어 있어
# 한 문구로 22점이 됐다. measurable 30,569행 중 2,838행(9.3%)에서 발동했고
# 1,993행(6.5%)의 등급을 부풀렸다.
def test_rolling_and_early_close_counted_once():
    assert RX_ROLLING.search("채용 시 마감")
    assert RX_EARLY_CLOSE.search("채용 시 마감"), "두 정규식이 같은 문구를 가진다는 전제"

    # 형태 A(jobkorea) · 형태 B(saramin) 모두 4 + 12 = 16. 26이면 이중 계상이다.
    assert _ev(f"{WIN30_A} 채용 시 마감 접수방법 즉시지원") == 16
    assert _ev(f"{WIN30_B} 채용 시 마감 접수방법 온라인") == 16


def test_early_close_alone_still_scores():
    """rolling이 없는 '조기 마감'은 그대로 +12. 수정 3이 E를 약화시키면 안 된다."""
    assert _ev(f"{WIN30_A} 접수방법 즉시지원 조기 마감될 수 있습니다") == 16


def test_rolling_alone_still_scores():
    """`채용시까지`는 RX_EARLY_CLOSE에 없다. 겹치지 않으므로 rolling이 +10을 낸다.

    수정 3을 '무조건 rolling을 빼기'로 구현하면 이 경로가 0점이 되어 조용히
    죽는다. 그래서 early_close가 실제로 발동했을 때만 양보한다."""
    assert _ev(f"{WIN30_A} 채용시까지 접수방법") == 14


def test_reason_keeps_rolling_wording():
    """근거 문장은 rolling 쪽 표현을 유지한다 — 앱이 사용자에게 보여주는 값이다."""
    reasons = extract_signals(f"{WIN30_A} 채용 시 마감 접수방법")[1]
    assert "채용 시 마감(충원되면 조기 종료)" in reasons


def test_always_open_branch_survives():
    """상시채용(+10)은 rolling과 배타 관계였다. 수정 3이 그 elif를 깨면 안 된다."""
    assert _ev("모집인원 1 명 마감일 상시채용 접수방법 온라인") == 10


# ---------------------------------------------------------------------------
# [수정 1] 지원자 수를 모집인원으로 오인하던 버그  (v3)
# ---------------------------------------------------------------------------
# jobkorea는 모집인원을 `○○`로 가려서 정식 패턴이 비고, 바로 옆 지원자 수가
# 대신 걸렸다. 의미가 정반대다 — 지원자가 많으면 그 공고는 덜 급하다.
def test_applicant_count_not_read_as_headcount():
    raw = ("접수기간 · 방법 시작일 2026.06.09(화) 마감일 2026.07.09(목) "
           "지원자 현황 통계 지원자 수 8 명 모집인원 ○○ 명")
    body = clean_body(raw, 'jobkorea')
    assert "지원자 수 8 명" not in body, "지원자 통계 블록이 제거돼야 한다"
    assert _ev(body) == 4, "모집인원 8명(+20)이 붙으면 안 된다"


def test_real_headcount_survives_the_removal():
    """제거 범위가 넓어져 진짜 모집인원까지 먹으면 안 된다 (끝 경계는 지원자 수까지)."""
    raw = ("접수기간 · 방법 시작일 2026.06.09(화) 마감일 2026.07.09(목) "
           "지원자 현황 통계 지원자 수 8 명 모집인원 12 명")
    assert _ev(clean_body(raw, 'jobkorea')) == 4 + 28   # 12명 -> 대규모(+28)


# ---------------------------------------------------------------------------
# [수정 2] 접수 창 길이 — 소스 통합, 조회 시점에 의존하지 않는다  (v3)
# ---------------------------------------------------------------------------
def test_window_is_same_quantity_across_sources():
    """형태 A와 형태 B가 같은 값을 돌려준다. 다르면 소스별로 다른 타깃이 된다."""
    assert parse_application_window(WIN30_A)[0] == 30
    assert parse_application_window(WIN30_B)[0] == 30


def test_days_remaining_is_not_scored():
    """`남은기간`은 공고가 아니라 조회 시점의 속성이라 라벨에서 뺐다."""
    assert _ev("접수기간 · 방법 남은기간 3일 접수방법 즉시지원") == 0


# ---------------------------------------------------------------------------
# rolling 앵커 — 앱에서만 생기던 train/serve skew
# ---------------------------------------------------------------------------
# 2-2 앱이 조립하는 헤더는 학습 데이터보다 짧아서, `접수기간` 뒤 90자 창이
# 본문 앞부분까지 먹는다. 본문 첫머리에 '채용 시 마감'이 있으면 접수 조건이
# 아닌데도 rolling이 켜졌다. 형태 A는 매치 구간 + 40자로 좁혀 막는다.
def test_rolling_anchor_does_not_reach_into_body():
    text = (f"{WIN30_A} 모집인원 1 명 "
            "우리 회사는 성장 중입니다 " * 3 + "채용 시 마감이라는 말이 본문에 있음")
    assert parse_application_window(text)[1] is False, "앵커 밖 문구로 rolling이 켜지면 안 된다"


# ---------------------------------------------------------------------------
# [수정 4] 어휘 폴백 재보정  (v4)
# ---------------------------------------------------------------------------
# v3까지 `min(3 + 가산, 5)`였고 상수보다 MAE가 나빴다. 보폭(SCALE)은
# validation이, 기준점(BASE)은 train 중앙값이 정한다 — calibrate_fallback.py.
def test_fallback_no_vocabulary_is_base():
    """어휘가 하나도 없으면 기준점 그대로. v3에서는 3점이었다."""
    assert FALLBACK_BASE == 2 and FALLBACK_SCALE == 0.25
    assert score_by_vocabulary("백엔드 개발자를 찾습니다. Python 경험자 우대.")[0] == 2


def test_fallback_never_asserts_top_grade_on_buzzwords():
    """v3 폴백은 버즈워드만으로 unmeasurable 877행에 5점(매우 높음)을 붙였다.

    TIER_A 5개가 전부 걸려도(가산 10) 2 + 10x0.25 = 4.5 -> 5점이 되지만,
    실제 데이터에서 관측된 최대 가산은 6점이다. 아래는 그 관측 최대치로
    상위 등급이 나오지 않는지 고정한다."""
    add, _ = vocabulary_evidence(
        "급성장하는 조직입니다. 애자일하게 일하며 빠른 성장을 지향합니다. "
        "신규 팀의 초기 멤버를 찾습니다.")
    assert add > 0, "버즈워드가 실제로 가산점을 만든다는 전제"
    assert apply_fallback_scale(add) <= 3, "버즈워드만으로 4·5점이 나오면 안 된다"


def test_fallback_still_orders_by_evidence():
    """재보정이 순서 정보까지 뭉개면 안 된다 — 가산이 크면 등급도 커야 한다."""
    assert apply_fallback_scale(0) <= apply_fallback_scale(4) <= apply_fallback_scale(8)
    assert apply_fallback_scale(0) < apply_fallback_scale(8)


def test_fallback_rounds_half_up_and_clips():
    """0.5는 항상 위로. 파이썬 내장 round()는 은행가 반올림이라 쓰지 않는다."""
    assert apply_fallback_scale(2, base=1, scale=0.25) == 2       # 1.5 -> 2
    assert apply_fallback_scale(6, base=2, scale=0.25) == 4       # 3.5 -> 4
    assert apply_fallback_scale(100, base=2, scale=0.25) == 5     # 상한
    assert apply_fallback_scale(0, base=0, scale=0.25) == 1       # 하한


def test_unmeasurable_reason_does_not_claim_neutral():
    """기준점이 3점이 아니게 됐으므로 '(중립)'이라고 쓰면 거짓말이 된다."""
    reason = score_posting("백엔드 개발자를 찾습니다.", 'wanted')['urgency_reason']
    assert "중립" not in reason
    assert "기준점" in reason


def test_measurable_and_unmeasurable_paths():
    """메타데이터가 없으면 어휘 폴백(미검증 경로)으로 빠진다."""
    assert score_posting(f"{WIN30_A} 접수방법", 'jobkorea')['measurable'] is True
    r = score_posting("백엔드 개발자를 찾습니다. Python 경험자 우대.", 'wanted')
    assert r['measurable'] is False
    assert r['evidence_score'] is None


# ---------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}  — {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
