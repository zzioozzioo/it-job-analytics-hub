"""calibrate_fallback.py — 어휘 폴백을 재보정하고 그 근거를 남긴다.  [v4 수정 4]

---------------------------------------------------------------------------
무엇이 문제였나
---------------------------------------------------------------------------
메타데이터가 없는 공고(전체의 24.2%)는 규칙이 점수를 계산할 수 없어서
`score_by_vocabulary()`가 어휘만 보고 등급을 붙인다. v3까지 그 식은

    level = min(3 + 가산, 5)          가산 = TIER_A x2 + TIER_B x1 + TIER_C x1

이었고, 2-1 README '한계 1'과 02 README '남은 근본 문제'가 둘 다
**"상수(항상 3점)보다 MAE가 나쁘다"** 고 실측해 두었다. 즉 어휘 가산이
실질적으로 아무 일도 하지 않는다. 그럼에도 남겨둔 이유는 "대체할 검증된
방법이 없어서"였다.

---------------------------------------------------------------------------
이 스크립트가 하는 일
---------------------------------------------------------------------------
새 신호도 새 어휘도 추가하지 않는다. 가산점을 등급으로 옮기는 **숫자 두 개**
(BASE, SCALE)만 다시 고른다.

    level = clip(BASE + 가산 x SCALE, 1, 5)

v3의 진단이 그대로 처방이 된다. 2-1 README는 전이 모델을 두고 "순서(ranking)는
배웠지만 점 예측(calibration)이 나쁘다"고 적었는데, 폴백이 정확히 같은 병이다.
QWK가 0이 아니라는 것은 순서 신호가 있다는 뜻이고, MAE가 상수보다 나쁘다는
것은 그 순서를 **엉뚱한 눈금 위에** 올려놨다는 뜻이다. 기준점(3점)이 너무 높고
보폭(가산 1점 = 등급 1칸)이 너무 크다.

---------------------------------------------------------------------------
측정 설정 — 그리고 그 설정의 한계
---------------------------------------------------------------------------
폴백이 실제로 쓰이는 unmeasurable 9,779행에는 **정답이 없다.** 그 행의 라벨이
바로 폴백의 출력이라서, 자기 자신과 비교하게 된다.

그래서 2-1 README '한계 1'과 같은 프록시를 쓴다 — measurable 행을
"메타데이터가 없는 척" 두고, 규칙이 계산한 라벨을 정답으로 삼아 폴백이 그걸
얼마나 맞히는지 본다.

⚠️ 이 설정의 순환은 그대로 남는다. 정답이 규칙의 출력이므로 여기서 재는 것은
   "폴백이 규칙을 얼마나 잘 흉내내는가"이지 "폴백이 맞는가"가 아니다.
   그리고 measurable 행과 unmeasurable 행은 애초에 다른 모집단이다.
   이 실험은 **현행보다 나은가**만 답할 수 있다.

---------------------------------------------------------------------------
선택 규칙 — test를 보기 전에 정한다
---------------------------------------------------------------------------
    "validation QWK가 현행 폴백 이상인 후보 중에서, validation MAE가 가장 낮은 것"

순서 신호를 잃지 않으면서 점 예측을 최대한 고친다는 뜻이다. QWK만 보면
MAE가 폭주하고(현행이 그 상태다), MAE만 보면 SCALE=0(상수)으로 붕괴해서
폴백을 없애는 것과 같아진다.

test는 고른 뒤 **한 번만** 본다. v2 설계의 결함(모델을 test로 고름)을
반복하지 않기 위해서다.

---------------------------------------------------------------------------
[중요] 프록시가 고를 수 있는 것은 SCALE뿐이다 — BASE는 아니다
---------------------------------------------------------------------------
위 규칙을 그대로 돌리면 **BASE=1, SCALE=0.25**가 뽑힌다. 그런데 그 BASE를
그대로 쓰면 안 된다는 것을 실측이 보여준다.

unmeasurable 9,779행 중 **8,144행(83.3%)은 어휘 가산점이 0**이다. 폴백은
재보정 전에도 후에도 사실상 상수이고, 재보정은 그 더미를 어느 등급에
쌓을지를 정할 뿐이다.

    폴백이 실제로 적용되는 9,779행의 쏠림
      v3 (3 + 가산)          3점에 83.3%
      BASE=1 (프록시 최적)    1점에 91.0%
      BASE=2 (채택)          2점에 91.0%

**BASE는 "관측할 수 없는 공고를 어디에 둘 것인가"라는 사전 확률이고, 프록시는
정의상 그 질문에 답할 수 없다.** 프록시 모집단(measurable)과 적용 대상
(unmeasurable)은 애초에 다른 모집단이며, 나뉘는 기준이 바로 "메타데이터가
있느냐"다. 프록시가 BASE=1을 좋아하는 것은 measurable 라벨이 1~2점에 몰려
있기 때문이지, 관측 불가능한 공고가 실제로 덜 급해서가 아니다.

그리고 그 결과물은 2-1 README '한계 2'가 이미 기각한 것과 같은 모양이다 —
*"예측이 1점에 79.8% 쏠리는데, 이는 학습 분포를 그대로 투사한 label shift이지
그 공고들이 덜 급하다는 증거가 아니다."* 모델로 하면 안 된다고 적어놓고
폴백으로 같은 일을 하면 앞뒤가 맞지 않는다.

그래서 이렇게 나눈다.

    SCALE = 0.25   프록시(validation)가 고른다. "어휘 하나가 등급을 얼마나
                   움직여야 하는가"는 순서/보폭 문제라 프록시가 답할 수 있다.
    BASE  = 2      프록시로 고르지 않는다. measurable train 라벨의 **중앙값**을
                   쓴다 — v3이 상수 베이스라인을 "항상 3점"에서 "train 중앙값"
                   으로 바꾼 것과 같은 규약이다.

BASE=2가 실제로 고치는 것은 분명하다. v3 폴백은 `급성장`·`애자일` 같은 단어
몇 개만으로 **877행에 5점(매우 높음), 758행에 4점(높음)** 을 붙이고 있었다.
재보정 후에는 상위 등급이 사라진다. 그게 이 수정의 실질이고, 더미를 1점으로
옮기는 것은 아니다.

두 후보를 test에서 나란히 보고한다. 판단에 동의하지 않으면 숫자를 보고
반박할 수 있어야 하기 때문이다.

실행:
    python calibrate_fallback.py            선택 + test 1회 평가
    python calibrate_fallback.py --grid     전체 그리드도 함께 출력
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding='utf-8')

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

from urgency_rule import (apply_fallback_scale, is_measurable,  # noqa: E402
                          score_posting, vocabulary_evidence)
from train_urgency_baseline import (RANDOM_STATE, TEST_FOLDS,  # noqa: E402
                                    data_path, dedup_and_group, evaluate, rule)

OUT_JSON = HERE / "fallback_calibration.json"

# 현행(v3)과, 탐색할 격자.
CURRENT = (3, 1.0)
BASES = [1, 2, 3]
SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_measurable():
    """measurable 행 + v4 규칙 라벨 + 어휘 가산점."""
    with open(data_path(), encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for x in data:
        raw = x.get('raw_text') or ''
        if not is_measurable(raw):
            continue
        rows.append({
            'raw_text': raw,
            'source': x['source'],
            # 규칙이 메타데이터를 읽고 만든 라벨 = 이 실험의 정답
            'y': score_posting(raw, x['source'])['urgency_score'],
            # 폴백이 보는 유일한 재료. 메타데이터를 못 본다고 가정한 값이다.
            'add': vocabulary_evidence(raw)[0],
        })
    return pd.DataFrame(rows)


def make_split(df):
    """train_urgency.make_split과 같은 방식의 group-aware 3분할."""
    df, groups, _ = dedup_and_group(df)
    strat = df['source'] + "__" + df['y'].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=TEST_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    rest_pos, te_pos = next(sgkf.split(df, strat, groups))
    sgkf2 = StratifiedGroupKFold(n_splits=TEST_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)
    tr_pos, va_pos = next(sgkf2.split(df.iloc[rest_pos], strat.iloc[rest_pos],
                                      groups.iloc[rest_pos]))
    rest = df.iloc[rest_pos]
    tr = rest.iloc[tr_pos].reset_index(drop=True)
    va = rest.iloc[va_pos].reset_index(drop=True)
    te = df.iloc[te_pos].reset_index(drop=True)
    shared = set(groups.iloc[te_pos]) & set(groups.iloc[rest_pos])
    print(f"  train {len(tr):,} / val {len(va):,} / test {len(te):,}")
    print(f"  공유 그룹 {len(shared)}개 {'(정상)' if not shared else '(!! 누출)'}")
    return tr, va, te


def predict(add, base, scale):
    return np.array([apply_fallback_scale(a, base, scale) for a in add])


def _report_unmeasurable_shift(variants):
    """폴백이 실제로 적용되는 구간에서 등급이 어디에 쌓이는지.

    프록시(measurable) 점수만 보면 놓치는 것이 여기서 드러난다 —
    unmeasurable 행의 83.3%는 어휘 가산점이 0이라 폴백은 사실상 상수이고,
    파라미터는 '그 더미를 몇 점에 둘 것인가'를 정할 뿐이다."""
    with open(data_path(), encoding='utf-8') as f:
        data = json.load(f)
    adds = [vocabulary_evidence(x.get('raw_text'))[0]
            for x in data if not is_measurable(x.get('raw_text'))]
    zero = sum(1 for a in adds if a == 0)
    print(f"    unmeasurable {len(adds):,}행 중 어휘 가산 0점 = "
          f"{zero:,}행 ({zero / len(adds) * 100:.1f}%)")
    print(f"    {'':22}" + "".join(f"{i}점".rjust(9) for i in range(1, 6)) + "     최빈 쏠림")
    for b, s in variants:
        lv = predict(np.array(adds), b, s)
        cnt = [int((lv == i).sum()) for i in range(1, 6)]
        print(f"    base={b} scale={s:<5}      "
              + "".join(f"{c:>9,}" for c in cnt)
              + f"     {max(cnt) / len(lv) * 100:.1f}%")


def score(df, base, scale):
    """evaluate()는 0-index 라벨을 받는다(N_CLASSES=5 기준)."""
    return evaluate(df['y'].values - 1, predict(df['add'].values, base, scale) - 1)


def main(show_grid):
    df = load_measurable()
    rule("STEP 1. 데이터")
    print(f"  measurable {len(df):,}행 (폴백 프록시 모집단)")
    add = df['add'].values
    print(f"  어휘 가산점 분포: " +
          "  ".join(f"{v}점 {int((add == v).sum()):,}"
                    for v in sorted(set(add.tolist()))[:8]))
    print(f"  가산 0점(어휘 없음): {int((add == 0).sum()):,}행 "
          f"({(add == 0).mean() * 100:.1f}%)")

    rule("STEP 2. Split (measurable 내부, group-aware)")
    tr, va, te = make_split(df)

    # 상수 베이스라인은 train 라벨 중앙값. (v2가 '항상 3점'과 비교한 것은
    # MAE를 최소화하는 상수가 아니라 자기에게 유리한 기준이었다 — v3에서 고친 것)
    const = int(np.median(tr['y'].values))

    rule("STEP 3. validation 그리드")
    print(f"  선택 규칙: 'val QWK >= 현행({CURRENT[0]} + 가산 x {CURRENT[1]}) 인 후보 중 val MAE 최소'")
    print("  (test는 고른 뒤 한 번만 본다)\n")

    cur_va = score(va, *CURRENT)
    print(f"  현행 폴백      base={CURRENT[0]} scale={CURRENT[1]:<5}"
          f"  MAE {cur_va['mae']:.4f}  QWK {cur_va['qwk']:.4f}")
    const_va = evaluate(va['y'].values - 1, np.full(len(va), const - 1))
    print(f"  상수({const}점)      {'':<18}MAE {const_va['mae']:.4f}  QWK {const_va['qwk']:.4f}")
    print()

    grid = []
    for b in BASES:
        for s in SCALES:
            m = score(va, b, s)
            grid.append({'base': b, 'scale': s, **m})
            if show_grid:
                mark = "  <- 현행" if (b, s) == CURRENT else ""
                print(f"    base={b} scale={s:<5} MAE {m['mae']:.4f}  "
                      f"QWK {m['qwk']:.4f}{mark}")

    ok = [g for g in grid if g['qwk'] >= cur_va['qwk']]
    print(f"\n  QWK 조건을 통과한 후보: {len(ok)}/{len(grid)}")
    if not ok:
        print("  !! 통과 후보 없음 — 현행을 유지한다")
        return
    best = min(ok, key=lambda g: (g['mae'], -g['qwk']))
    print(f"  선택: base={best['base']}  scale={best['scale']}"
          f"   (val MAE {best['mae']:.4f}, QWK {best['qwk']:.4f})")

    # BASE는 프록시로 고르지 않는다(상단 주석). SCALE만 가져오고 BASE는
    # measurable train 라벨의 중앙값 = 상수 베이스라인과 같은 규약을 쓴다.
    shipped = (const, best['scale'])
    rule("STEP 4. 채택안 — BASE는 프록시가 아니라 train 중앙값으로")
    print(f"  프록시 최적: base={best['base']} scale={best['scale']}")
    print(f"  채택       : base={shipped[0]} scale={shipped[1]}   (base = train 라벨 중앙값)")
    print("  이유: BASE는 '관측 불가능한 공고의 사전 확률'이라 프록시가 답할 수 없다.")
    print("        상세는 이 파일 상단 '[중요] 프록시가 고를 수 있는 것은 SCALE뿐이다'.")
    print("\n  폴백이 실제로 적용되는 구간(unmeasurable)에서의 쏠림:")
    _report_unmeasurable_shift([CURRENT, (best['base'], best['scale']), shipped])

    rule("STEP 5. test 1회 평가")
    print(f"  hold-out {len(te):,}행\n")
    res = {
        f'상수 ({const}점)': evaluate(te['y'].values - 1, np.full(len(te), const - 1)),
        f'현행 폴백 ({CURRENT[0]} + 가산)': score(te, *CURRENT),
        f"프록시 최적 ({best['base']} + 가산 x {best['scale']})": score(te, best['base'], best['scale']),
        f"채택 ({shipped[0]} + 가산 x {shipped[1]})": score(te, *shipped),
    }
    print(f"  {'방법':<30}{'MAE':>9}{'QWK':>9}{'±1점':>9}{'정확도':>9}")
    for name, m in res.items():
        print(f"  {name:<30}{m['mae']:>9.4f}{m['qwk']:>9.4f}"
              f"{m['off_by_1'] * 100:>8.1f}%{m['accuracy'] * 100:>8.1f}%")

    cur, new = res[f'현행 폴백 ({CURRENT[0]} + 가산)'], res[list(res)[-1]]
    print(f"\n  현행 대비: MAE {cur['mae']:.4f} -> {new['mae']:.4f} "
          f"({(new['mae'] - cur['mae']) / cur['mae'] * 100:+.1f}%)  |  "
          f"QWK {cur['qwk']:.4f} -> {new['qwk']:.4f}")
    dominates = new['mae'] < cur['mae'] and new['qwk'] > cur['qwk']
    print(f"  현행을 두 지표 모두에서 이기는가: {'예' if dominates else '아니오'}")
    c = res[f'상수 ({const}점)']
    print(f"  상수 대비: MAE {c['mae']:.4f} -> {new['mae']:.4f} "
          f"({(new['mae'] - c['mae']) / c['mae'] * 100:+.1f}%)  |  "
          f"QWK {c['qwk']:.4f} -> {new['qwk']:.4f}")
    print("  -> 상수를 MAE로 이기지는 못한다. 다만 '완전히 지배당하는' 위치에서")
    print("     'MAE를 조금 내주고 순서 정보를 얻는' 위치로 옮겨간다.")

    out = {'selection_rule': 'val QWK >= current, then min val MAE (SCALE only)',
           'base_rule': 'measurable train label median (proxy cannot inform the intercept)',
           'constant': const, 'current': {'base': CURRENT[0], 'scale': CURRENT[1]},
           'proxy_optimal': {'base': best['base'], 'scale': best['scale']},
           'chosen': {'base': shipped[0], 'scale': shipped[1]},
           'val_grid': grid, 'test': res,
           'n': {'train': len(tr), 'val': len(va), 'test': len(te)}}
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n저장: {OUT_JSON}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', action='store_true', help='validation 그리드 전체 출력')
    main(ap.parse_args().grid)
