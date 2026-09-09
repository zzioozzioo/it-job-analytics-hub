"""
compare_labels.py

"새 라벨이 옛 라벨보다 낫다"를 **지표 비교의 함정을 피해서** 확인한다.

---------------------------------------------------------------------------
왜 별도 스크립트가 필요한가
---------------------------------------------------------------------------
train_urgency.py의 EXP-B는 전이 QWK를 v2 라벨 0.0486 -> v3 라벨 0.1309로
보고한다. 그런데 이 두 숫자를 그냥 나란히 놓으면 안 된다.

  QWK도 MAE도 **라벨 분포에 의존**한다.
  v2 measurable: jobkorea가 4·5점에 45.2% 몰려 있다.
  v3 measurable: 2점에 50% 몰려 있다.

분포가 다르면 "아무것도 안 하는 예측기"의 점수부터 다르다. 라벨을 바꿔놓고
raw 지표만 비교하면, 개선인지 그냥 문제가 쉬워진 것인지 구분할 수 없다.

그래서 **각 라벨 세트마다 자기 상수 베이스라인을 놓고, 그 대비 개선폭**을
본다. 상수 예측기의 QWK는 정의상 0이므로 QWK는 그대로 쓰고, MAE는
'베이스라인 대비 몇 % 줄었나'로 환산한다.

베이스라인은 train 라벨의 **중앙값**을 상수로 찍는다(MAE를 최소화하는 상수).
train만 보고 정하므로 hold-out을 훔쳐보지 않는다.

v4 라운드도 같은 방식으로 본다(`--v4`).

실행: python compare_labels.py         # v2 라벨 vs v3 라벨
      python compare_labels.py --v4    # v3 라벨 vs v4 라벨
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

sys.stdout.reconfigure(encoding='utf-8')

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from urgency_rule import is_measurable  # noqa: E402
from train_urgency_baseline import (RANDOM_STATE, build_tfidf,  # noqa: E402
                                    evaluate, make_transform, rule)
from common.hf_data import fetch  # noqa: E402

VARIANT = 'masked+clean'
MAX_FEATURES = 30000

# (옛 라벨, 새 라벨) — 파일명만 들고 있다가 common.hf_data.fetch()로 연다.
ROUNDS = {False: ('v2', 'v3'), True: ('v3', 'v4')}


def load(old_tag, new_tag):
    """두 라벨 세트를 같은 행에 나란히 붙인다.

    ⚠️ (source, job_id) dict 조인을 쓰지 않는다. 같은 키가 777종(1,554행)
       중복돼 있어 서로 다른 공고가 짝지어진다(`urgency_rule.load_labels()`
       주석 참조). 모든 라벨 파일이 master_merged.json을 같은 순서로 훑어
       만들어지므로 위치로 맞춘다."""
    with open(fetch(f"master_merged_{new_tag}.json"), encoding='utf-8') as f:
        rows_new = json.load(f)
    with open(fetch(f"master_merged_{old_tag}.json"), encoding='utf-8') as f:
        rows_old = json.load(f)
    if len(rows_old) != len(rows_new) or any(
            (a.get('source'), a.get('job_id')) != (b.get('source'), b.get('job_id'))
            for a, b in zip(rows_old, rows_new)):
        raise SystemExit(f"  [!] {old_tag}와 {new_tag}의 행 정렬이 다르다 — 비교 불가")

    df = pd.DataFrame(rows_new)
    df['raw_text'] = df['raw_text'].fillna('').astype(str)
    df['source'] = df['source'].fillna('unknown').astype(str)
    df = df.rename(columns={'urgency_score': 'y_new'})
    df['y_new'] = df['y_new'].astype(int)
    df['y_old'] = [r['urgency_score'] for r in rows_old]
    df['y_old'] = df['y_old'].astype(int)
    return df[df['raw_text'].map(is_measurable)].reset_index(drop=True)


def main(use_v4=False):
    old_tag, new_tag = ROUNDS[use_v4]
    tag_of = {'y_old': old_tag, 'y_new': new_tag}
    meas = load(old_tag, new_tag)
    conv = make_transform(VARIANT)
    srcs = [s for s in meas['source'].unique() if (meas['source'] == s).sum() >= 500]

    rule("leave-one-source-out 전이 — 상수 베이스라인 대비로 비교")
    print("  라벨 세트가 다르면 지표의 기준선도 다르다. 각자의 베이스라인을 깔고 본다.")
    print("  베이스라인 = train 라벨의 중앙값을 상수로 찍기 (MAE 최소 상수, train만 사용)")
    print()

    rows = []
    for label in ['y_old', 'y_new']:
        for src in srcs:
            tr_m, te_m = meas['source'] != src, meas['source'] == src
            ytr = (meas.loc[tr_m, label] - 1).to_numpy()
            yte = (meas.loc[te_m, label] - 1).to_numpy()

            base = int(np.median(ytr))                 # train만 보고 결정
            b = evaluate(yte, np.full(len(yte), base))

            vec, Xtr, (Xte,), _ = build_tfidf(
                conv(meas.loc[tr_m, 'raw_text'], meas.loc[tr_m, 'source']),
                [conv(meas.loc[te_m, 'raw_text'], meas.loc[te_m, 'source'])],
                MAX_FEATURES)

            # ⚠️ class_weight='balanced'는 소수 클래스 재현율을 위해 예측을
            #    일부러 퍼뜨린다. 라벨이 한 클래스에 몰린 v3에서는 그 자체로
            #    MAE가 나빠진다 — 상수 대비 MAE로만 재면 balanced 모델이
            #    부당하게 불리하다. 가중치 없는 모델을 함께 놓아 분리한다.
            fits = {}
            for tag, kw in [('balanced', {'class_weight': 'balanced'}),
                            ('plain', {})]:
                svc = LinearSVC(C=1.0, max_iter=5000,
                                random_state=RANDOM_STATE, **kw).fit(Xtr, ytr)
                fits[tag] = evaluate(yte, svc.predict(Xte))

            rows.append({
                'label': tag_of[label], 'target': src, 'n': int(te_m.sum()),
                'base_const': base + 1,
                'base_mae': b['mae'], 'base_off_by_1': b['off_by_1'],
                'bal_mae': fits['balanced']['mae'],
                'bal_mae_gain_%': (b['mae'] - fits['balanced']['mae']) / b['mae'] * 100,
                'bal_qwk': fits['balanced']['qwk'],
                'plain_mae': fits['plain']['mae'],
                'plain_mae_gain_%': (b['mae'] - fits['plain']['mae']) / b['mae'] * 100,
                'plain_qwk': fits['plain']['qwk'],
            })

    df = pd.DataFrame(rows)
    print(f"  {'라벨':<5}{'대상':<10}{'n':>8}{'상수':>5}{'상수MAE':>9}"
          f"{'bal MAE':>9}{'개선':>8}{'bal QWK':>9}"
          f"{'plain MAE':>11}{'개선':>8}{'plain QWK':>11}")
    for _, r in df.iterrows():
        print(f"  {r['label']:<5}{r['target']:<10}{r['n']:>8,}{r['base_const']:>5}"
              f"{r['base_mae']:>9.4f}{r['bal_mae']:>9.4f}{r['bal_mae_gain_%']:>7.1f}%"
              f"{r['bal_qwk']:>9.4f}"
              f"{r['plain_mae']:>11.4f}{r['plain_mae_gain_%']:>7.1f}%"
              f"{r['plain_qwk']:>11.4f}")

    print()
    print("  === 라벨 세트별 평균 ===")
    cols = ['bal_mae_gain_%', 'bal_qwk', 'plain_mae_gain_%', 'plain_qwk']
    g = df.groupby('label')[cols].mean()
    print(f"    {'라벨':<6}{'bal MAE개선':>13}{'bal QWK':>10}"
          f"{'plain MAE개선':>15}{'plain QWK':>12}")
    for lab, r in g.iterrows():
        print(f"    {lab:<6}{r['bal_mae_gain_%']:>12.1f}%{r['bal_qwk']:>10.4f}"
              f"{r['plain_mae_gain_%']:>14.1f}%{r['plain_qwk']:>12.4f}")

    print()
    print("  읽는 법")
    print("    · QWK는 라벨 세트가 달라도 '상수 예측기 = 0'이 공통 기준점이라 비교 가능하다.")
    print("    · MAE 개선은 각 라벨 세트의 자기 상수 대비 %이므로 분포 차이가 상쇄된다.")
    print("    · MAE 개선이 음수면 그 모델은 '항상 같은 값 찍기'보다 못하다는 뜻이다.")

    out = HERE / "compare_labels.json"
    out.write_text(json.dumps(df.to_dict('records'), ensure_ascii=False,
                              indent=2, default=float), encoding='utf-8')
    print(f"\n  저장: {out.name}")


if __name__ == '__main__':
    main('--v4' in sys.argv)
