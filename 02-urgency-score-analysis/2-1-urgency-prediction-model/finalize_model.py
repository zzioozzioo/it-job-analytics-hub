"""
finalize_model.py

최종 모델 확정 + 순서형 예측 방식 비교.

[왜 기댓값 예측을 보는가]
라벨 1~5는 순서형인데 학습은 multi:softmax(명목형)로 했다. 예측도 argmax라
'2점일 확률 0.45 / 3점일 확률 0.44'인 공고를 2점으로 단정한다.
순서형에서는 클래스 확률의 기댓값 E[y] = sum(p_i * i)를 반올림하는 편이
인접 오차를 줄여 QWK/MAE가 좋아지는 경우가 많다.

이미 학습된 모델의 predict_proba만 쓰므로 재학습 비용이 0이다.
4점 클래스가 계속 무너졌던 것(precision 0.5 수준)도 구간 경계 문제라
기댓값 반올림이 완화할 수 있다.

실행: python finalize_model.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding='utf-8')

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from rescore_urgency import is_measurable  # noqa: E402
from train_urgency_baseline import (LOCAL_DATA, N_CLASSES,  # noqa: E402
                                    RANDOM_STATE, dedup_and_group, evaluate,
                                    make_transform, rule)

MODEL_DIR = HERE / "models_transfer"
VARIANT = 'masked+clean'
TEST_FOLDS = 5


def rebuild_test_split():
    """train_urgency_transfer.py와 동일한 test 집합을 재현한다.
    (시드/분할 규칙이 결정론적이라 그대로 재현된다)"""
    with open(LOCAL_DATA, encoding='utf-8') as f:
        df = pd.DataFrame(json.load(f))
    df['raw_text'] = df['raw_text'].fillna('').astype(str)
    df['source'] = df['source'].fillna('unknown').astype(str)
    df = df[df['urgency_score'].notna()].copy()
    df['urgency_score'] = df['urgency_score'].astype(int)
    df = df.reset_index(drop=True)
    df['measurable'] = df['raw_text'].map(is_measurable)

    meas = df[df.measurable].reset_index(drop=True)
    meas, groups, _ = dedup_and_group(meas)
    strat = meas['source'] + "__" + meas['urgency_score'].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=TEST_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    _, te_pos = next(sgkf.split(meas, strat, groups))
    return meas.iloc[te_pos].reset_index(drop=True)


def main():
    rule("STEP 1. 저장된 모델 로드 & test 재현")
    clf = joblib.load(MODEL_DIR / "transfer_model.joblib")
    vec = joblib.load(MODEL_DIR / "transfer_tfidf.joblib")
    te = rebuild_test_split()
    print(f"  test {len(te):,}행")

    conv = make_transform(VARIANT)
    X = vec.transform(conv(te['raw_text'], te['source']))
    y = (te['urgency_score'] - 1).to_numpy()

    rule("STEP 2. 예측 방식 비교  [개선 5 · 9]")
    proba = clf.predict_proba(X)
    pred_argmax = proba.argmax(axis=1)
    ev = proba @ np.arange(N_CLASSES)          # 기댓값 E[y]
    pred_ev = np.clip(np.rint(ev), 0, N_CLASSES - 1).astype(int)

    rows = {}
    for name, p in [('argmax (현행)', pred_argmax), ('기댓값 반올림', pred_ev)]:
        s = evaluate(y, p)
        rows[name] = s
        print(f"  {name:<16} QWK {s['qwk']:>7.4f}  MAE {s['mae']:>6.4f}  "
              f"MacroF1 {s['macro_f1']:>6.4f}  Acc {s['accuracy']:>6.4f}  "
              f"인접오차내 {s['off_by_1']:>6.4f}")

    d_qwk = rows['기댓값 반올림']['qwk'] - rows['argmax (현행)']['qwk']
    d_mae = rows['기댓값 반올림']['mae'] - rows['argmax (현행)']['mae']
    use_ev = d_qwk > 0 and d_mae < 0
    print()
    print(f"  차이: QWK {d_qwk:+.4f} / MAE {d_mae:+.4f}")
    print(f"  -> 최종 예측 방식: {'기댓값 반올림' if use_ev else 'argmax'}")

    final = pred_ev if use_ev else pred_argmax
    rule("STEP 3. 최종 모델 상세")
    print(classification_report(y, final, labels=list(range(N_CLASSES)),
                                target_names=[f'{i}점' for i in range(1, 6)],
                                digits=3, zero_division=0))
    cm = confusion_matrix(y, final, labels=list(range(N_CLASSES)))
    print("  Confusion Matrix (행=실제, 열=예측):")
    print("        " + "".join(f"{f'pred{i}':>8}" for i in range(1, 6)))
    for i, row in enumerate(cm):
        print(f"  true{i+1} " + "".join(f"{v:>8,}" for v in row))

    print()
    print("  source별:")
    print(f"    {'source':<10}{'n':>8}{'QWK':>9}{'MAE':>8}{'MacroF1':>10}")
    per_source = {}
    for src in sorted(te['source'].unique()):
        m = (te['source'] == src).to_numpy()
        s = evaluate(y[m], final[m])
        per_source[src] = s
        print(f"    {src:<10}{int(m.sum()):>8,}{s['qwk']:>9.4f}"
              f"{s['mae']:>8.4f}{s['macro_f1']:>10.4f}")

    rule("STEP 4. 저장")
    meta_path = MODEL_DIR / "transfer_meta.json"
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    meta['prediction'] = {
        'method': 'expected_value_rounding' if use_ev else 'argmax',
        'rationale': ('순서형 라벨이므로 클래스 확률의 기댓값을 반올림한다. '
                      'test에서 QWK/MAE가 함께 개선될 때만 채택.'),
        'comparison': {k: {kk: vv for kk, vv in v.items()} for k, v in rows.items()},
        'final': evaluate(y, final),
        'per_source': per_source,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float),
                         encoding='utf-8')
    print(f"  갱신: {meta_path.name}")


if __name__ == '__main__':
    main()
