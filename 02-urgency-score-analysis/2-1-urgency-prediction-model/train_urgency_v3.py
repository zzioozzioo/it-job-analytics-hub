"""
train_urgency_v3.py

v3 라벨(`urgency_rule.py`)로 다시 학습하고, v2 대비 무엇이 나아졌는지를
같은 파이프라인 위에서 측정한다.

---------------------------------------------------------------------------
이 스크립트가 답하려는 질문
---------------------------------------------------------------------------
v2 모델의 결론은 "in-domain은 QWK 0.79로 쓸 만한데 cross-source 전이는
QWK 0.04로 무작위"였고, 원인을 **라벨**로 지목했다. 소스마다 다른 규칙으로
만든 라벨은 소스별로 다른 타깃이라는 것이다.

v3은 그 진단대로 라벨을 고쳤다(`urgency_rule.py` 참조).
  [수정 1] 지원자 수를 모집인원으로 오인하던 버그
  [수정 2] 마감 신호를 소스 무관 '접수 창 길이'로 통일

진단이 맞았다면 **leave-one-source-out QWK가 올라야 한다.** 그것 하나가
이 작업의 성패를 가른다. in-domain QWK는 올라도 의미가 약하다 — 라벨이
바뀌었으니 난이도도 바뀌었고, 규칙을 잘 복제한다는 뜻일 뿐이다.

그래서 EXP-B(전이)를 v2 라벨과 v3 라벨 **양쪽에서 똑같이** 돌린다.
같은 행 · 같은 split · 같은 모델 · 같은 텍스트 조건이고 라벨만 다르다.
(v2와 v3의 is_measurable은 동일하므로 학습 행 집합이 정확히 같다.)

---------------------------------------------------------------------------
v2 설계에서 고친 것
---------------------------------------------------------------------------
train_urgency_transfer.py는 EXP-A의 모델을 **test QWK로** 골랐다
(`best_name = max(results['exp_a'], key=... test qwk)`). 모델 선택에 test를
쓰면 그 test 점수는 더 이상 hold-out이 아니다. v3은 validation으로 고른다.
예측 방식(argmax vs 기댓값 반올림)도 validation에서 정한다.

실행:
  python train_urgency_v3.py                # 전체 (약 15분)
  python train_urgency_v3.py --skip-xgb     # 선형만 (수십 초, 스모크)
  python train_urgency_v3.py --sample 8000  # 축소
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

sys.stdout.reconfigure(encoding='utf-8')

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # 02-urgency-score-analysis/
sys.path.insert(0, str(HERE.parents[1]))      # repo root

from urgency_rule import is_measurable, score_by_vocabulary  # noqa: E402
from train_urgency_baseline import (N_CLASSES, RANDOM_STATE,  # noqa: E402
                                    Timer, build_tfidf, dedup_and_group,
                                    evaluate, make_transform, rule)

DATA_V3 = HERE.parents[1] / "data" / "master_merged_v3.json"
DATA_V2 = HERE.parents[1] / "data" / "master_merged_v2.json"
OUT_DIR = HERE / "models_v3"
SMOKE_DIR = HERE / "models_v3_smoke"

VARIANT = 'masked+clean'
TEST_FOLDS = 5


# ---------------------------------------------------------------------------
def load_with_both_labels():
    """v3 데이터를 읽고 같은 행에 v2 라벨을 붙인다.

    (source, job_id)로 조인한다. 두 파일 모두 master_merged.json에서
    라벨만 갈아끼운 것이므로 행 집합이 같다."""
    rule("STEP 1. 데이터 로드 (v3 라벨 + v2 라벨 나란히)")
    with open(DATA_V3, encoding='utf-8') as f:
        df = pd.DataFrame(json.load(f))
    df['raw_text'] = df['raw_text'].fillna('').astype(str)
    df['source'] = df['source'].fillna('unknown').astype(str)
    df = df.rename(columns={'urgency_score': 'y_v3'})
    df['y_v3'] = df['y_v3'].astype(int)

    with open(DATA_V2, encoding='utf-8') as f:
        v2 = {(r['source'], r['job_id']): r['urgency_score'] for r in json.load(f)}
    df['y_v2'] = [v2.get((s, j)) for s, j in zip(df['source'], df['job_id'])]
    missing = int(df['y_v2'].isna().sum())
    if missing:
        print(f"  [!] v2 라벨을 못 찾은 행 {missing:,}개 -> 비교에서 제외")
        df = df[df['y_v2'].notna()].copy()
    df['y_v2'] = df['y_v2'].astype(int)
    df = df.reset_index(drop=True)

    df['measurable'] = df['raw_text'].map(is_measurable)
    m = df[df.measurable].reset_index(drop=True)
    u = df[~df.measurable].reset_index(drop=True)
    print(f"  전체 {len(df):,}행  /  measurable {len(m):,} ({len(m)/len(df)*100:.1f}%) "
          f"/ unmeasurable {len(u):,}")
    print(f"  measurable source: {m['source'].value_counts().to_dict()}")

    print()
    print("  measurable 라벨 분포 (%) — 소스별로 같은 모양이어야 '같은 타깃'이다")
    for tag in ['y_v2', 'y_v3']:
        print(f"    [{tag[-2:]}]")
        for s in ['jobkorea', 'saramin']:
            sub = m[m['source'] == s]
            if not len(sub):
                continue
            vc = sub[tag].value_counts(normalize=True).mul(100)
            line = '  '.join(f"{lv}:{vc.get(lv, 0.0):5.1f}%" for lv in range(1, 6))
            top2 = vc.get(4, 0.0) + vc.get(5, 0.0)
            print(f"      {s:<9} {line}   상위등급(4+5) {top2:5.1f}%")
    print()
    print("  -> v2는 jobkorea와 saramin의 상위등급 비율이 몇 배씩 차이 난다.")
    print("     같은 개념을 쟀다면 나올 수 없는 격차이고, 이것이 전이 실패의 원인이다.")
    return m, u


def make_split(meas):
    """중복 제거 + group-aware 3분할. 라벨과 무관하게 한 번만 만든다.

    v2/v3 비교가 공정하려면 split이 같아야 한다. stratify 키는 v3 라벨로
    잡는다(어차피 최종 모델은 v3)."""
    meas, groups, dedup_stats = dedup_and_group(meas)
    rule("STEP 2. Split (measurable 내부, group-aware)")
    strat = meas['source'] + "__" + meas['y_v3'].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=TEST_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    rest_pos, te_pos = next(sgkf.split(meas, strat, groups))
    sgkf2 = StratifiedGroupKFold(n_splits=TEST_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)
    tr_pos, va_pos = next(sgkf2.split(meas.iloc[rest_pos],
                                      strat.iloc[rest_pos], groups.iloc[rest_pos]))
    rest = meas.iloc[rest_pos]
    tr = rest.iloc[tr_pos].reset_index(drop=True)
    va = rest.iloc[va_pos].reset_index(drop=True)
    te = meas.iloc[te_pos].reset_index(drop=True)
    shared = set(groups.iloc[te_pos]) & set(groups.iloc[rest_pos])
    print(f"  train {len(tr):,} / val {len(va):,} / test {len(te):,}")
    print(f"  공유 그룹 {len(shared)}개 {'(정상)' if not shared else '(!! 누출)'}")
    return meas, tr, va, te, dedup_stats


def fit_models(txt_tr, ytr, txt_va, yva, args, tag, skip_xgb=None):
    """(이름 -> 모델) 과 vectorizer. 예측은 호출부에서 필요한 만큼 한다."""
    skip = args.skip_xgb if skip_xgb is None else skip_xgb
    vec, Xtr, (Xva,), vs = build_tfidf(txt_tr, [txt_va], args.max_features)
    classes = np.unique(ytr)
    wmap = dict(zip(classes, compute_class_weight('balanced', classes=classes, y=ytr)))
    sw = np.array([wmap[y] for y in ytr])

    out = {}
    with Timer() as t:
        svc = LinearSVC(class_weight='balanced', C=1.0, max_iter=5000,
                        random_state=RANDOM_STATE).fit(Xtr, ytr)
    out['linear_svc'] = svc
    print(f"    [{tag}] TF-IDF {Xtr.shape[1]:,}f ({vs:.1f}s) · linear_svc {t.cpu:.1f}s")

    if not skip:
        xgb = XGBClassifier(
            n_estimators=args.n_estimators, early_stopping_rounds=args.early_stopping,
            max_depth=7, learning_rate=0.2, subsample=0.9, colsample_bytree=0.7,
            tree_method='hist', objective='multi:softmax', num_class=len(classes),
            n_jobs=-1, random_state=RANDOM_STATE, eval_metric='mlogloss')
        with Timer() as t:
            xgb.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
        out['xgboost_es'] = xgb
        print(f"    [{tag}] xgboost_es {t.cpu:.1f}s "
              f"(early stop @ {int(xgb.best_iteration) + 1}/{args.n_estimators})")
    return out, vec


def show(name, y_true, y_pred, indent='    '):
    s = evaluate(y_true, y_pred)
    print(f"{indent}{name:<28} QWK {s['qwk']:>7.4f}  MAE {s['mae']:>6.4f}  "
          f"±1 {s['off_by_1']:>6.4f}  MacroF1 {s['macro_f1']:>6.4f}  "
          f"Acc {s['accuracy']:>6.4f}")
    return s


def expected_round(proba):
    """클래스 확률의 기댓값을 반올림. 순서형 라벨에서 argmax보다 낫다.

    argmax는 '2점 0.45 / 3점 0.44'인 공고를 2점으로 단정한다. 기댓값은
    2.5 근처를 돌려주므로 QWK·MAE가 개선된다(v2에서 확인된 효과)."""
    ev = proba @ np.arange(N_CLASSES)
    return np.clip(np.rint(ev), 0, N_CLASSES - 1).astype(int)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=None)
    ap.add_argument('--max-features', type=int, default=30000)
    ap.add_argument('--n-estimators', type=int, default=400)
    ap.add_argument('--early-stopping', type=int, default=30)
    ap.add_argument('--skip-xgb', action='store_true')
    args = ap.parse_args()
    t_start = time.time()

    meas, unmeas = load_with_both_labels()
    if args.sample:
        meas = meas.sample(n=min(args.sample, len(meas)),
                           random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"  [!] --sample: measurable {len(meas):,}행만 사용")

    meas, tr, va, te, dedup_stats = make_split(meas)

    conv = make_transform(VARIANT)
    txt = {k: conv(d['raw_text'], d['source']) for k, d in
           [('tr', tr), ('va', va), ('te', te)]}
    print(f"  텍스트 조건 = {VARIANT} "
          f"(평균 {tr['raw_text'].str.len().mean():.0f}자 -> {txt['tr'].str.len().mean():.0f}자)")

    results = {}

    # -----------------------------------------------------------------------
    rule("STEP 3. EXP-A  in-domain (v3 라벨) — 모델 선택은 validation으로")
    print("  ⚠️ v2 스크립트는 여기서 test QWK로 모델을 골랐다(test contamination).")
    print("     v3은 validation으로 고르고, test는 마지막에 한 번만 본다.")
    print()
    y = {k: (d['y_v3'] - 1).to_numpy() for k, d in [('tr', tr), ('va', va), ('te', te)]}
    models, vec = fit_models(txt['tr'], y['tr'], txt['va'], y['va'], args, 'A')
    Xva, Xte = vec.transform(txt['va']), vec.transform(txt['te'])

    print()
    print("  [validation] 모델 선택")
    val_scores = {n: show(n, y['va'], m.predict(Xva), '      ')
                  for n, m in models.items()}
    best_name = max(val_scores, key=lambda k: val_scores[k]['qwk'])
    best = models[best_name]
    print(f"      -> 채택: {best_name}")
    results['exp_a_val'] = val_scores
    results['selected_model'] = best_name

    # -----------------------------------------------------------------------
    rule("STEP 4. EXP-B  cross-source 전이 — v2 라벨 vs v3 라벨  ★핵심")
    print("  한 소스로 배워 다른 소스를 맞힌다. source 교란이 제거된 진짜 일반화다.")
    print("  같은 행 · 같은 텍스트 · 같은 모델(LinearSVC)이고 라벨만 다르다.")
    print("  라벨 수정이 옳았다면 v3의 QWK가 v2보다 높아야 한다.")
    print()
    loso = {}
    srcs = [s for s in meas['source'].unique() if (meas['source'] == s).sum() >= 500]
    for label_col in ['y_v2', 'y_v3']:
        loso[label_col] = {}
        print(f"  [{label_col[-2:]} 라벨]")
        for src in srcs:
            tr_m, te_m = meas['source'] != src, meas['source'] == src
            sub, sub_vec = fit_models(
                conv(meas.loc[tr_m, 'raw_text'], meas.loc[tr_m, 'source']),
                (meas.loc[tr_m, label_col] - 1).to_numpy(),
                txt['va'], (va[label_col] - 1).to_numpy(),
                args, f'B/{label_col[-2:]}/{src}', skip_xgb=True)
            pred = sub['linear_svc'].predict(sub_vec.transform(
                conv(meas.loc[te_m, 'raw_text'], meas.loc[te_m, 'source'])))
            loso[label_col][src] = show(
                f'-> {src} ({int(te_m.sum()):,}행)',
                (meas.loc[te_m, label_col] - 1).to_numpy(), pred, '      ')
        print()
    results['exp_b_loso'] = loso

    print("  === 전이 QWK 요약 ===")
    print(f"    {'평가 대상':<14}{'v2 라벨':>10}{'v3 라벨':>10}{'변화':>10}")
    for src in srcs:
        a, b = loso['y_v2'][src]['qwk'], loso['y_v3'][src]['qwk']
        print(f"    {src:<14}{a:>10.4f}{b:>10.4f}{b - a:>+10.4f}")
    mv2 = float(np.mean([loso['y_v2'][s]['qwk'] for s in srcs]))
    mv3 = float(np.mean([loso['y_v3'][s]['qwk'] for s in srcs]))
    print(f"    {'평균':<14}{mv2:>10.4f}{mv3:>10.4f}{mv3 - mv2:>+10.4f}")
    results['exp_b_summary'] = {'mean_qwk_v2': mv2, 'mean_qwk_v3': mv3}

    # -----------------------------------------------------------------------
    rule("STEP 5. EXP-C  어휘 폴백 대체 검증 (v3 라벨)")
    print("  measurable hold-out을 '메타데이터 없는 상태'로 두고 같은 정답 위에서 비교한다.")
    print()
    fb = np.array([score_by_vocabulary(t)[0] - 1 for t in te['raw_text']])
    results['exp_c'] = {
        'always_3': show('상수 baseline (항상 3점)', y['te'], np.full(len(y['te']), 2)),
        'vocab_fallback': show('현행 어휘 폴백', y['te'], fb),
    }

    # -----------------------------------------------------------------------
    rule("STEP 6. 예측 방식 확정 (validation) 후 test 1회 평가")
    proba_va = best.predict_proba(Xva) if hasattr(best, 'predict_proba') else None
    if proba_va is None:
        print("  선택된 모델에 predict_proba가 없다 -> argmax 사용")
        mode = 'argmax'
    else:
        s_arg = show('validation / argmax', y['va'], best.predict(Xva), '    ')
        s_exp = show('validation / 기댓값 반올림', y['va'], expected_round(proba_va), '    ')
        mode = 'expected_round' if s_exp['qwk'] >= s_arg['qwk'] else 'argmax'
        print(f"    -> 채택: {mode}")
        results['prediction_choice'] = {'argmax': s_arg, 'expected_round': s_exp,
                                        'selected': mode}

    def final_predict(X):
        if mode == 'expected_round':
            return expected_round(best.predict_proba(X))
        return best.predict(X)

    print()
    print("  [test] 최종 1회 평가")
    final = show(f'{best_name} / {mode}', y['te'], final_predict(Xte), '    ')
    results['final_test'] = final
    print()
    print(f"    모델 QWK {final['qwk']:.4f}  vs  어휘 폴백 "
          f"{results['exp_c']['vocab_fallback']['qwk']:.4f}  vs  상수 "
          f"{results['exp_c']['always_3']['qwk']:.4f}")

    # -----------------------------------------------------------------------
    rule("STEP 7. 저장")
    out = SMOKE_DIR if (args.sample or args.skip_xgb) else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    if out is SMOKE_DIR:
        print(f"  [!] 축소 실행이므로 {out.name}/ 에 저장")
    joblib.dump(best, out / "urgency_model.joblib")
    joblib.dump(vec, out / "urgency_tfidf.joblib")
    (out / "model_meta.json").write_text(json.dumps({
        'rule_version': 'v3',
        'data': DATA_V3.name,
        'variant': VARIANT,
        'model': best_name,
        'prediction': mode,
        'model_selection': 'validation QWK (test는 최종 1회만)',
        'dedup': dedup_stats,
        'split': {'train': len(tr), 'val': len(va), 'test': len(te),
                  'unmeasurable': len(unmeas)},
        'results': results,
    }, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    for p in sorted(out.iterdir()):
        print(f"  저장: {p.name}  ({p.stat().st_size / 1e6:.2f} MB)")
    print()
    print(f"  총 소요 {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
