"""
train_urgency_transfer.py

개선 6번: 모델에 존재 이유를 주는 재설계.

[왜 v1/baseline 설계로는 안 되는가]
urgency_score는 rescore_urgency.py가 raw_text에서 정규식으로 만든 결정론적
라벨이다. 같은 raw_text를 입력으로 주면 정규식을 그냥 돌려서 F1 = 1.0을 얻는다.
모델이 20분 학습해서 0.86을 내는 것은 정규식의 열화 복제일 뿐이다.

[모델이 실제로 값어치를 갖는 지점]
규칙이 '계산할 수 없는' 공고가 있다. rescore_urgency.is_measurable()로 나뉜다.

  measurable   30,569행 (75.8%)  마감일/접수기간/모집인원 같은 메타데이터가 본문에
                                 있어 규칙이 근거를 읽고 라벨을 만들었다. 검증된 라벨.
  unmeasurable  9,779행 (24.2%)  메타데이터가 아예 없다. 규칙이 '어조 프록시'
                                 (score_by_vocabulary)로 3점에서 가산만 했다.
                                 rescore_urgency.py 주석이 스스로 '검증된 채용
                                 시급성이 아니다'라고 적어둔, 미검증 라벨이다.

  unmeasurable 내역: saramin 5,315 / wanted 4,372 / jobkorea 92
  -> wanted만의 문제가 아니다. 전체의 1/4이 미검증 라벨이다.
  -> 라벨 분포도 갈린다: unmeasurable은 3점이 83.3%, 1·2점이 0%.
     폴백이 3점에서 시작해 가산만 하기 때문이다.

따라서 과제를 이렇게 다시 정의한다.

  학습: measurable 공고만 (검증된 라벨)
  적용: unmeasurable 공고 (현행 어휘 폴백을 대체)

[텍스트 조건이 masked+clean 이어야 하는 이유]
정직한 평가 때문만이 아니다. 도메인을 맞추기 위해서다.
적용 대상인 unmeasurable 공고에는 마감일·모집인원 구간이 애초에 없고
(wanted는 사이트 템플릿도 없다), 학습 텍스트에만 그게 남아 있으면
train/serve skew가 생긴다. 모델이 실전에서 절대 못 보는 단서에 의존하게 된다.
그래서 학습 텍스트에서도 같은 구간을 지운다.

[핵심 검증: EXP-C]
unmeasurable에는 정답이 없어서 '폴백보다 나은가'를 직접 잴 수 없다.
그래서 measurable hold-out을 메타데이터 없는 상태로 만들어 시뮬레이션한다.
같은 정답(규칙 라벨) 위에서 두 방법을 비교한다.

  정답      : 규칙 라벨 (메타데이터를 읽고 만든 것)
  방법 1    : score_by_vocabulary() = 현행 어휘 폴백
  방법 2    : 이 모델 (masked+clean 텍스트로 예측)

모델이 폴백보다 정답에 가까우면, unmeasurable 9,779행의 라벨을 모델로
교체하는 것이 데이터셋 품질을 올린다는 직접 증거가 된다.

실행:
  python train_urgency_transfer.py                # 전체
  python train_urgency_transfer.py --skip-xgb     # 선형만 (수 초)
  python train_urgency_transfer.py --sample 8000  # 스모크
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

sys.stdout.reconfigure(encoding='utf-8')

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from rescore_urgency import is_measurable, score_by_vocabulary  # noqa: E402
from train_urgency_baseline import (LOCAL_DATA, N_CLASSES,  # noqa: E402
                                    RANDOM_STATE, Timer, build_tfidf,
                                    dedup_and_group, evaluate, make_transform,
                                    rule)

OUT_DIR = HERE / "models_transfer"
SMOKE_DIR = HERE / "models_transfer_smoke"

VARIANT = 'masked+clean'   # 적용 도메인과 텍스트 분포를 맞추는 조건
TEST_FOLDS = 5


# ---------------------------------------------------------------------------
def load_split():
    rule("STEP 1. 데이터 로드 & measurable 분할  [개선 6]")
    with open(LOCAL_DATA, encoding='utf-8') as f:
        df = pd.DataFrame(json.load(f))
    df['raw_text'] = df['raw_text'].fillna('').astype(str)
    df['source'] = df['source'].fillna('unknown').astype(str)
    df = df[df['urgency_score'].notna()].copy()
    df['urgency_score'] = df['urgency_score'].astype(int)
    df = df.reset_index(drop=True)

    df['measurable'] = df['raw_text'].map(is_measurable)
    m, u = df[df.measurable].copy(), df[~df.measurable].copy()
    print(f"  전체 {len(df):,}행")
    print(f"  measurable   {len(m):,}행 ({len(m)/len(df)*100:.1f}%)  <- 학습 (검증된 라벨)")
    print(f"    source: {m['source'].value_counts().to_dict()}")
    print(f"  unmeasurable {len(u):,}행 ({len(u)/len(df)*100:.1f}%)  <- 적용 (미검증 폴백 라벨)")
    print(f"    source: {u['source'].value_counts().to_dict()}")
    print()
    print("  라벨 분포 비교 (%):")
    comp = pd.DataFrame({
        'measurable': m['urgency_score'].value_counts(normalize=True).mul(100),
        'unmeasurable': u['urgency_score'].value_counts(normalize=True).mul(100),
    }).sort_index().round(1).fillna(0.0)
    print(comp.to_string())
    print("  -> unmeasurable은 3점에 몰려 있다. 폴백이 3점에서 가산만 하기 때문이다.")
    return m.reset_index(drop=True), u.reset_index(drop=True)


def fit_models(txt_tr, ytr, txt_va, yva, txt_te, args, tag):
    """선형 + XGBoost를 학습해 (이름 -> (모델, 예측)) 로 돌려준다."""
    vec, Xtr, (Xva, Xte), vs = build_tfidf(txt_tr, [txt_va, txt_te], args.max_features)
    print(f"    TF-IDF {Xtr.shape[1]:,} features (fit {vs:.1f}s)")

    classes = np.unique(ytr)
    wmap = dict(zip(classes, compute_class_weight('balanced', classes=classes, y=ytr)))
    sw = np.array([wmap[y] for y in ytr])

    out = {}
    with Timer() as t:
        svc = LinearSVC(class_weight='balanced', C=1.0, max_iter=5000,
                        random_state=RANDOM_STATE).fit(Xtr, ytr)
    out['linear_svc'] = (svc, svc.predict(Xte), t.cpu)
    print(f"    [{tag}/linear_svc] 학습 {t.cpu:.1f}s")

    if not args.skip_xgb:
        xgb = XGBClassifier(
            n_estimators=args.n_estimators, early_stopping_rounds=args.early_stopping,
            max_depth=7, learning_rate=0.2, subsample=0.9, colsample_bytree=0.7,
            tree_method='hist', objective='multi:softmax', num_class=len(classes),
            n_jobs=-1, random_state=RANDOM_STATE, eval_metric='mlogloss')
        with Timer() as t:
            xgb.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
        out['xgboost_es'] = (xgb, xgb.predict(Xte), t.cpu)
        print(f"    [{tag}/xgboost_es] 학습 {t.cpu:.1f}s "
              f"(early stop @ {int(xgb.best_iteration)+1}/{args.n_estimators})")
    return out, vec


def show(name, y_true, y_pred, extra=''):
    s = evaluate(y_true, y_pred)
    print(f"    {name:<22} QWK {s['qwk']:>7.4f}  MAE {s['mae']:>6.4f}  "
          f"MacroF1 {s['macro_f1']:>6.4f}  Acc {s['accuracy']:>6.4f} {extra}")
    return s


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

    meas, unmeas = load_split()
    if args.sample:
        meas = meas.sample(n=min(args.sample, len(meas)),
                           random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"  [!] --sample {args.sample}: measurable {len(meas):,}행만 사용")

    # --- 중복 제거 + group split (measurable 안에서) ---
    meas, groups, dedup_stats = dedup_and_group(meas)

    rule("STEP 2. Split (measurable 내부, group-aware)")
    strat = meas['source'] + "__" + meas['urgency_score'].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=TEST_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rest_pos, te_pos = next(sgkf.split(meas, strat, groups))
    rest, te = meas.iloc[rest_pos], meas.iloc[te_pos]
    sgkf2 = StratifiedGroupKFold(n_splits=TEST_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    tr_pos, va_pos = next(sgkf2.split(rest, strat.iloc[rest_pos], groups.iloc[rest_pos]))
    tr, va = rest.iloc[tr_pos].reset_index(drop=True), rest.iloc[va_pos].reset_index(drop=True)
    te = te.reset_index(drop=True)
    print(f"  train {len(tr):,} / val {len(va):,} / test {len(te):,}")
    shared = set(groups.iloc[te_pos]) & set(groups.iloc[rest_pos])
    print(f"  공유 그룹 {len(shared)}개 {'(정상)' if not shared else '(!! 누출)'}")

    conv = make_transform(VARIANT)
    txt = {k: conv(d['raw_text'], d['source']) for k, d in
           [('tr', tr), ('va', va), ('te', te)]}
    y = {k: (d['urgency_score'] - 1).to_numpy() for k, d in
         [('tr', tr), ('va', va), ('te', te)]}
    print(f"  텍스트 조건 = {VARIANT} "
          f"(평균 {tr['raw_text'].str.len().mean():.0f}자 -> {txt['tr'].str.len().mean():.0f}자)")

    results = {}

    # --- EXP-A: in-domain ---
    rule("STEP 3. EXP-A  in-domain (measurable 학습 -> measurable hold-out)")
    models, vec = fit_models(txt['tr'], y['tr'], txt['va'], y['va'], txt['te'], args, 'A')
    print()
    results['exp_a'] = {n: show(n, y['te'], p) for n, (_, p, _) in models.items()}
    best_name = max(results['exp_a'], key=lambda k: results['exp_a'][k]['qwk'])
    best_model = models[best_name][0]
    print(f"    -> 채택: {best_name}")

    # --- EXP-B: leave-one-source-out ---
    rule("STEP 4. EXP-B  cross-source 전이 (LOSO, 선형만 — 비용 대비 스크리닝)")
    print("    source 교란을 제거한 진짜 일반화. 한 소스로 배워 다른 소스를 맞힌다.")
    print()
    loso = {}
    srcs = [s for s in meas['source'].unique() if (meas['source'] == s).sum() >= 500]
    for src in srcs:
        tr_m = meas['source'] != src
        te_m = meas['source'] == src
        a = argparse.Namespace(**{**vars(args), 'skip_xgb': True})
        sub, _ = fit_models(conv(meas.loc[tr_m, 'raw_text'], meas.loc[tr_m, 'source']),
                            (meas.loc[tr_m, 'urgency_score'] - 1).to_numpy(),
                            txt['va'], y['va'],
                            conv(meas.loc[te_m, 'raw_text'], meas.loc[te_m, 'source']),
                            a, f'B/{src}')
        loso[src] = show(f'-> {src} ({int(te_m.sum()):,})',
                         (meas.loc[te_m, 'urgency_score'] - 1).to_numpy(),
                         sub['linear_svc'][1])
    results['exp_b_loso'] = loso

    # --- EXP-C: 폴백 대체 검증 (핵심) ---
    rule("STEP 5. EXP-C  어휘 폴백 대체 검증  ★핵심")
    print("    measurable hold-out을 '메타데이터 없는 상태'로 두고 두 방법을 비교한다.")
    print("    정답 = 규칙 라벨(메타데이터를 읽고 만든 것). 같은 정답 위에서 비교.")
    print()
    print("    ⚠️ 이 비교는 폴백에 불리한 쪽으로 편향돼 있다. 폴백은 3점에서 시작하도록")
    print("       설계됐는데 measurable 라벨은 1·2점이 70%다. 그 몫을 분리하려고")
    print("       '항상 3점' 상수 baseline을 같이 놓는다. 폴백이 상수와 비슷하다면")
    print("       폴백의 어휘 가산이 실질적으로 아무 일도 안 한다는 뜻이다.")
    print()
    fb = np.array([score_by_vocabulary(t)[0] - 1 for t in te['raw_text']])
    pred_model = models[best_name][1]
    results['exp_c'] = {
        'always_3': show('상수 baseline (항상 3점)', y['te'], np.full(len(y['te']), 2)),
        'vocab_fallback': show('현행 어휘 폴백', y['te'], fb),
        f'model_{best_name}': show(f'모델 ({best_name})', y['te'], pred_model),
    }
    dq = (results['exp_c'][f'model_{best_name}']['qwk']
          - results['exp_c']['vocab_fallback']['qwk'])
    dm = (results['exp_c'][f'model_{best_name}']['mae']
          - results['exp_c']['vocab_fallback']['mae'])
    print()
    print(f"    차이: QWK {dq:+.4f} / MAE {dm:+.4f}")
    print("    QWK가 오르고 MAE가 내려가면 모델이 폴백보다 낫다는 직접 증거다.")

    # --- EXP-D: 실제 unmeasurable 적용 ---
    rule("STEP 6. EXP-D  unmeasurable 적용 (현행 라벨 교체 후보)")
    txt_u = conv(unmeas['raw_text'], unmeas['source'])
    pred_u = best_model.predict(vec.transform(txt_u)) + 1
    cur = unmeas['urgency_score'].to_numpy()
    cmp = pd.DataFrame({
        '현행(폴백)': pd.Series(cur).value_counts(normalize=True).mul(100),
        '모델 예측': pd.Series(pred_u).value_counts(normalize=True).mul(100),
    }).sort_index().round(1).fillna(0.0)
    print("  라벨 분포 비교 (%):")
    print(cmp.to_string())
    agree = float((cur == pred_u).mean())
    print()
    print(f"  현행 폴백과의 일치율 {agree*100:.1f}%  "
          f"(QWK {evaluate(cur - 1, pred_u - 1)['qwk']:.4f})")
    print("  ※ 정답이 없으므로 이건 '얼마나 다른가'이지 '누가 맞는가'가 아니다.")
    print("     누가 맞는가의 근거는 EXP-C에, 그 근거를 믿어도 되는지는 EXP-B에 있다.")
    print()
    print("  ⚠️ 라벨 분포 이동(label shift)에 주의할 것. 모델은 measurable 분포")
    print("     (1·2점 70%)를 학습했으므로 그것을 unmeasurable에 그대로 투사한다.")
    print("     예측이 저점에 쏠리는 것은 학습 분포의 반영이지 unmeasurable 공고가")
    print("     실제로 덜 급하다는 증거가 아니다. EXP-B 전이 성능이 낮다면")
    print("     이 적용은 신뢰할 수 없다.")
    results['exp_d'] = {'agreement': agree,
                        'dist_current': cmp['현행(폴백)'].to_dict(),
                        'dist_model': cmp['모델 예측'].to_dict()}

    # --- 저장 ---
    rule("STEP 7. 저장")
    out = SMOKE_DIR if (args.sample or args.skip_xgb) else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    if out is SMOKE_DIR:
        print(f"  [!] 축소 실행이므로 {out.name}/ 에 저장")
    joblib.dump(best_model, out / "transfer_model.joblib")
    joblib.dump(vec, out / "transfer_tfidf.joblib")
    unmeas_out = unmeas[['source', 'job_id', 'urgency_score']].copy()
    unmeas_out['urgency_score_model'] = pred_u
    unmeas_out.to_json(out / "unmeasurable_predictions.json",
                       orient='records', force_ascii=False, indent=1)
    (out / "transfer_meta.json").write_text(json.dumps({
        'design': 'measurable 학습 -> unmeasurable 적용 (개선 6)',
        'variant': VARIANT,
        'variant_rationale': '적용 대상에 메타데이터·템플릿이 없어 학습 텍스트도 맞춘다',
        'dedup': dedup_stats,
        'split': {'train': len(tr), 'val': len(va), 'test': len(te),
                  'unmeasurable': len(unmeas)},
        'selected_model': best_name,
        'results': results,
    }, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    for p in sorted(out.iterdir()):
        print(f"  저장: {p.name}  ({p.stat().st_size/1e6:.2f} MB)")
    print()
    print(f"  총 소요 {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
