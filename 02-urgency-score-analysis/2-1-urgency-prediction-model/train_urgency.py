"""
train_urgency.py

새 라벨(`urgency_rule.py`)로 학습하고, 직전 라벨 대비 무엇이 나아졌는지를
같은 파이프라인 위에서 측정한다. 라운드는 `--v4`/`--v5` 로 고른다(기본값 v3).
현행 정본은 **v5**다.

※ 파일명에 버전을 넣지 않는다. v3 전용이던 시절 이름이 `train_urgency_v3.py`
   였는데 v4까지 이 스크립트가 학습하게 되면서 이름이 내용과 어긋났다.

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
  python train_urgency.py                # 전체 (약 15분) — 정본 models_v3/
  python train_urgency.py --skip-xgb     # 선형만 (수십 초, 스모크)
  python train_urgency.py --sample 8000  # 축소
  python train_urgency.py --struct       # [실험] 구조화 피처 추가
  python train_urgency.py --struct-only  # [실험] 전이를 구조화 피처만으로
  python train_urgency.py --v4           # v4 라벨 학습 + v3 대비 비교 -> models_v4/
  python train_urgency.py --v5           # v5 라벨 학습 + v4 대비 비교 -> models_v5/ (현행 정본)

---------------------------------------------------------------------------
v4 라운드 (--v4)
---------------------------------------------------------------------------
스크립트를 복제하지 않고 라벨 버전만 파라미터로 뺐다(ROUNDS). 파이프라인이
두 벌로 갈라지면 통제 조건이 조용히 어긋나기 때문이다. 기본값은 v3 라운드
그대로라 기존 재현 절차는 바뀌지 않는다.

v4가 고친 것(`urgency_rule.py` [수정 3][수정 4]):
  [수정 3] '채용 시 마감'이 rolling(+10)과 조기마감(+12)에 이중 계상되던 것
  [수정 4] 어휘 폴백 재보정 (3 + 가산 -> 2 + 가산 x 0.25)

⚠️ [수정 4]는 unmeasurable 행에만 영향을 준다. 학습은 measurable만 쓰므로
   EXP-A/B의 변화는 전부 [수정 3]에서 온다. EXP-C만 [수정 4]를 반영한다.

⚠️ 라벨 파일은 얼려도 규칙 코드는 얼지 않는다. EXP-C는 `urgency_rule`에서
   score_by_vocabulary를 **실행 시점에** 임포트하므로, 기본(v3) 라운드로
   돌려도 폴백 수치는 v4 재보정이 적용된 값이 나온다. 2-1 README에 적힌
   v3 당시의 폴백 수치(MAE 2.0093)를 재현하려면 FALLBACK_BASE=3,
   FALLBACK_SCALE=1.0으로 되돌려야 한다. EXP-A/B는 라벨 파일만 쓰므로
   영향받지 않는다.

---------------------------------------------------------------------------
--struct 를 기본값으로 켜지 않는 이유
---------------------------------------------------------------------------
구조화 피처 11개는 전부 mask_text()가 지우는 구간에서 나온다
(`window_days` <- `접수기간…`, `is_urgent_word` <- `급구|긴급채용`, ...).
붙이는 순간 masked+clean 통제가 무효가 되고, 전이 QWK가 0.1309 -> 0.6757로
뛰지만 그건 통제가 걷어냈던 누출이 되돌아온 것이다. 규칙을 선형모델로
재현한 것이지 일반화가 아니다(2-1/02 README의 ablation 참조).

그래서 실험 플래그로만 두고, 켜면 결과를 models_<라운드>_struct/ 에 저장한다
(--v5 면 models_v5_struct/). 정본 디렉터리는 건드리지 않는다.
앱이 로드하는 정본 models_v5/ 는 통제가 걸린 모델로 유지된다.
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

from urgency_rule import (is_measurable, score_by_vocabulary, # noqa: E402
                            structured_features, STRUCT_FEATURE_NAMES)
from scipy.sparse import hstack, csr_matrix
    
from train_urgency_baseline import (N_CLASSES, RANDOM_STATE,  # noqa: E402
                                    Timer, build_tfidf, dedup_and_group,
                                    evaluate, make_transform, rule)
from common.hf_data import (MASTER_V2, MASTER_V3,  # noqa: E402
                            MASTER_V4, MASTER_V5, fetch as hf_fetch)

# 어떤 라벨 버전을 학습하고 무엇과 비교할지. --v4 / --v5 로 고른다.
#   new = 학습·저장 대상,  old = EXP-B에서 나란히 놓을 직전 버전
# 경로가 아니라 **파일명**을 들고 있다가 common.hf_data.fetch()로 연다.
# fetch()가 "data/에 있으면 그것, 없으면 data/로 다운로드, 둘 다 안 되면
# 만드는 명령 안내"를 한 곳에서 처리한다. 라벨 파일은 --write 산출물이지만
# v2~v5가 전부 허깅페이스에도 올라가 있으므로(v5는 2026-09-21), 로컬에 없어도
# 그냥 받아진다.
ROUNDS = {
    'v3': {'new': ('v3', MASTER_V3), 'old': ('v2', MASTER_V2), 'out': 'models_v3'},
    'v4': {'new': ('v4', MASTER_V4), 'old': ('v3', MASTER_V3), 'out': 'models_v4'},
    'v5': {'new': ('v5', MASTER_V5), 'old': ('v4', MASTER_V4), 'out': 'models_v5'},
}

VARIANT = 'masked+clean'
TEST_FOLDS = 5


# ---------------------------------------------------------------------------
def load_with_both_labels(round_cfg):
    """학습할 라벨(new)을 읽고 같은 행에 직전 버전 라벨(old)을 붙인다.

    ⚠️ (source, job_id)로 조인하면 안 된다. 이 데이터셋에는 같은 키가
       777종(1,554행) 중복돼 있어 dict가 뒤엣것만 남기고, 서로 다른 공고끼리
       짝지어진다. v4 작업 중 규칙상 불가능한 등급 전이가 나와서 발견했다
       (`urgency_rule.load_labels()` 주석 참조).

       모든 라벨 파일이 master_merged.json을 같은 순서로 훑어 만들어지므로
       위치로 맞추는 것이 정확하다. 정렬이 어긋나면 멈춘다 — 조용히 잘못된
       비교를 내놓는 것보다 낫다."""
    new_tag, new_file = round_cfg['new']
    old_tag, old_file = round_cfg['old']
    rule(f"STEP 1. 데이터 로드 ({new_tag} 라벨 + {old_tag} 라벨 나란히)")
    new_path = hf_fetch(new_file)
    print(f"  {new_tag}: {new_path}")
    with open(new_path, encoding='utf-8') as f:
        rows_new = json.load(f)
    df = pd.DataFrame(rows_new)
    df['raw_text'] = df['raw_text'].fillna('').astype(str)
    df['source'] = df['source'].fillna('unknown').astype(str)
    df = df.rename(columns={'urgency_score': 'y_new'})
    df['y_new'] = df['y_new'].astype(int)

    op = hf_fetch(old_file)
    print(f"  {old_tag}: {op}")
    with open(op, encoding='utf-8') as f:
        rows_old = json.load(f)
    if len(rows_old) != len(rows_new):
        raise SystemExit(f"  [!] {old_tag}({len(rows_old):,})와 "
                         f"{new_tag}({len(rows_new):,})의 행 수가 다르다 — 비교 불가")
    for a, b in zip(rows_old, rows_new):
        if (a.get('source'), a.get('job_id')) != (b.get('source'), b.get('job_id')):
            raise SystemExit(f"  [!] {old_tag}와 {new_tag}의 행 정렬이 다르다 — 비교 불가")
    print(f"  {old_tag} 라벨 정렬 확인 (위치 기준 {len(rows_old):,}행 일치)")

    df['y_old'] = [r['urgency_score'] for r in rows_old]
    df['y_old'] = df['y_old'].astype(int)
    df = df.reset_index(drop=True)

    df['measurable'] = df['raw_text'].map(is_measurable)
    m = df[df.measurable].reset_index(drop=True)
    u = df[~df.measurable].reset_index(drop=True)
    print(f"  전체 {len(df):,}행  /  measurable {len(m):,} ({len(m)/len(df)*100:.1f}%) "
          f"/ unmeasurable {len(u):,}")
    print(f"  measurable source: {m['source'].value_counts().to_dict()}")

    print()
    print("  measurable 라벨 분포 (%) — 소스별로 같은 모양이어야 '같은 타깃'이다")
    for col, tag in [('y_old', old_tag), ('y_new', new_tag)]:
        print(f"    [{tag}]")
        for s in ['jobkorea', 'saramin']:
            sub = m[m['source'] == s]
            if not len(sub):
                continue
            vc = sub[col].value_counts(normalize=True).mul(100)
            line = '  '.join(f"{lv}:{vc.get(lv, 0.0):5.1f}%" for lv in range(1, 6))
            top2 = vc.get(4, 0.0) + vc.get(5, 0.0)
            print(f"      {s:<9} {line}   상위등급(4+5) {top2:5.1f}%")
    print()
    print("  -> v2는 jobkorea와 saramin의 상위등급 비율이 몇 배씩 차이 난다.")
    print("     같은 개념을 쟀다면 나올 수 없는 격차이고, 이것이 전이 실패의 원인이다.")
    print("     (v4에서는 이 격차가 다시 벌어진다 — 이중 계상이 양쪽을 함께")
    print("      부풀리고 있었기 때문이다. 02 README '정합성 지표는 나빠진다' 참조)")
    return m, u


def make_split(meas):
    """중복 제거 + group-aware 3분할. 라벨과 무관하게 한 번만 만든다.

    v2/v3 비교가 공정하려면 split이 같아야 한다. stratify 키는 v3 라벨로
    잡는다(어차피 최종 모델은 v3)."""
    meas, groups, dedup_stats = dedup_and_group(meas)
    rule("STEP 2. Split (measurable 내부, group-aware)")
    strat = meas['source'] + "__" + meas['y_new'].astype(str)
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

def build_struct_matrix(df):
    """structured_features를 이용해 sparse matrix를 만든다.

    ⚠️ 기본 학습 경로에서는 쓰지 않는다. `--struct` / `--struct-only`를
    줬을 때만 붙는다. 이유는 main()의 STEP 3 주석 참조."""
    feats = [structured_features(t, s) for t, s in zip(df['raw_text'], df['source'])]
    arr = np.array([[f[k] for k in STRUCT_FEATURE_NAMES] for f in feats], dtype=float)
    return csr_matrix(arr)


def stack(X, struct):
    """구조화 피처가 있으면 옆에 붙이고, 없으면 그대로 둔다."""
    return X if struct is None else hstack([X, struct]).tocsr()

def fit_models(txt_tr, ytr, txt_va, yva, args, tag, 
                struct_tr=None, struct_va=None, skip_xgb=None, use_text=True):
    """(이름 -> 모델) 과 vectorizer. 예측은 호출부에서 필요한 만큼 한다.

    use_text=False면 TF-IDF를 아예 빼고 구조화 피처만 쓴다.
    (구조화 피처가 순전히 규칙 재료를 재현하는 것인지 확인하는 ablation용)"""
    skip = args.skip_xgb if skip_xgb is None else skip_xgb

    if use_text:
        vec, Xtr, (Xva,), vs = build_tfidf(txt_tr, [txt_va], args.max_features)
        Xtr, Xva = stack(Xtr, struct_tr), stack(Xva, struct_va)
    else:
        vec = None
        Xtr, Xva = struct_tr, struct_va
        vs = 0.0          # ← 추가: TF-IDF 안 만드니 벡터화 시간은 0

    classes = np.unique(ytr)
    wmap = dict(zip(classes, compute_class_weight('balanced', classes=classes, y=ytr)))
    sw = np.array([wmap[y] for y in ytr])

    out = {}
    with Timer() as t:
        svc = LinearSVC(class_weight='balanced', C=1.0, max_iter=5000,
                        random_state=RANDOM_STATE).fit(Xtr, ytr)
    out['linear_svc'] = svc
    
    feat_label = ("TF-IDF+구조화" if struct_tr is not None else "TF-IDF") \
        if use_text else "구조화"
    print(f"    [{tag}] {feat_label} {Xtr.shape[1]:,}f ({vs:.1f}s) · linear_svc {t.cpu:.1f}s")

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
    ap.add_argument('--struct', action='store_true',
                    help='[실험] TF-IDF 옆에 구조화 피처 11개를 붙인다. '
                         '기본값 꺼짐 — 켜면 models_<라운드>_struct/ 에 저장된다')
    ap.add_argument('--struct-only', action='store_true',
                    help='[실험] EXP-B(전이)를 TF-IDF 없이 구조화 피처 11개만으로 '
                         '돌리는 ablation. --struct 를 함께 켠 것으로 취급한다')
    ap.add_argument('--v4', action='store_true',
                    help='v4 라벨로 학습하고 v3와 비교한다 (기본값은 v3 vs v2). '
                         '저장 위치는 models_v4/')
    ap.add_argument('--v5', action='store_true',
                    help='v5 라벨로 학습하고 v4와 비교한다. 저장 위치는 models_v5/. '
                         'v5는 v4와 38행만 다르므로(라벨의 0.09%%) 성능 차이는 '
                         '학습 변동 수준일 것으로 예상된다 — urgency_rule [수정 5]')
    args = ap.parse_args()
    use_struct = args.struct or args.struct_only
    cfg = ROUNDS['v5' if args.v5 else ('v4' if args.v4 else 'v3')]
    new_tag, old_tag = cfg['new'][0], cfg['old'][0]
    t_start = time.time()

    meas, unmeas = load_with_both_labels(cfg)
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
    rule(f"STEP 3. EXP-A  in-domain ({new_tag} 라벨) — 모델 선택은 validation으로")
    print("  ⚠️ v2 스크립트는 여기서 test QWK로 모델을 골랐다(test contamination).")
    print("     v3은 validation으로 고르고, test는 마지막에 한 번만 본다.")
    print()
    y = {k: (d['y_new'] - 1).to_numpy() for k, d in [('tr', tr), ('va', va), ('te', te)]}

    # 구조화 피처는 기본값에서 끈다(--struct 로만 켠다).
    # 11개 전부가 mask_text()가 지우는 구간에서 나오는 값이라, 붙이는 순간
    # masked+clean 통제(2-1 README: Macro F1 -0.1301을 치르고 세운 것)가
    # 무효가 된다. 실험은 아래 EXP-B에서만 의미가 있고, 여기서 켜면 앱이
    # 쓰는 정본 모델이 통제 없는 모델로 바뀐다.
    if use_struct:
        struct_tr = build_struct_matrix(tr)
        struct_va = build_struct_matrix(va)
        struct_te = build_struct_matrix(te)
        print(f"  [!] --struct: 구조화 피처 {len(STRUCT_FEATURE_NAMES)}개 사용 "
              f"(라벨 누출 있음 · 저장 위치 {cfg['out']}_struct/)")
    else:
        struct_tr = struct_va = struct_te = None

    models, vec = fit_models(txt['tr'], y['tr'], txt['va'], y['va'], args, 'A',
                             struct_tr=struct_tr, struct_va=struct_va)
    Xva = stack(vec.transform(txt['va']), struct_va)
    Xte = stack(vec.transform(txt['te']), struct_te)

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
    rule(f"STEP 4. EXP-B  cross-source 전이 — {old_tag} 라벨 vs {new_tag} 라벨  ★핵심")
    print("  한 소스로 배워 다른 소스를 맞힌다. source 교란이 제거된 진짜 일반화다.")
    print("  같은 행 · 같은 텍스트 · 같은 모델(LinearSVC)이고 라벨만 다르다.")
    print(f"  라벨 수정이 옳았다면 {new_tag}의 QWK가 {old_tag}보다 높아야 한다.")
    print()
    loso = {}
    srcs = [s for s in meas['source'].unique() if (meas['source'] == s).sum() >= 500]

    tag_of = {'y_old': old_tag, 'y_new': new_tag}
    for label_col in ['y_old', 'y_new']:
        loso[label_col] = {}
        print(f"  [{tag_of[label_col]} 라벨]")
        for src in srcs:
            tr_m, te_m = meas['source'] != src, meas['source'] == src

            if use_struct:
                struct_tr_sub = build_struct_matrix(meas.loc[tr_m])
                struct_te_sub = build_struct_matrix(meas.loc[te_m])
            else:
                struct_tr_sub = struct_te_sub = None

            sub, sub_vec = fit_models(
                conv(meas.loc[tr_m, 'raw_text'], meas.loc[tr_m, 'source']),
                (meas.loc[tr_m, label_col] - 1).to_numpy(),
                txt['va'], (va[label_col] - 1).to_numpy(),
                args, f'B/{tag_of[label_col]}/{src}',
                struct_tr=struct_tr_sub, struct_va=struct_va,
                skip_xgb=True,
                use_text=not args.struct_only)

            if args.struct_only:
                Xte_sub = struct_te_sub
            else:
                Xte_sub = stack(sub_vec.transform(
                    conv(meas.loc[te_m, 'raw_text'], meas.loc[te_m, 'source'])),
                    struct_te_sub)

            pred = sub['linear_svc'].predict(Xte_sub)
            loso[label_col][src] = show(
                f'-> {src} ({int(te_m.sum()):,}행)',
                (meas.loc[te_m, label_col] - 1).to_numpy(), pred, '      ')
        print()
        
    results['exp_b_loso'] = loso

    print("  === 전이 QWK 요약 ===")
    print(f"    {'평가 대상':<14}{old_tag + ' 라벨':>10}{new_tag + ' 라벨':>10}{'변화':>10}")
    for src in srcs:
        a, b = loso['y_old'][src]['qwk'], loso['y_new'][src]['qwk']
        print(f"    {src:<14}{a:>10.4f}{b:>10.4f}{b - a:>+10.4f}")
    mv2 = float(np.mean([loso['y_old'][s]['qwk'] for s in srcs]))
    mv3 = float(np.mean([loso['y_new'][s]['qwk'] for s in srcs]))
    print(f"    {'평균':<14}{mv2:>10.4f}{mv3:>10.4f}{mv3 - mv2:>+10.4f}")
    results['exp_b_summary'] = {f'mean_qwk_{old_tag}': mv2, f'mean_qwk_{new_tag}': mv3}

    # -----------------------------------------------------------------------
    rule(f"STEP 5. EXP-C  어휘 폴백 검증 ({new_tag} 라벨)")
    print("  measurable hold-out을 '메타데이터 없는 상태'로 두고 같은 정답 위에서 비교한다.")
    # 상수는 train 라벨 중앙값을 쓴다. v2는 '항상 3점'과 비교했는데 그건 MAE를
    # 최소화하는 상수가 아니라 모델에 유리한 기준이었다(v3에서 고친 것).
    # 여기도 같은 규약을 적용한다 — 폴백을 후하게 봐줄 이유가 없다.
    const = int(np.median(y['tr']))
    print(f"  상수 베이스라인 = train 라벨 중앙값 ({const + 1}점)")
    print()
    fb = np.array([score_by_vocabulary(t)[0] - 1 for t in te['raw_text']])
    results['exp_c'] = {
        'constant': show(f'상수 baseline ({const + 1}점)', y['te'],
                         np.full(len(y['te']), const)),
        'vocab_fallback': show(f'어휘 폴백 ({new_tag})', y['te'], fb),
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
          f"{results['exp_c']['constant']['qwk']:.4f}")
    print(f"    모델 MAE {final['mae']:.4f}  vs  상수 "
          f"{results['exp_c']['constant']['mae']:.4f}")

    # -----------------------------------------------------------------------
    rule("STEP 7. 저장")
    base_out = HERE / cfg['out']
    if args.sample or args.skip_xgb:
        out = HERE / f"{cfg['out']}_smoke"
        note = "축소 실행이므로"
    elif use_struct:
        out = HERE / f"{cfg['out']}_struct"
        note = f"구조화 피처 실험이므로 (정본 {base_out.name}/ 은 건드리지 않는다)"
    else:
        out, note = base_out, None
    out.mkdir(parents=True, exist_ok=True)
    if note:
        print(f"  [!] {note} {out.name}/ 에 저장")
    joblib.dump(best, out / "urgency_model.joblib")
    joblib.dump(vec, out / "urgency_tfidf.joblib")
    (out / "model_meta.json").write_text(json.dumps({
        'rule_version': new_tag,
        'data': cfg['new'][1],
        'compared_against': old_tag,
        'variant': VARIANT,
        # 추론 쪽(urgency_model.py)이 같은 피처 행렬을 다시 만들려면 이 목록이
        # 필요하다. null 이면 TF-IDF만 쓴 정본 모델이라는 뜻.
        'struct_features': STRUCT_FEATURE_NAMES if use_struct else None,
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
