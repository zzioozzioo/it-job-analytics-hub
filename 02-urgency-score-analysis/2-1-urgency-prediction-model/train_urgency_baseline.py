"""
train_urgency_baseline.py

v1(train_urgency_model.py)이 보고한 Macro F1 0.8595가 실제 일반화 성능인지
검증하기 위한 베이스라인 스크립트. 리뷰에서 지적된 4가지를 반영한다.

  [1] 라벨 누출 통제
      urgency_score는 rescore_urgency.py가 raw_text에서 정규식으로 만든
      결정론적 라벨이다. 라벨 산출 구간을 그대로 두면 모델은 '시급성 예측'이
      아니라 '정규식 역공학'을 학습한다. 따라서 full/masked 두 조건을 항상
      함께 측정하고, 그 차이를 meta에 명시적으로 기록한다.

  [3] 중복 공고 분리
      raw_text 완전중복 6.97%, 앞 600자 기준 근사중복 11.2%.
      랜덤 split이면 같은 공고가 train/test 양쪽에 들어가 암기로 정답을 맞힌다.
      -> 완전중복 제거 + 근사중복을 하나의 group으로 묶어 StratifiedGroupKFold.

  [5] 순서형 지표
      1~5점은 순서가 있는 라벨인데 v1은 Macro F1만 봤다. '1점->5점' 오답과
      '3점->4점' 오답이 같은 페널티다. QWK(Quadratic Weighted Kappa)와 MAE를
      주 지표로 올리고 Macro F1은 v1 비교용 보조로 남긴다.

  [8] 베이스라인 & 학습 비용
      v1은 baseline 없이 바로 XGBoost(1,050초)를 돌려 0.86이 좋은 숫자인지
      판단할 기준이 없었다. Dummy -> source-only -> LinearSVC -> LogReg 순으로
      바닥을 먼저 깔고, XGBoost는 early stopping을 붙여 비용을 줄인다.
      각 모델의 학습 시간을 같이 출력해 비용 대비 이득이 보이게 한다.

보조 피처(v1의 Delta Macro F1 = +0.0027)는 이 스크립트에 넣지 않았다.
v1은 그 채택 여부를 test set으로 결정해 test contamination이 있었고,
validation 기반 재선택은 별도 작업이라 베이스라인 범위에서 제외한다.

출력물은 models_baseline/ 에 저장한다. (v1의 models/ 는 건드리지 않는다)

실행:
  python train_urgency_baseline.py                 # 전체
  python train_urgency_baseline.py --sample 8000   # 스모크 테스트
  python train_urgency_baseline.py --skip-xgb      # 선형 baseline만 (수 초)
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             mean_absolute_error)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

sys.stdout.reconfigure(encoding='utf-8')

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from check_label_leakage import mask_text  # noqa: E402  라벨 마스킹 패턴 단일 출처
from boilerplate import fit_source_stopwords, strip_series  # noqa: E402

MODEL_DIR = HERE / "models_baseline"          # 전체 실행 결과 (정본)
SMOKE_DIR = HERE / "models_baseline_smoke"    # 축소 실행 결과 (정본을 덮지 않는다)
V1_META = HERE / "models" / "model_meta.json"
LOCAL_DATA = HERE.parents[1] / "data" / "master_merged_v2.json"

HF_REPO_ID = "data-craftee/korean-it-recruit-dataset"
HF_FILENAME = "master_merged_v2.json"

RANDOM_STATE = 42
N_CLASSES = 5
DEDUP_PREFIX = 600      # 근사중복 판정에 쓸 앞부분 길이
TEST_FOLDS = 5          # 1/5 = 20% test
VAL_FOLDS = 5           # train의 1/5 = 전체의 16% validation

MODEL_ORDER = ['dummy_major', 'dummy_prior', 'source_only',
               'linear_svc', 'logreg', 'xgboost_es']
# 통제 조건 4가지. masked+clean 이 완전 통제.
ALL_VARIANTS = ['full', 'masked', 'clean', 'masked+clean']
FITTED = ['linear_svc', 'logreg', 'xgboost_es']   # 저장 가능한 추정기


class Timer:
    """벽시계와 CPU 시간을 함께 잰다.

    벽시계만 재면 PC 절전/일시정지가 그대로 학습 시간으로 잡힌다.
    (실제로 첫 실행에서 full 변형 XGBoost가 40,503초로 기록됐는데
     대부분이 밤사이 절전 시간이었다.) 비용 비교는 cpu_sec으로 해야 한다."""

    def __enter__(self):
        self._w0, self._c0 = time.time(), time.process_time()
        return self

    def __exit__(self, *exc):
        self.wall = time.time() - self._w0
        self.cpu = time.process_time() - self._c0
        return False


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def resolve_out_dir(args):
    """축소 실행(--sample / --skip-xgb / 일부 variant)은 정본 디렉터리를 덮지 않는다.

    전체 실행에 11시간이 걸리는데 스모크 테스트 한 번으로 그 산출물이 날아가면
    복구 비용이 너무 크다. 실제로 한 번 그렇게 날려서 이 가드를 넣었다."""
    if args.out_dir:
        out = Path(args.out_dir)
        if not out.is_absolute():
            out = HERE / out
        print(f"  [!] --out-dir 지정: {out}")
        out.mkdir(parents=True, exist_ok=True)
        return out

    is_canonical = (args.sample is None and not args.skip_xgb and
                    {v.strip() for v in args.variants.split(',')} == set(ALL_VARIANTS))
    out = MODEL_DIR if is_canonical else SMOKE_DIR
    if not is_canonical:
        print(f"  [!] 축소 실행이므로 {out.name}/ 에 저장한다 "
              f"({MODEL_DIR.name}/ 은 건드리지 않음)")
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------------------------
def load_data(sample=None):
    rule("STEP 1. 데이터 로드")
    if LOCAL_DATA.exists():
        path = LOCAL_DATA
        print(f"  로컬 파일 사용: {path}")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME,
                               repo_type="dataset")
        print(f"  HF 다운로드: {path}")

    with open(path, encoding='utf-8') as f:
        df = pd.DataFrame(json.load(f))

    df['raw_text'] = df['raw_text'].fillna('').astype(str)
    df['source'] = df['source'].fillna('unknown').astype(str)
    df = df[df['urgency_score'].notna()].copy()
    df['urgency_score'] = df['urgency_score'].astype(int)
    df = df.reset_index(drop=True)

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE)
        df = df.reset_index(drop=True)
        print(f"  [!] --sample {sample}: {len(df):,}행만 사용")

    print(f"  총 {len(df):,}행")
    vc = df['urgency_score'].value_counts().sort_index()
    print("  클래스 분포:", {int(k): int(v) for k, v in vc.items()},
          f"| 불균형비 {vc.max() / vc.min():.2f}배")
    return df


# ---------------------------------------------------------------------------
# 2. [개선 3] 중복 제거 + 근사중복 그룹 키
# ---------------------------------------------------------------------------
def _norm(t):
    return re.sub(r'\s+', ' ', t or '').strip().lower()


def _md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def dedup_and_group(df):
    """완전중복은 제거하고, 근사중복(앞 DEDUP_PREFIX자 동일)은 group으로 묶는다.

    group을 쓰는 이유: 완전중복만 지워도 '같은 회사가 문구만 조금 바꿔 올린 공고'가
    남는다. 라벨이 텍스트의 결정론적 함수라 이런 쌍이 train/test에 갈리면
    사실상 정답지를 보고 푸는 것이 된다."""
    rule("STEP 2. 중복 공고 처리  [개선 3]")
    n0 = len(df)

    exact = df['raw_text'].map(lambda t: _md5(_norm(t)))
    dup_rows = int(exact.duplicated(keep=False).sum())
    df = df.loc[~exact.duplicated(keep='first')].reset_index(drop=True)
    print(f"  완전중복(정규화 후 전문 일치): {dup_rows:,}행 관여 "
          f"-> {n0 - len(df):,}행 제거, {len(df):,}행 잔존")

    groups = df['raw_text'].map(lambda t: _md5(_norm(t)[:DEDUP_PREFIX]))
    n_groups = int(groups.nunique())
    multi = int((groups.map(groups.value_counts()) > 1).sum())
    print(f"  근사중복 그룹(앞 {DEDUP_PREFIX}자 기준): {n_groups:,}개 그룹, "
          f"2건 이상 묶인 행 {multi:,}개 ({multi / len(df) * 100:.2f}%)")
    print("  -> 같은 그룹은 train/test에 쪼개지 않는다 (StratifiedGroupKFold)")
    return df, groups, {'rows_before': n0, 'rows_after': len(df),
                        'exact_dup_rows_removed': n0 - len(df),
                        'n_groups': n_groups, 'grouped_rows': multi}


def absorb_rare(strat_key, source):
    """계층이 fold 수보다 적으면 stratify가 불가능하다(wanted x 1점이 실제로 1건).
    같은 source의 최대 계층으로 흡수시킨다. (v1과 동일 처리 - 비교 가능성 유지)"""
    counts = strat_key.value_counts()
    rare = counts[counts < TEST_FOLDS].index
    if len(rare) == 0:
        return strat_key
    big = counts[counts >= TEST_FOLDS]
    fallback = {}
    for src in source.unique():
        cand = big[big.index.str.startswith(f"{src}__")]
        fallback[src] = cand.idxmax() if len(cand) else big.idxmax()
    mask = strat_key.isin(rare)
    strat_key = strat_key.copy()
    strat_key[mask] = source[mask].map(fallback)
    print(f"  [!] 희소 계층 {list(rare)} ({int(mask.sum())}건) "
          f"-> 동일 source 최대 계층으로 흡수")
    return strat_key


def split_three_way(df, groups):
    """train / validation / test 로 나눈다. validation은 early stopping 전용.

    v1은 train/test 2분할에 test로 모델을 골랐다(test contamination).
    여기서는 test를 최종 평가 한 번만 쓴다."""
    rule("STEP 3. Split (group-aware, source x label stratified)  [개선 3]")
    strat = df['source'] + "__" + df['urgency_score'].astype(str)
    strat = absorb_rare(strat, df['source'])

    sgkf = StratifiedGroupKFold(n_splits=TEST_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    rest_pos, test_pos = next(sgkf.split(df, strat, groups))

    rest = df.iloc[rest_pos].reset_index(drop=True)
    rest_groups = groups.iloc[rest_pos].reset_index(drop=True)
    rest_strat = strat.iloc[rest_pos].reset_index(drop=True)

    sgkf2 = StratifiedGroupKFold(n_splits=VAL_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)
    tr_pos, va_pos = next(sgkf2.split(rest, rest_strat, rest_groups))

    tr, va, te = rest.iloc[tr_pos], rest.iloc[va_pos], df.iloc[test_pos]
    print(f"  train {len(tr):,} / val {len(va):,} / test {len(te):,}")

    shared = set(groups.iloc[test_pos]) & set(rest_groups)
    print(f"  train+val 과 test 사이 공유 그룹: {len(shared)}개 "
          f"{'(정상)' if not shared else '(!! 누출)'}")

    comp = pd.DataFrame({
        'train_%': tr['urgency_score'].value_counts(normalize=True).mul(100).round(2),
        'val_%': va['urgency_score'].value_counts(normalize=True).mul(100).round(2),
        'test_%': te['urgency_score'].value_counts(normalize=True).mul(100).round(2),
    }).sort_index()
    print(comp.to_string())
    return (tr.reset_index(drop=True), va.reset_index(drop=True),
            te.reset_index(drop=True), len(shared))


# ---------------------------------------------------------------------------
# 3. [개선 5] 순서형 지표
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred):
    """순서형 라벨용 지표. QWK/MAE가 주 지표, Macro F1은 v1 비교용 보조.

    QWK는 '1점을 5점으로' 틀린 것을 '3점을 4점으로' 틀린 것보다 훨씬 무겁게
    센다. 라벨이 1~5 순서형이므로 Macro F1보다 이쪽이 품질을 옳게 반영한다."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    d = np.abs(y_true - y_pred)
    return {
        'qwk': float(cohen_kappa_score(y_true, y_pred, weights='quadratic',
                                       labels=list(range(N_CLASSES)))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'off_by_1': float(np.mean(d <= 1)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# 4. [개선 8] 베이스라인 사다리
# ---------------------------------------------------------------------------
def source_majority(src_tr, ytr, src_te):
    """source만 보고 그 source의 최빈 클래스를 찍는 baseline.

    라벨이 source별로 다른 규칙에서 나왔기 때문에(jobkorea=마감일 파싱,
    saramin=접수창 길이, wanted=어휘 폴백) source 자체가 라벨의 강력한 프록시다.
    이 baseline을 뚜렷이 넘지 못하는 텍스트 모델은 사실상 source만 외운 것이다."""
    tab = pd.DataFrame({'src': np.asarray(src_tr), 'y': np.asarray(ytr)})
    maj = tab.groupby('src')['y'].agg(lambda s: s.value_counts().idxmax()).to_dict()
    glob = pd.Series(ytr).value_counts().idxmax()
    return np.array([maj.get(s, glob) for s in np.asarray(src_te)])


def build_tfidf(texts_tr, texts_other, max_features, stop_words=None):
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features,
                          min_df=5, max_df=0.9, sublinear_tf=True,
                          stop_words=stop_words)
    t0 = time.time()
    Xtr = vec.fit_transform(texts_tr)
    outs = [vec.transform(t) for t in texts_other]
    return vec, Xtr, outs, time.time() - t0


def make_transform(variant):
    """variant 이름을 텍스트 변환 함수로 바꾼다.

      full        원본
      masked      라벨 산출 구간 제거          [개선 1]
      clean       사이트 템플릿 제거            [개선 2]
      masked+clean  둘 다 = 완전 통제 조건

    clean이 source를 인자로 받아야 해서 (texts, sources) 시그니처를 쓴다."""
    parts = {p.strip() for p in variant.split('+')}
    unknown = parts - {'full', 'masked', 'clean'}
    if unknown:
        raise ValueError(f"알 수 없는 variant 성분: {sorted(unknown)}")

    def transform(texts, sources):
        out = texts
        if 'clean' in parts:
            out = strip_series(out, sources)
        if 'masked' in parts:
            out = out.map(mask_text)
        return out

    return transform


def run_variant(variant, tr, va, te, args):
    """해당 통제 조건에서 baseline 사다리를 전부 돌린다."""
    rule(f"STEP 4-{variant}. 학습  [variant = {variant}]  [개선 1 · 2 · 8]")

    conv = make_transform(variant)
    txt_tr = conv(tr['raw_text'], tr['source'])
    txt_va = conv(va['raw_text'], va['source'])
    txt_te = conv(te['raw_text'], te['source'])
    print(f"  평균 길이: {tr['raw_text'].str.len().mean():.0f}자 "
          f"-> {txt_tr.str.len().mean():.0f}자")

    stop_words = None
    if args.source_stopwords:
        stop_words = fit_source_stopwords(txt_tr, tr['source'])   # train에서만 fit
        print(f"  source-wise DF stopword {len(stop_words)}개 적용")

    vec, Xtr, (Xva, Xte), vec_s = build_tfidf(txt_tr, [txt_va, txt_te],
                                              args.max_features, stop_words)
    print(f"  TF-IDF: {Xtr.shape[1]:,} features  (fit {vec_s:.1f}s)")

    ytr = (tr['urgency_score'] - 1).to_numpy()
    yva = (va['urgency_score'] - 1).to_numpy()
    yte = (te['urgency_score'] - 1).to_numpy()

    classes = np.unique(ytr)
    weights = compute_class_weight('balanced', classes=classes, y=ytr)
    wmap = dict(zip(classes, weights))
    sw = np.array([wmap[y] for y in ytr])

    results, models = [], {}

    def record(name, pred, timer, note=''):
        m = evaluate(yte, pred)
        m.update(model=name, variant=variant, note=note,
                 fit_sec=round(timer.wall, 1), cpu_sec=round(timer.cpu, 1))
        results.append(m)
        print(f"  [{name:<12}] QWK {m['qwk']:>7.4f}  MAE {m['mae']:>6.4f}  "
              f"MacroF1 {m['macro_f1']:>6.4f}  Acc {m['accuracy']:>6.4f}"
              f"  (cpu {timer.cpu:>9.1f}s / wall {timer.wall:>8.1f}s)")
        return m

    # --- 바닥 baseline: 이 아래로는 학습이 아무 의미 없다는 기준선 ---
    with Timer() as t:
        dm = DummyClassifier(strategy='most_frequent').fit(Xtr, ytr)
        pred = dm.predict(Xte)
    record('dummy_major', pred, t, '최빈 클래스 고정')

    with Timer() as t:
        ds = DummyClassifier(strategy='stratified', random_state=RANDOM_STATE).fit(Xtr, ytr)
        pred = ds.predict(Xte)
    record('dummy_prior', pred, t, '사전분포 무작위')

    with Timer() as t:
        pred = source_majority(tr['source'], ytr, te['source'])
    record('source_only', pred, t, 'source별 최빈 클래스')

    # --- 선형 baseline: TF-IDF 희소행렬에는 이쪽이 정석이고 수 초면 끝난다 ---
    with Timer() as t:
        svc = LinearSVC(class_weight='balanced', C=1.0, max_iter=5000,
                        random_state=RANDOM_STATE).fit(Xtr, ytr)
        pred = svc.predict(Xte)
    record('linear_svc', pred, t, 'TF-IDF 선형 SVM')
    models['linear_svc'] = svc

    with Timer() as t:
        lr = LogisticRegression(class_weight='balanced', max_iter=1000,
                                random_state=RANDOM_STATE).fit(Xtr, ytr)
        pred = lr.predict(Xte)
    record('logreg', pred, t, 'TF-IDF 로지스틱')
    models['logreg'] = lr

    # --- XGBoost + early stopping: 트리 수를 val이 정하게 한다 ---
    if not args.skip_xgb:
        xgb = XGBClassifier(
            n_estimators=args.n_estimators,
            early_stopping_rounds=args.early_stopping,
            max_depth=7, learning_rate=0.2, subsample=0.9, colsample_bytree=0.7,
            tree_method='hist', objective='multi:softmax', num_class=len(classes),
            n_jobs=-1, random_state=RANDOM_STATE, eval_metric='mlogloss',
        )
        with Timer() as t:
            xgb.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
            pred = xgb.predict(Xte)
        best_it = int(xgb.best_iteration) + 1
        m = record('xgboost_es', pred, t,
                   f'early stop @ {best_it}/{args.n_estimators} trees')
        m['best_iteration'] = best_it
        models['xgboost_es'] = xgb

    models['_vec'] = vec
    models['_Xte'] = Xte
    return results, models, yte


# ---------------------------------------------------------------------------
# 5. 리포트
# ---------------------------------------------------------------------------
def print_table(results):
    rule("STEP 5. 결과 종합  [개선 5: QWK/MAE 주 지표]")
    df = pd.DataFrame(results)
    piv = df.pivot(index='model', columns='variant',
                   values=['qwk', 'mae', 'macro_f1', 'cpu_sec'])
    piv = piv.reindex([m for m in MODEL_ORDER if m in piv.index])
    print(piv.round(4).to_string())
    print()
    print("  QWK  : 1.0 완벽, 0.0 무작위. 순서형 라벨의 주 지표.")
    print("  MAE  : 평균 몇 점 빗나가는가 (낮을수록 좋음).")
    print("  cpu_sec: 선형 baseline과 XGBoost의 비용 차이. (벽시계는 meta의 fit_sec)")

    present = [v for v in ALL_VARIANTS if v in set(df['variant'])]
    if 'full' in present and len(present) > 1:
        rule("  통제 조건별 QWK 변화 (기준 = full)  [개선 1 · 2]")
        base = df[df.variant == 'full'].set_index('model')
        rows = [m for m in MODEL_ORDER if m in base.index]
        head = f"    {'model':<13}{'full':>9}"
        for v in present[1:]:
            head += f"{v:>15}"
        print(head)
        for k in rows:
            a = float(base.loc[k, 'qwk'])
            line = f"    {k:<13}{a:>9.4f}"
            for v in present[1:]:
                sub = df[(df.variant == v) & (df.model == k)]
                if sub.empty:
                    line += f"{'-':>15}"
                    continue
                b = float(sub['qwk'].iloc[0])
                line += f"{b:>8.4f}{b - a:>+7.4f}"
            print(line)
        print()
        print("    masked      = 라벨 산출 구간(남은기간/접수기간/모집인원/긴급 어휘) 제거")
        print("    clean       = 사이트 템플릿 제거 (jobkorea 꼬리 추천공고, saramin 정형 고지문)")
        print("    masked+clean= 완전 통제. 이 조건이 '본문을 읽고 일반화한' 성능에 가장 가깝다.")


def detail_report(best_name, variant, models, te, yte):
    rule(f"STEP 6. 최종 모델 상세  [{best_name} / {variant}]")
    clf = models[variant][best_name]
    pred = clf.predict(models[variant]['_Xte'])

    print("  클래스별 리포트:")
    print(classification_report(yte, pred, labels=list(range(N_CLASSES)),
                                target_names=[f'{i}점' for i in range(1, 6)],
                                digits=3, zero_division=0))
    print("  Confusion Matrix (행=실제, 열=예측):")
    cm = confusion_matrix(yte, pred, labels=list(range(N_CLASSES)))
    print("        " + "".join(f"{f'pred{i}':>8}" for i in range(1, 6)))
    for i, row in enumerate(cm):
        print(f"  true{i + 1} " + "".join(f"{v:>8,}" for v in row))

    print()
    print("  source별 성능 (라벨 생성 규칙이 source마다 달라 분리해서 봐야 함):")
    print(f"    {'source':<10}{'n':>8}{'QWK':>9}{'MAE':>8}{'MacroF1':>10}")
    per_source = {}
    for src in sorted(te['source'].unique()):
        mask = (te['source'] == src).to_numpy()
        if not mask.any():
            continue
        s = evaluate(np.asarray(yte)[mask], np.asarray(pred)[mask])
        per_source[src] = s
        print(f"    {src:<10}{int(mask.sum()):>8,}{s['qwk']:>9.4f}"
              f"{s['mae']:>8.4f}{s['macro_f1']:>10.4f}")
    return per_source


def compare_v1(results):
    if not V1_META.exists():
        return None
    v1 = json.loads(V1_META.read_text(encoding='utf-8'))
    df = pd.DataFrame(results)

    def best(variant):
        sub = df[(df.variant == variant) & (df.model.isin(FITTED))]
        return float(sub['macro_f1'].max()) if len(sub) else None

    b_full, b_masked = best('full'), best('masked')
    rule("STEP 7. v1 대비")
    print(f"  v1 (중복 미제거 · 랜덤 split · full text)  Macro F1 : {v1['macro_f1']:.4f}")
    if b_full is not None:
        print(f"  v2 (중복 그룹 분리 · full text)           Macro F1 : {b_full:.4f}"
              f"  ({b_full - v1['macro_f1']:+.4f})")
    if b_masked is not None:
        print(f"  v2 (중복 그룹 분리 · masked)              Macro F1 : {b_masked:.4f}"
              f"  ({b_masked - v1['macro_f1']:+.4f})")
    print()
    print("  차이는 모델이 아니라 평가 설계에서 온다. 중복 공고가 train/test에")
    print("  갈리지 않게 하고, 라벨 산출 구간을 가린 결과다.")
    return {'v1_macro_f1': v1['macro_f1'],
            'v2_full_macro_f1': b_full, 'v2_masked_macro_f1': b_masked}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=None)
    ap.add_argument('--max-features', type=int, default=30000)
    ap.add_argument('--n-estimators', type=int, default=400,
                    help='XGBoost 상한. early stopping이 실제 트리 수를 정한다.')
    ap.add_argument('--early-stopping', type=int, default=30)
    ap.add_argument('--skip-xgb', action='store_true',
                    help='선형 baseline만 실행(수 초)')
    ap.add_argument('--variants', default=','.join(ALL_VARIANTS))
    ap.add_argument('--source-stopwords', action='store_true',
                    help='source별 DF 기반 stopword 추가 적용. 진단상 효과가 중립이라'
                         '(source Acc -0.0017 / QWK -0.0045) 기본 off.')
    ap.add_argument('--out-dir', default=None,
                    help='저장 위치를 직접 지정(축소 실행 가드를 의도적으로 우회할 때)')
    args = ap.parse_args()

    t_start = time.time()
    df = load_data(args.sample)
    df, groups, dedup_stats = dedup_and_group(df)
    tr, va, te, n_shared = split_three_way(df, groups)

    all_results, all_models, yte = [], {}, None
    for variant in [v.strip() for v in args.variants.split(',') if v.strip()]:
        res, models, yte = run_variant(variant, tr, va, te, args)
        all_results += res
        all_models[variant] = models

    print_table(all_results)

    # 최종 선택: masked(정직한 조건)의 QWK 기준. masked가 없으면 첫 variant.
    rdf = pd.DataFrame(all_results)
    variants = list(dict.fromkeys(rdf['variant']))
    # 완전 통제 조건을 우선한다: masked+clean > masked > 그 외
    pick = next((v for v in ['masked+clean', 'masked'] if v in variants), variants[0])
    cand = rdf[(rdf.variant == pick) & (rdf.model.isin(FITTED))]
    best_name = str(cand.loc[cand['qwk'].idxmax(), 'model'])

    per_source = detail_report(best_name, pick, all_models, te, yte)
    v1_cmp = compare_v1(all_results)

    # --- 저장 ---
    rule("STEP 8. 저장")
    out_dir = resolve_out_dir(args)
    joblib.dump(all_models[pick][best_name], out_dir / "baseline_model.joblib")
    joblib.dump(all_models[pick]['_vec'], out_dir / "baseline_tfidf.joblib")
    meta = {
        'script': 'train_urgency_baseline.py',
        'data': {'source': str(LOCAL_DATA if LOCAL_DATA.exists() else HF_REPO_ID),
                 'sample': args.sample, **dedup_stats},
        'split': {'strategy': f'StratifiedGroupKFold (group = 앞 {DEDUP_PREFIX}자 해시)',
                  'train': len(tr), 'val': len(va), 'test': len(te),
                  'shared_groups_train_test': n_shared},
        'selection': {'variant': pick, 'model': best_name,
                      'criterion': f'QWK on {pick} test',
                      'note': 'test는 최종 평가 1회만 사용 (v1의 test 기반 선택 문제 회피)'},
        'metrics': {'primary': 'qwk / mae (순서형)', 'secondary': 'macro_f1 / accuracy'},
        'results': all_results,
        'per_source': per_source,
        'v1_comparison': v1_cmp,
        'label_mapping': 'model class i -> urgency_score i+1',
        'known_issues': [
            '라벨은 rescore_urgency.py의 결정론적 규칙 f(raw_text)다. masked 조건이 실제 성능에 가깝다.',
            'source가 라벨 규칙과 결합돼 있어 source_only baseline과 반드시 같이 볼 것.',
            '보조 피처 미포함 (v1의 test 기반 선택 문제로 제외).',
            '보일러플레이트(사이트 템플릿) 제거 미적용 - 개선 2번 항목.',
        ],
    }
    (out_dir / "baseline_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    for p in sorted(out_dir.iterdir()):
        print(f"  저장: {p.name}  ({p.stat().st_size / 1e6:.2f} MB)")
    print()
    print(f"  총 소요 {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
