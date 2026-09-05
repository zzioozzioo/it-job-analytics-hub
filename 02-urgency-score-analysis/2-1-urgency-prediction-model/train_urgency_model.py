"""
train_urgency_model.py

채용 시급성(urgency_score) 예측 모델 학습 스크립트.

데이터: HuggingFace `data-craftee/korean-it-recruit-dataset` / master_merged_v2.json
라벨:  이미 부여된 urgency_score(1~5)를 그대로 사용한다. (재라벨링 하지 않음)

파이프라인
  1) 데이터 로드 + 클래스 분포 확인
  2) 피처 엔지니어링 (TF-IDF 1~2gram + 구조화 보조 피처)
  3) source × label stratified split
  4) compute_class_weight('balanced') -> XGBoost sample_weight
  5) Macro F1 + confusion matrix 평가
  6) 모델/벡터라이저 joblib 저장
  7) 요약 리포트 (Macro F1, 클래스별 P/R, 중요 피처 top 20)

실행: python train_urgency_model.py
      python train_urgency_model.py --sample 8000   # 빠른 스모크 테스트
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

sys.stdout.reconfigure(encoding='utf-8')

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HF_REPO_ID = "data-craftee/korean-it-recruit-dataset"
HF_FILENAME = "master_merged_v2.json"

RANDOM_STATE = 42
LIST_COLS = ['hard_skills', 'soft_skills', 'preferences', 'culture_keywords', 'final_techs']


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------------------------
def load_data(sample=None):
    rule("STEP 1. 데이터 로드 & 클래스 분포")
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, repo_type="dataset")
    print(f"  다운로드: {path}")
    with open(path, encoding='utf-8') as f:
        df = pd.DataFrame(json.load(f))

    df['raw_text'] = df['raw_text'].fillna('').astype(str)
    for c in LIST_COLS:
        df[c] = df[c].apply(lambda v: v if isinstance(v, list) else [])

    df = df[df['urgency_score'].notna()].copy()
    df['urgency_score'] = df['urgency_score'].astype(int)

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"  ⚠️ --sample {sample}: {len(df):,}행만 사용")

    print(f"  총 {len(df):,}행")

    print()
    print("  urgency_score 클래스 분포:")
    vc = df['urgency_score'].value_counts().sort_index()
    for k, v in vc.items():
        bar = '█' * int(v / len(df) * 45)
        print(f"    {k}점: {v:>7,} ({v/len(df)*100:5.2f}%) {bar}")
    n_cls = df['urgency_score'].nunique()
    imbalance = vc.max() / vc.min()
    print(f"    -> 클래스 {n_cls}개, 불균형비(최다/최소) = {imbalance:.2f}배")

    print()
    print("  source × urgency_score 교차표:")
    print(pd.crosstab(df['source'], df['urgency_score']).to_string())
    return df


# ---------------------------------------------------------------------------
# 2. 피처 엔지니어링
# ---------------------------------------------------------------------------
def build_aux_features(df):
    """구조화 보조 피처: 리스트 컬럼의 개수 + 텍스트 길이 지표.
    (특정 키워드 존재 여부는 TF-IDF가 이미 담당하므로 개수/길이 위주로 구성)"""
    feats = pd.DataFrame(index=df.index)
    for c in LIST_COLS:
        feats[f'n_{c}'] = df[c].apply(len)
    feats['n_skills_total'] = feats[[f'n_{c}' for c in LIST_COLS]].sum(axis=1)
    feats['len_raw_text'] = df['raw_text'].str.len()
    feats['n_lines'] = df['raw_text'].str.count('\n') + 1
    feats['n_digits'] = df['raw_text'].str.count(r'\d')
    return feats


def vectorize(train_texts, test_texts):
    rule("STEP 2. 피처 엔지니어링")
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        min_df=5,
        max_df=0.9,
        sublinear_tf=True,
    )
    t0 = time.time()
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    print(f"  TF-IDF(1~2gram): {Xtr.shape[1]:,} features  (fit {time.time()-t0:.1f}s)")
    print(f"    train {Xtr.shape}  test {Xte.shape}")
    return vec, Xtr, Xte


# ---------------------------------------------------------------------------
# 3~5. 학습 & 평가
# ---------------------------------------------------------------------------
def train_eval(Xtr, ytr, Xte, yte, label, n_estimators=300):
    classes = np.unique(ytr)
    weights = compute_class_weight('balanced', classes=classes, y=ytr)
    w_map = dict(zip(classes, weights))
    sample_weight = np.array([w_map[y] for y in ytr])

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=7,
        learning_rate=0.2,
        subsample=0.9,
        colsample_bytree=0.7,
        tree_method='hist',
        objective='multi:softmax',
        num_class=len(classes),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss',
    )
    t0 = time.time()
    clf.fit(Xtr, ytr, sample_weight=sample_weight)
    took = time.time() - t0
    pred = clf.predict(Xte)
    macro_f1 = f1_score(yte, pred, average='macro')
    print(f"  [{label}] Macro F1 = {macro_f1:.4f}   (학습 {took:.1f}s)")
    return clf, pred, macro_f1, w_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=None)
    ap.add_argument('--n-estimators', type=int, default=300)
    args = ap.parse_args()

    df = load_data(args.sample)

    # --- 3. source × label stratified split ---
    # 요청은 source 비율 유지지만, 불균형 다중분류라 label 비율도 함께 지켜야
    # 평가가 왜곡되지 않는다. 두 축을 결합한 키로 stratify 한다.
    strat_key = df['source'].astype(str) + "__" + df['urgency_score'].astype(str)

    # stratify는 모든 계층이 최소 2건이어야 한다.
    # 실제로 wanted×1점 조합이 단 1건 존재해서, 이런 희소 조합은
    # '같은 source에서 가장 큰 계층'으로 흡수시킨다.
    # (별도 __rare 키로 빼면 그 키가 또 1건짜리라 동일한 에러가 난다)
    counts = strat_key.value_counts()
    rare_keys = counts[counts < 2].index
    if len(rare_keys) > 0:
        big = counts[counts >= 2]
        fallback = {}
        for src in df['source'].unique():
            cand = big[big.index.str.startswith(f"{src}__")]
            fallback[src] = cand.idxmax() if len(cand) else big.idxmax()
        mask = strat_key.isin(rare_keys)
        strat_key = strat_key.copy()
        strat_key[mask] = df.loc[mask, 'source'].map(fallback)
        print(f"  ⚠️ 희소 계층 {list(rare_keys)} ({int(mask.sum())}건) → "
              f"동일 source의 최대 계층으로 흡수")

    idx_tr, idx_te = train_test_split(
        df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=strat_key
    )
    tr, te = df.loc[idx_tr], df.loc[idx_te]

    rule("STEP 3. Train/Test split (source × label stratified)")
    print(f"  train {len(tr):,} / test {len(te):,}")
    comp = pd.DataFrame({
        'train_%': tr['source'].value_counts(normalize=True).mul(100).round(2),
        'test_%': te['source'].value_counts(normalize=True).mul(100).round(2),
    })
    print(comp.to_string())
    comp2 = pd.DataFrame({
        'train_%': tr['urgency_score'].value_counts(normalize=True).mul(100).round(2),
        'test_%': te['urgency_score'].value_counts(normalize=True).mul(100).round(2),
    }).sort_index()
    print(comp2.to_string())

    # --- 2. 피처 ---
    vec, Xtr_txt, Xte_txt = vectorize(tr['raw_text'], te['raw_text'])

    aux_tr = build_aux_features(tr)
    aux_te = build_aux_features(te)
    scaler = StandardScaler()
    Atr = scaler.fit_transform(aux_tr.values.astype(float))
    Ate = scaler.transform(aux_te.values.astype(float))
    print(f"  보조 피처 {aux_tr.shape[1]}개: {list(aux_tr.columns)}")

    ytr = (tr['urgency_score'] - 1).to_numpy()   # XGBoost는 0-index 필요
    yte = (te['urgency_score'] - 1).to_numpy()

    # --- 4. 텍스트 단독 vs 텍스트+보조 비교 ---
    rule("STEP 4. 학습 (class_weight='balanced' -> sample_weight)")
    clf_txt, pred_txt, f1_txt, w_map = train_eval(
        Xtr_txt, ytr, Xte_txt, yte, "TF-IDF only", args.n_estimators)

    Xtr_all = sparse.hstack([Xtr_txt, sparse.csr_matrix(Atr)]).tocsr()
    Xte_all = sparse.hstack([Xte_txt, sparse.csr_matrix(Ate)]).tocsr()
    clf_all, pred_all, f1_all, _ = train_eval(
        Xtr_all, ytr, Xte_all, yte, "TF-IDF + aux", args.n_estimators)

    print()
    print("  클래스 가중치(balanced):",
          {int(k) + 1: round(float(v), 3) for k, v in sorted(w_map.items())})

    use_aux = f1_all > f1_txt
    print()
    print(f"  -> 보조 피처 {'채택' if use_aux else '미채택'} "
          f"(Δ Macro F1 = {f1_all - f1_txt:+.4f})")

    clf, pred, macro_f1 = (clf_all, pred_all, f1_all) if use_aux else (clf_txt, pred_txt, f1_txt)
    feat_names = list(vec.get_feature_names_out())
    if use_aux:
        feat_names += list(aux_tr.columns)

    # --- 5. 평가 ---
    rule("STEP 5. 평가")
    print(f"  Macro F1-Score = {macro_f1:.4f}")
    print()
    print("  클래스별 리포트:")
    print(classification_report(yte, pred,
                                target_names=[f'{i}점' for i in range(1, 6)],
                                digits=3, zero_division=0))
    print("  Confusion Matrix (행=실제, 열=예측):")
    cm = confusion_matrix(yte, pred)
    hdr = "        " + "".join(f"{f'pred{i}':>8}" for i in range(1, 6))
    print(hdr)
    for i, row in enumerate(cm):
        print(f"  true{i+1} " + "".join(f"{v:>8,}" for v in row))

    # --- 6. 저장 ---
    rule("STEP 6. 모델 저장")
    joblib.dump(clf, MODEL_DIR / "xgb_urgency_model.joblib")
    joblib.dump(vec, MODEL_DIR / "tfidf_vectorizer.joblib")
    meta = {
        'hf_repo_id': HF_REPO_ID,
        'hf_filename': HF_FILENAME,
        'n_rows': int(len(df)),
        'use_aux_features': bool(use_aux),
        'aux_columns': list(aux_tr.columns),
        'macro_f1': float(macro_f1),
        'macro_f1_text_only': float(f1_txt),
        'macro_f1_with_aux': float(f1_all),
        'class_weights': {int(k) + 1: float(v) for k, v in w_map.items()},
        'label_mapping': 'model class i -> urgency_score i+1',
    }
    if use_aux:
        joblib.dump(scaler, MODEL_DIR / "aux_scaler.joblib")
        meta['aux_scaler'] = 'aux_scaler.joblib'
    with open(MODEL_DIR / "model_meta.json", 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    for p in sorted(MODEL_DIR.iterdir()):
        print(f"  저장: {p.name}  ({p.stat().st_size/1e6:.2f} MB)")

    # --- 7. 요약 ---
    rule("STEP 7. 요약")
    print(f"  Macro F1-Score : {macro_f1:.4f}")
    print(f"  보조 피처 사용  : {use_aux}")
    print()
    print("  클래스별 precision / recall / f1:")
    rep = classification_report(yte, pred, target_names=[f'{i}점' for i in range(1, 6)],
                                output_dict=True, zero_division=0)
    print(f"    {'클래스':<8}{'precision':>11}{'recall':>9}{'f1':>9}{'support':>10}")
    for i in range(1, 6):
        r = rep[f'{i}점']
        print(f"    {i}점{'':<5}{r['precision']:>11.3f}{r['recall']:>9.3f}"
              f"{r['f1-score']:>9.3f}{int(r['support']):>10,}")

    print()
    print("  중요 피처 TOP 20 (XGBoost gain):")
    imp = clf.feature_importances_
    top = np.argsort(imp)[::-1][:20]
    for rank, i in enumerate(top, 1):
        print(f"    {rank:>2}. {feat_names[i]:<32} {imp[i]:.5f}")

    return macro_f1


if __name__ == '__main__':
    main()
