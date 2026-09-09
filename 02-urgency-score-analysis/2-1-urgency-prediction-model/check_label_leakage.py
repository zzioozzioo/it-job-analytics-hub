"""
check_label_leakage.py

라벨 누출(label leakage) 진단용 ablation.

배경:
  master_merged_v2.json의 urgency_score는 rescore_urgency.py가 raw_text에서
  정규식으로 추출한 신호(남은기간 N일 / 접수기간 날짜 / 모집인원 N명 / 긴급 어휘)로
  결정론적으로 계산된 값이다. 즉 라벨 = f(raw_text) 이며, 그 f가 읽은 구간이
  학습 피처(raw_text)에 그대로 들어 있다.

  이 상태에서 TF-IDF 모델의 Macro F1이 높게 나오는 것은 '시급성을 예측했다'가 아니라
  '정규식을 역공학했다'에 가깝다.

이 스크립트는 라벨 생성에 쓰인 구간을 raw_text에서 가려(mask) 놓고 다시 학습해서,
성능이 얼마나 떨어지는지로 누출 규모를 정량화한다.

  - full   : 원본 raw_text (= train_urgency_model.py와 동일 조건)
  - masked : 라벨 산출에 쓰인 패턴을 <MASK>로 치환한 raw_text

실행: python check_label_leakage.py [--sample N]
"""

import argparse
import json
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

sys.stdout.reconfigure(encoding='utf-8')

RANDOM_STATE = 42
HF_REPO_ID = "data-craftee/korean-it-recruit-dataset"
HF_FILENAME = "master_merged_v2.json"

# rescore_urgency.py가 라벨을 만들 때 실제로 읽은 패턴들
LABEL_SOURCE_PATTERNS = [
    r'남은기간\s*[0-9]+\s*일',
    r'접수\s*기간.{0,90}',
    r'마감일\s*[0-9]{4}\.[0-9]{2}\.[0-9]{2}',
    r'마감일\s*상시채용',
    r'시작일\s*[0-9]{4}\.[0-9]{2}\.[0-9]{2}',
    r'모집인원\s*[0-9]+\s*명',
    r'[0-9]{1,3}\s*명\s*(?:내외\s*)?(?:모집|채용|선발)',
    r'다수\s*(?:모집|채용|선발)',
    r'급구|긴급\s*채용|긴급채용|시급히|서둘러',
    r'즉시\s*(?:입사|출근|근무|투입|합류)|바로\s*출근|조속히|즉시\s*채용',
    r'결원|충원|대체\s*인력|공석',
    r'조기\s*마감|마감\s*임박|충원\s*시\s*마감|채용\s*시\s*마감',
    r'합격\s*축하금|입사\s*축하금|사이닝\s*보너스|정착\s*지원금',
    r'상시\s*채용|수시\s*채용|상시\s*모집|연중\s*수시',
    r'채용\s*시\s*마감|채용시까지|채용\s*시\s*까지',
]
MASK_RX = re.compile('|'.join(LABEL_SOURCE_PATTERNS))


def mask_text(t: str) -> str:
    return MASK_RX.sub(' <MASK> ', t or '')


def run(texts_tr, texts_te, ytr, yte, label, n_estimators):
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000,
                          min_df=5, max_df=0.9, sublinear_tf=True)
    Xtr = vec.fit_transform(texts_tr)
    Xte = vec.transform(texts_te)

    classes = np.unique(ytr)
    w = compute_class_weight('balanced', classes=classes, y=ytr)
    wmap = dict(zip(classes, w))
    sw = np.array([wmap[y] for y in ytr])

    clf = XGBClassifier(n_estimators=n_estimators, max_depth=7, learning_rate=0.2,
                        subsample=0.9, colsample_bytree=0.7, tree_method='hist',
                        objective='multi:softmax', num_class=len(classes),
                        n_jobs=-1, random_state=RANDOM_STATE, eval_metric='mlogloss')
    t0 = time.time()
    clf.fit(Xtr, ytr, sample_weight=sw)
    f1 = f1_score(yte, clf.predict(Xte), average='macro')
    print(f"  [{label:<8}] Macro F1 = {f1:.4f}  (features {Xtr.shape[1]:,}, {time.time()-t0:.0f}s)")
    return f1, clf, vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=12000)
    ap.add_argument('--n-estimators', type=int, default=150)
    args = ap.parse_args()

    # 지연 임포트. 이 모듈의 mask_text 는 2-2 앱의 추론 경로에도 들어가는데,
    # huggingface_hub 를 최상단에서 import 하면 앱 requirements 에 없는 무거운
    # 의존성이 딸려온다(없으면 모델 로딩이 조용히 실패한다).
    # train_urgency_baseline.load_data() 도 같은 이유로 이 방식을 쓴다.
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, repo_type="dataset")
    df = pd.DataFrame(json.load(open(path, encoding='utf-8')))
    df['raw_text'] = df['raw_text'].fillna('').astype(str)
    df = df[df['urgency_score'].notna()].copy()
    df['urgency_score'] = df['urgency_score'].astype(int)

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"진단 표본: {len(df):,}행 (n_estimators={args.n_estimators})")

    strat = df['source'].astype(str) + "__" + df['urgency_score'].astype(str)
    counts = strat.value_counts()
    rare = counts[counts < 2].index
    if len(rare):
        big = counts[counts >= 2]
        fb = {s: (big[big.index.str.startswith(f"{s}__")].idxmax()
                  if len(big[big.index.str.startswith(f'{s}__')]) else big.idxmax())
              for s in df['source'].unique()}
        m = strat.isin(rare)
        strat = strat.copy()
        strat[m] = df.loc[m, 'source'].map(fb)

    tr, te = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=strat)
    ytr = (tr['urgency_score'] - 1).to_numpy()
    yte = (te['urgency_score'] - 1).to_numpy()

    print()
    f1_full, _, _ = run(tr['raw_text'], te['raw_text'], ytr, yte, "full", args.n_estimators)
    f1_mask, clf_m, vec_m = run(tr['raw_text'].map(mask_text), te['raw_text'].map(mask_text),
                                ytr, yte, "masked", args.n_estimators)

    print()
    print("=" * 70)
    print(f"  라벨 산출 구간 제거 시 Macro F1: {f1_full:.4f} -> {f1_mask:.4f} "
          f"({f1_mask - f1_full:+.4f}, {(f1_mask-f1_full)/f1_full*100:+.1f}%)")
    drop = (f1_full - f1_mask) / f1_full * 100
    if drop > 30:
        print(f"  => 성능의 {drop:.0f}%가 라벨 생성 구간에서 나왔다. 심각한 라벨 누출.")
    elif drop > 10:
        print(f"  => 성능의 {drop:.0f}%가 라벨 생성 구간 의존. 상당한 누출.")
    else:
        print(f"  => 누출 영향 제한적({drop:.0f}%).")
    print("=" * 70)

    names = vec_m.get_feature_names_out()
    imp = clf_m.feature_importances_
    print()
    print("  masked 모델의 상위 피처 15 (누출 제거 후 실제로 쓰인 단서):")
    for r, i in enumerate(np.argsort(imp)[::-1][:15], 1):
        print(f"    {r:>2}. {names[i]:<30} {imp[i]:.5f}")


if __name__ == '__main__':
    main()
