"""
check_source_boilerplate.py

보일러플레이트 제거가 실제로 되었는지 검증하는 진단. 개선 2번.

[검증 아이디어]
'템플릿이 지워졌는가'를 직접 재는 방법은 source 분류 정확도다.
텍스트만 보고 jobkorea/saramin/wanted를 맞히는 분류기를 세우면,
템플릿이 남아 있는 한 정확도가 1.0에 가깝다. 템플릿을 지울수록 떨어진다.

바닥은 0.33(3-class 무작위)이 아니라 그보다 높다. 템플릿을 다 지워도
세 사이트의 문체와 직군 구성이 실제로 다르기 때문이다(wanted는 스타트업
직무기술서, saramin은 SI/공공 공고가 많다). 그 차이는 진짜 내용이므로
지우면 안 된다. 그래서 목표는 0.33이 아니라 '급격한 하락'이다.

단계:
  raw        원본 raw_text
  strip      strip_template() 적용 (꼬리 절단 + 정형 고지문 삭제 + 브랜드명)
  strip+sw   위 + source-wise DF stopword (train에서만 fit)

각 단계에서
  (a) source 분류 정확도  -> 낮을수록 템플릿이 지워진 것
  (b) urgency 예측 성능    -> 너무 떨어지면 본문 신호까지 깎은 것
을 함께 본다. (a)만 낮추고 (b)를 지키는 지점이 목표다.

실행: python check_source_boilerplate.py [--sample N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.svm import LinearSVC

sys.stdout.reconfigure(encoding='utf-8')

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from boilerplate import (fit_source_stopwords, overlap_report,  # noqa: E402
                         strip_series)
from check_label_leakage import mask_text  # noqa: E402

LOCAL_DATA = HERE.parents[1] / "data" / "master_merged_v2.json"
RANDOM_STATE = 42
N_CLASSES = 5


def rule(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def load():
    with open(LOCAL_DATA, encoding='utf-8') as f:
        df = pd.DataFrame(json.load(f))
    df['raw_text'] = df['raw_text'].fillna('').astype(str)
    df['source'] = df['source'].fillna('unknown').astype(str)
    df = df[df['urgency_score'].notna()].copy()
    df['urgency_score'] = df['urgency_score'].astype(int)
    return df.reset_index(drop=True)


def fit_predict(txt_tr, ytr, txt_te, stop_words=None):
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=5,
                          max_df=0.9, sublinear_tf=True, stop_words=stop_words)
    Xtr = vec.fit_transform(txt_tr)
    Xte = vec.transform(txt_te)
    clf = LinearSVC(class_weight='balanced', C=1.0, max_iter=5000,
                    random_state=RANDOM_STATE).fit(Xtr, ytr)
    return clf.predict(Xte), vec, clf


def top_source_features(vec, clf, k=12):
    """source 분류기가 가장 크게 의존하는 단어 = 남아 있는 지문."""
    names = np.array(vec.get_feature_names_out())
    out = {}
    for i, cls in enumerate(clf.classes_):
        w = clf.coef_[i] if clf.coef_.ndim > 1 else clf.coef_
        out[cls] = list(names[np.argsort(-w)[:k]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=12000)
    args = ap.parse_args()

    t0 = time.time()
    df = load()
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=RANDOM_STATE).reset_index(drop=True)

    rule("STEP 1. 표본")
    print(f"  {len(df):,}행")
    print("  source 분포:", df['source'].value_counts().to_dict())

    # 단순 랜덤 split (이 스크립트는 진단 전용이라 group 분리까지는 하지 않는다.
    # 절대 수치가 아니라 단계별 '차이'를 보는 것이 목적이다.)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(len(df))
    cut = int(len(df) * 0.8)
    tr, te = df.iloc[perm[:cut]], df.iloc[perm[cut:]]

    rule("STEP 2. 템플릿 제거량")
    stripped_tr = strip_series(tr['raw_text'], tr['source'])
    stripped_te = strip_series(te['raw_text'], te['source'])
    for src in sorted(df['source'].unique()):
        m = (tr['source'] == src).to_numpy()
        before = tr['raw_text'][m].str.len().mean()
        after = stripped_tr[m].str.len().mean()
        print(f"  {src:<10} 평균 {before:7.0f}자 -> {after:7.0f}자 "
              f"({(1 - after / before) * 100:5.1f}% 제거)")

    ov = overlap_report(tr['raw_text'], tr['source'], mask_text)
    print()
    print(f"  제거 총량 {ov['removed_chars']:,}자 중 라벨 패턴 매치 "
          f"{ov['label_matches_removed']:,}건이 함께 제거됨")
    print("  -> clean과 masked는 완전히 독립이 아니다. 완전 통제는 masked+clean 조건.")

    stop_words = fit_source_stopwords(stripped_tr, tr['source'])
    print()
    print(f"  source-wise DF stopword: {len(stop_words)}개")
    print(f"    예시: {stop_words[:15]}")

    # --- 단계별 평가 ---
    stages = [
        ('raw', tr['raw_text'], te['raw_text'], None),
        ('strip', stripped_tr, stripped_te, None),
        ('strip+sw', stripped_tr, stripped_te, stop_words),
    ]

    rule("STEP 3. (a) source 분류 정확도  — 낮을수록 템플릿이 지워진 것")
    print(f"    {'stage':<10}{'source Acc':>12}{'source F1':>12}{'초':>8}")
    src_rows, keep = [], {}
    for name, a, b, sw in stages:
        t1 = time.time()
        pred, vec, clf = fit_predict(a, tr['source'], b, sw)
        acc = accuracy_score(te['source'], pred)
        f1 = f1_score(te['source'], pred, average='macro')
        src_rows.append({'stage': name, 'source_acc': acc, 'source_f1': f1})
        keep[name] = (vec, clf)
        print(f"    {name:<10}{acc:>12.4f}{f1:>12.4f}{time.time() - t1:>8.1f}")

    rule("STEP 4. (b) urgency 예측 성능  — 너무 떨어지면 본문 신호까지 깎은 것")
    print(f"    {'stage':<10}{'QWK':>10}{'MacroF1':>10}{'초':>8}")
    urg_rows = []
    ytr = (tr['urgency_score'] - 1).to_numpy()
    yte = (te['urgency_score'] - 1).to_numpy()
    for name, a, b, sw in stages:
        t1 = time.time()
        pred, _, _ = fit_predict(a, ytr, b, sw)
        qwk = cohen_kappa_score(yte, pred, weights='quadratic',
                                labels=list(range(N_CLASSES)))
        mf1 = f1_score(yte, pred, average='macro', zero_division=0)
        urg_rows.append({'stage': name, 'qwk': qwk, 'macro_f1': mf1})
        print(f"    {name:<10}{qwk:>10.4f}{mf1:>10.4f}{time.time() - t1:>8.1f}")

    rule("STEP 5. 남아 있는 source 지문 (각 stage의 source 분류기 상위 단어)")
    for name in ['raw', 'strip+sw']:
        vec, clf = keep[name]
        print(f"  [{name}]")
        for cls, words in top_source_features(vec, clf, k=10).items():
            print(f"    {cls:<10}: {', '.join(words)}")
        print()

    rule("STEP 6. 요약")
    s = pd.DataFrame(src_rows).set_index('stage')
    u = pd.DataFrame(urg_rows).set_index('stage')
    both = s.join(u).round(4)
    print(both.to_string())
    print()
    d_src = s.loc['strip+sw', 'source_acc'] - s.loc['raw', 'source_acc']
    d_urg = u.loc['strip+sw', 'qwk'] - u.loc['raw', 'qwk']
    print(f"  source 식별력 {d_src:+.4f}  /  urgency QWK {d_urg:+.4f}")
    print("  source 식별력이 크게 떨어지고 urgency는 조금만 떨어졌다면 성공이다.")
    print("  urgency가 source 식별력만큼 떨어졌다면, 그 성능은 원래 source를")
    print("  알아맞히는 데서 나온 것이었다는 뜻이다.")

    (HERE / "source_boilerplate_report.json").write_text(
        json.dumps({'n_rows': len(df), 'strip_overlap': ov,
                    'n_stopwords': len(stop_words),
                    'stopwords_sample': stop_words[:80],
                    'source_clf': src_rows, 'urgency_clf': urg_rows},
                   ensure_ascii=False, indent=2), encoding='utf-8')
    print()
    print(f"  저장: source_boilerplate_report.json   (총 {time.time() - t0:.1f}s)")


if __name__ == '__main__':
    main()
