"""
predict_urgency.py

최종 모델의 추론 인터페이스.

[왜 이 파일이 필요한가]
models_transfer/transfer_model.joblib 을 그냥 로드해서 .predict()를 부르면
argmax가 나온다. 그런데 최종 확정된 예측 방식은 '클래스 확률의 기댓값 반올림'
이다(finalize_model.py 참조). 즉 joblib 파일만으로는 최종 성능이 재현되지 않는다.
전처리(masked+clean)도 학습 때와 똑같이 걸어야 한다.

이 모듈이 그 세 가지를 하나로 묶는다.
  1) 텍스트 변환  masked+clean  (학습 도메인과 일치시킨다)
  2) TF-IDF 변환
  3) 기댓값 반올림 예측

[적용 범위 — 반드시 지킬 것]
이 모델은 '메타데이터가 있는 공고'(rescore_urgency.is_measurable() == True)
에서만 검증됐다. 그 밖에는 근거가 없다.
  - 메타데이터 없는 공고: leave-one-source-out 전이 QWK 0.04. 사실상 무작위다.
  - 학습에 없던 채용 사이트: 같은 이유로 신뢰할 수 없다.
predict()는 범위를 벗어난 입력에 대해 scope 플래그를 함께 돌려준다.
호출부에서 그 플래그를 무시하지 말 것.

[더 중요한 사실]
메타데이터가 있는 공고라면 rescore_urgency.py 규칙을 그대로 돌리는 편이 낫다.
라벨 자체가 그 규칙의 출력이므로 규칙은 정의상 정답을 준다. 이 모델은 규칙의
근사치다. 이 모듈은 '규칙을 못 쓰는 상황에서의 참고값'과 '연구 재현' 용도다.

실행(자기검증): python predict_urgency.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from rescore_urgency import is_measurable  # noqa: E402
from train_urgency_baseline import make_transform  # noqa: E402

MODEL_DIR = HERE / "models_transfer"
VARIANT = 'masked+clean'
N_CLASSES = 5


class UrgencyPredictor:
    """학습 때와 동일한 전처리 + 기댓값 반올림 예측을 한 곳에 묶은 추론기."""

    def __init__(self, model_dir=MODEL_DIR):
        self.model = joblib.load(Path(model_dir) / "transfer_model.joblib")
        self.vec = joblib.load(Path(model_dir) / "transfer_tfidf.joblib")
        self._transform = make_transform(VARIANT)

    def predict_proba(self, texts, sources):
        texts = pd.Series(list(texts))
        sources = pd.Series(list(sources), index=texts.index)
        return self.model.predict_proba(self.vec.transform(
            self._transform(texts, sources)))

    def predict(self, texts, sources):
        """1~5 점수를 돌려준다. 기댓값 반올림(최종 확정 방식).

        argmax가 아닌 이유: 라벨이 순서형이라 '2점 0.45 / 3점 0.44'인 공고를
        2점으로 단정하는 것보다 기댓값을 반올림하는 편이 QWK/MAE가 낫다.
        (test에서 QWK 0.7775 -> 0.7874, MAE 0.4407 -> 0.4291,
         ±1점 이내 정확도 93.7% -> 96.4%)"""
        proba = self.predict_proba(texts, sources)
        ev = proba @ np.arange(N_CLASSES)
        return np.clip(np.rint(ev), 0, N_CLASSES - 1).astype(int) + 1

    def predict_with_scope(self, texts, sources):
        """예측과 함께 '적용 범위 안인가'를 돌려준다.

        in_scope=False면 그 예측은 검증된 근거가 없다. 표시하더라도
        참고값임을 반드시 함께 알릴 것."""
        texts = list(texts)
        return pd.DataFrame({
            'urgency_score': self.predict(texts, sources),
            'in_scope': [bool(is_measurable(t)) for t in texts],
        })


def _self_check():
    """저장된 test 성능이 이 인터페이스로 재현되는지 확인한다."""
    import json
    from sklearn.model_selection import StratifiedGroupKFold
    from train_urgency_baseline import (LOCAL_DATA, RANDOM_STATE,
                                        dedup_and_group, evaluate, rule)

    sys.stdout.reconfigure(encoding='utf-8')
    rule("자기검증: 저장된 최종 성능이 재현되는가")

    with open(LOCAL_DATA, encoding='utf-8') as f:
        df = pd.DataFrame(json.load(f))
    df['raw_text'] = df['raw_text'].fillna('').astype(str)
    df['source'] = df['source'].fillna('unknown').astype(str)
    df = df[df['urgency_score'].notna()].copy()
    df['urgency_score'] = df['urgency_score'].astype(int)
    df = df.reset_index(drop=True)
    meas = df[df['raw_text'].map(is_measurable)].reset_index(drop=True)
    meas, groups, _ = dedup_and_group(meas)
    strat = meas['source'] + "__" + meas['urgency_score'].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    _, te_pos = next(sgkf.split(meas, strat, groups))
    te = meas.iloc[te_pos].reset_index(drop=True)

    pred = UrgencyPredictor().predict(te['raw_text'], te['source'])
    got = evaluate(te['urgency_score'] - 1, pred - 1)
    want = json.loads((MODEL_DIR / "transfer_meta.json").read_text(
        encoding='utf-8'))['prediction']['final']

    print(f"  test {len(te):,}행")
    ok = True
    for k in ['qwk', 'mae', 'off_by_1', 'macro_f1', 'accuracy']:
        same = abs(got[k] - want[k]) < 1e-9
        ok &= same
        print(f"    {k:<10} 저장 {want[k]:.4f}  재현 {got[k]:.4f}  "
              f"{'일치' if same else '!! 불일치'}")
    print()
    print("  " + ("모든 지표 일치. 인터페이스가 최종 모델을 정확히 재현한다."
                  if ok else "!! 불일치. 전처리나 예측 방식이 어긋났다."))
    return ok


if __name__ == '__main__':
    sys.exit(0 if _self_check() else 1)
