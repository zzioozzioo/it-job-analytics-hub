"""
urgency_model.py — v3 모델의 추론 인터페이스 (앱과 스크립트가 공유)

`.joblib`을 직접 로드하면 안 되는 이유는 2-1의 predict_urgency.py와 같다.
저장된 성능은 (전처리 masked+clean) + (기댓값 반올림 예측)이 함께 걸렸을 때의
숫자다. 셋을 한 곳에 묶어둔다.

이 모듈은 **모델을 두 번째 의견으로만** 쓰도록 설계돼 있다. 규칙이 계산
가능한 공고에서는 규칙이 정의상 정답이고 모델은 그 근사치다. 왜 그런데도
모델을 두는지는 2-2 앱의 '한계' 탭에 적었다.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "2-1-urgency-prediction-model"))

from urgency_rule import is_measurable  # noqa: E402
from train_urgency_baseline import make_transform  # noqa: E402

MODEL_DIR = HERE / "2-1-urgency-prediction-model" / "models_v3"
VARIANT = 'masked+clean'
N_CLASSES = 5


class UrgencyModel:
    """학습 때와 동일한 전처리 + 확정된 예측 방식."""

    def __init__(self, model_dir=MODEL_DIR):
        import json
        model_dir = Path(model_dir)
        if not (model_dir / "urgency_model.joblib").exists():
            raise FileNotFoundError(
                f"{model_dir} 에 모델이 없습니다. "
                f"2-1에서 `python train_urgency_v3.py`를 먼저 실행하세요.")
        self.model = joblib.load(model_dir / "urgency_model.joblib")
        self.vec = joblib.load(model_dir / "urgency_tfidf.joblib")
        self.meta = json.loads((model_dir / "model_meta.json").read_text(encoding='utf-8'))
        self.mode = self.meta.get('prediction', 'argmax')
        self._transform = make_transform(VARIANT)

    def _matrix(self, texts, sources):
        texts = pd.Series(list(texts))
        sources = pd.Series(list(sources), index=texts.index)
        return self.vec.transform(self._transform(texts, sources))

    def predict(self, texts, sources):
        """1~5 점수."""
        X = self._matrix(texts, sources)
        if self.mode == 'expected_round' and hasattr(self.model, 'predict_proba'):
            ev = self.model.predict_proba(X) @ np.arange(N_CLASSES)
            return np.clip(np.rint(ev), 0, N_CLASSES - 1).astype(int) + 1
        return np.asarray(self.model.predict(X)).astype(int) + 1

    def predict_proba(self, texts, sources):
        """클래스 확률. predict_proba가 없는 모델이면 None."""
        if not hasattr(self.model, 'predict_proba'):
            return None
        return self.model.predict_proba(self._matrix(texts, sources))

    def predict_with_scope(self, texts, sources):
        """예측 + '검증된 적용 범위 안인가' 플래그.

        in_scope=False면 그 공고에는 채용 메타데이터가 없어 모델이 검증되지
        않은 영역이다. 화면에서 이 플래그를 반드시 함께 보여줄 것."""
        texts = list(texts)
        return pd.DataFrame({
            'urgency_score': self.predict(texts, sources),
            'in_scope': [bool(is_measurable(t)) for t in texts],
        })

    # --- 저장된 성능 지표 (화면 표시용) ---
    @property
    def test_metrics(self):
        return self.meta.get('results', {}).get('final_test', {})

    @property
    def transfer_summary(self):
        return self.meta.get('results', {}).get('exp_b_summary', {})
