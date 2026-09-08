"""리포 전체가 공유하는 허깅페이스 데이터셋 다운로드 헬퍼.
01, 02(그리고 앞으로 추가될 03/04)가 전부 여기서 데이터를 받는다.

⚠️ `huggingface_hub`을 이 파일 최상단에서 import 하면 안 된다.
이 모듈은 `urgency_rule.py`를 통해 2-2 앱의 추론 경로에도 딸려 들어오는데,
`2-2-urgency-app/requirements.txt`에는 huggingface_hub이 없다(앱은 데이터셋을
내려받지 않으므로 넣을 이유도 없다). 최상단에서 import 하면 배포된 앱이
데이터와 무관한 의존성 때문에 임포트 단계에서 죽는다.
그래서 실제로 다운로드가 필요한 fetch() 안에서만 지연 import 한다.
(같은 이유의 선례: `2-1-.../check_label_leakage.py`의 hf_hub_download 지연 import)
"""
import os

HF_REPO_ID = "data-craftee/korean-it-recruit-dataset"


def get_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        import streamlit as st
        return st.secrets.get("HF_TOKEN", None)
    except Exception:
        return None


def fetch(filename: str, repo_id: str = HF_REPO_ID) -> str:
    """허깅페이스에서 파일을 받아 로컬 캐시 경로를 반환한다.
    hf_hub_download가 자체 캐싱하므로 재실행 시 재다운로드하지 않는다."""
    from huggingface_hub import hf_hub_download   # 지연 import — 모듈 상단 주석 참조

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=get_hf_token(),
    )