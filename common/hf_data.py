"""리포 전체가 공유하는 허깅페이스 데이터셋 다운로드 헬퍼.
01, 02(그리고 앞으로 추가될 03/04)가 전부 여기서 데이터를 받는다."""
import os
from huggingface_hub import hf_hub_download

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
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=get_hf_token(),
    )