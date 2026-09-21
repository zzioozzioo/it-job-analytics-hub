"""리포 전체가 공유하는 데이터셋 접근 헬퍼 — **로컬 우선, 없으면 그 자리에 받는다.**
01, 02(그리고 앞으로 추가될 03/04)가 전부 여기서 데이터를 연다.

파일명 상수도 여기 있다. 경로든 이름이든 데이터셋에 관한 것은 이 모듈만 본다.

---------------------------------------------------------------------------
왜 한 곳으로 모았나
---------------------------------------------------------------------------
전에는 로컬 우선 판정이 세 벌 있었고 그중 둘이 로컬을 **보지 않았다.**

    train_urgency_baseline.data_path()   로컬 우선 -> 없으면 HF   (v2만)
    urgency_rule.py / 01/app.py          항상 HF 캐시
    train_urgency_model.py               hf_hub_download 직접 호출
    train_urgency_transfer.py            로컬 경로 직접 (없으면 그냥 죽음)

그래서 `data/master_merged.json`이 바로 옆에 있는데도 캐시에서 178MB를 읽었고,
같은 파일이 `data/`와 `~/.cache/huggingface/`에 이중으로 쌓였다(캐시만 680MB,
스냅샷 리비전이 둘이라 원본이 두 벌씩 들어 있었다).

지금은 `fetch()` 하나뿐이다.

    1. `data/<파일명>`이 있으면 그것을 쓴다.
    2. 없으면 **`data/` 안으로** 내려받는다.
       (별도 캐시 디렉터리에 두지 않는다 — 그래야 두 벌이 생기지 않는다)
    3. 허깅페이스에도 없으면, 만드는 명령을 알려주고 멈춘다.

---------------------------------------------------------------------------
3번을 목록으로 관리하지 않는 이유
---------------------------------------------------------------------------
처음에는 "허깅페이스에 있는 파일" 목록을 상수로 들고 있었다. 그러면 파일을
하나 올릴 때마다 코드를 고쳐야 하고, 안 고치면 **올려놨는데도 없다고 우긴다.**

그래서 목록을 지웠다. 그냥 받아보고, 없다고 하면(`EntryNotFoundError`) 그때
안내한다. 새 파일을 허깅페이스에 올리면 코드 변경 없이 곧바로 받아진다.

`GENERATED_BY`는 "받을 수 없을 때 어떻게 만드는가"를 적어둔 것이지 가용성
목록이 아니다. `master_merged_v4.json`이 나중에 업로드되면 1·2번에서 해결되고
이 표는 쓰이지 않게 된다 — 지워도 되지만, 라벨 파일이 **규칙의 산출물**이라는
사실 자체는 남겨둘 가치가 있어 둔다.

---------------------------------------------------------------------------
⚠️ `huggingface_hub`을 이 파일 최상단에서 import 하면 안 된다.
이 모듈은 `urgency_rule.py`를 통해 2-2 앱의 추론 경로에도 딸려 들어오는데,
`2-2-urgency-app/requirements.txt`에는 huggingface_hub이 없다(앱은 데이터셋을
내려받지 않으므로 넣을 이유도 없다). 최상단에서 import 하면 배포된 앱이
데이터와 무관한 의존성 때문에 임포트 단계에서 죽는다.
그래서 실제로 다운로드가 필요한 순간에만 지연 import 한다.
"""
import os
from pathlib import Path

HF_REPO_ID = "data-craftee/korean-it-recruit-dataset"

# 저장소 루트의 data/. git 제외 대상이고, 없으면 만든다.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# 데이터셋 파일명 — 문자열을 각자 들고 있지 말 것
# ---------------------------------------------------------------------------
MASTER = "master_merged.json"          # 원본 40,348건
MASTER_V2 = "master_merged_v2.json"    # 라벨 규칙 v2 (rescore_urgency.py)
MASTER_V3 = "master_merged_v3.json"    # 라벨 규칙 v3
MASTER_V4 = "master_merged_v4.json"    # 라벨 규칙 v4 (비교 기준으로 보존)
MASTER_V5 = "master_merged_v5.json"    # 라벨 규칙 v5 — 현재 정본
#                                      # 2026-09-21에 HF 업로드 완료.

#: 라벨 버전 -> 파일명. `master('v5')` 처럼 쓴다.
MASTER_BY_VERSION = {'raw': MASTER, 'v2': MASTER_V2, 'v3': MASTER_V3,
                     'v4': MASTER_V4, 'v5': MASTER_V5}

# 내려받을 수 없을 때 안내할 "만드는 법". 가용성 목록이 아니다(상단 주석 참조).
GENERATED_BY = {
    MASTER: "python build_master_dataset.py            (원본 병합)",
    MASTER_V2: "python rescore_urgency.py --write        (규칙 v2)",
    MASTER_V3: "python 02-urgency-score-analysis/urgency_rule.py --write   (당시 규칙 v3)",
    MASTER_V4: "python 02-urgency-score-analysis/urgency_rule.py --write   (당시 규칙 v4)",
    MASTER_V5: "python 02-urgency-score-analysis/urgency_rule.py --write   (규칙 v5, 현재)",
}


def master(version: str = 'v5') -> str:
    """라벨 버전 태그로 파일명을 얻는다. 'raw' 는 라벨 이전의 원본."""
    try:
        return MASTER_BY_VERSION[version]
    except KeyError:
        raise ValueError(
            f"알 수 없는 라벨 버전 {version!r}. "
            f"가능한 값: {', '.join(MASTER_BY_VERSION)}") from None


def get_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        import streamlit as st
        return st.secrets.get("HF_TOKEN", None)
    except Exception:
        return None


def local_path(filename: str) -> Path:
    """`data/<filename>` 경로. 존재 여부는 보지 않는다."""
    return DATA_DIR / filename


def fetch(filename: str, repo_id: str = HF_REPO_ID) -> str:
    """데이터 파일의 로컬 경로를 돌려준다. 없으면 `data/` 안으로 받아온다.

    반환형이 str인 것은 기존 호출부(`open(fetch(...))`)와의 호환 때문이다."""
    p = local_path(filename)
    if p.exists():
        return str(p)

    # 지연 import — 상단 주석 참조
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # local_dir 를 주면 별도 캐시 블롭이 아니라 이 디렉터리에 그대로 놓인다.
        # 두 벌이 생기지 않게 하려는 것이 이 함수의 요점이다.
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=get_hf_token(),
            local_dir=str(DATA_DIR),
        )
    except LocalEntryNotFoundError as e:
        # 네트워크에 못 닿은 것이다. 여기서 "직접 만드세요"라고 하면 오진이다.
        raise FileNotFoundError(
            f"{p} 가 없고 허깅페이스에도 닿지 못했습니다(네트워크 확인).\n"
            f"  원본: {repo_id} / {filename}") from e
    except EntryNotFoundError as e:
        how = GENERATED_BY.get(filename)
        raise FileNotFoundError(
            f"{p} 가 없고, 허깅페이스 {repo_id} 에도 {filename} 이 없습니다.\n"
            + (f"  이 파일은 내려받는 것이 아니라 만드는 것입니다:\n    {how}\n"
               if how else "")
            + "  (업로드했다면 코드 수정 없이 바로 받아집니다 — 목록을 두지 않습니다)"
        ) from e
