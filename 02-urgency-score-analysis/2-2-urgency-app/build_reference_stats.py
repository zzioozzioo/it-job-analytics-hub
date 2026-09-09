"""build_reference_stats.py — 앱이 쓰는 `reference_stats.json`을 다시 만든다.

---------------------------------------------------------------------------
왜 이 스크립트가 생겼나
---------------------------------------------------------------------------
앱은 채점 결과 옆에 "같은 사이트 공고 N건 중 이 점수 이하가 X%"와 "데이터셋의
접수 창 길이는 중앙값 30일"을 붙여 준다. 점수 하나만 보여주면 4점이 높은
것인지 낮은 것인지 알 수 없기 때문이다.

그 분포는 40,348행을 매번 읽을 수 없으니 미리 계산해 JSON으로 들고 있는데,
**만드는 스크립트가 저장소에 없었다.** 그래서 라벨 규칙이 v3 -> v4로 바뀌었을
때 이 파일만 v3 분포인 채로 남았고, 앱이 조용히 틀린 백분위를 보여줬다.

    v3 measurable 분포   1:9,744  2:15,457  3:4,427  4:712  5:229
    v4 measurable 분포   1:9,744  2:17,008  3:3,156  4:594  5: 67

'사전 계산 산출물에 만드는 방법이 붙어 있지 않으면 반드시 원본과 어긋난다'는
사례라 파일을 남긴다. 라벨을 다시 만들 때마다 이것도 함께 돌릴 것.

실행:
    python build_reference_stats.py            # 미리보기만
    python build_reference_stats.py --write    # reference_stats.json 갱신
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # 02-urgency-score-analysis/
sys.path.insert(0, str(HERE.parents[1]))      # repo root

from urgency_rule import (RULE_VERSION, clean_body,  # noqa: E402
                          is_measurable, parse_application_window, score_posting)
from common.hf_data import MASTER, fetch  # noqa: E402

SRC = MASTER
OUT = HERE / "reference_stats.json"


def build():
    with open(fetch(SRC), encoding='utf-8') as f:
        data = json.load(f)

    by_source, overall, meas_dist, windows = {}, Counter(), Counter(), []
    for x in data:
        raw = x.get('raw_text') or ''
        src = x['source']
        # 저장된 라벨을 믿지 않고 **현재 규칙으로 다시 채점한다.**
        # 이 파일이 어긋났던 원인이 "저장된 값과 규칙이 갈라진 것"이라,
        # 여기서만은 규칙을 한 번 더 돌리는 비용을 치른다.
        r = score_posting(raw, src)
        lv = r['urgency_score']
        overall[lv] += 1
        by_source.setdefault(src, Counter())[lv] += 1
        if r['measurable']:
            meas_dist[lv] += 1
            w = parse_application_window(clean_body(raw, src))[0]
            if w is not None:
                windows.append(w)

    def dist(c):
        return {str(i): int(c.get(i, 0)) for i in range(1, 6)}

    w = np.array(windows)
    return {
        'rule_version': RULE_VERSION,          # 어긋남을 눈으로 잡을 수 있게
        'n': len(data),
        'by_source': {s: {'n': int(sum(c.values())), 'dist': dist(c)}
                      for s, c in sorted(by_source.items())},
        'overall': dist(overall),
        'window_percentiles': {str(p): int(np.percentile(w, p))
                               for p in (10, 25, 50, 75, 90)},
        'measurable': {'n': int(sum(meas_dist.values())), 'dist': dist(meas_dist)},
        'window_n': len(windows),
    }


def main(write):
    stats = build()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if OUT.exists():
        old = json.loads(OUT.read_text(encoding='utf-8'))
        ov, nv = old.get('rule_version', '(없음)'), stats['rule_version']
        print(f"\n기존 파일의 rule_version: {ov}  ->  {nv}")
        if old.get('measurable', {}).get('dist') != stats['measurable']['dist']:
            print("  measurable 분포가 다르다 — 앱의 백분위가 어긋나 있었다는 뜻이다.")
    if write:
        OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\n저장: {OUT}")
    else:
        print("\n(--write 를 붙이면 reference_stats.json 을 갱신합니다)")


if __name__ == '__main__':
    main('--write' in sys.argv)
