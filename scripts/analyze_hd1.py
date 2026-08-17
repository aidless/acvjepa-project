#!/usr/bin/env python3
"""H-D1 配对判定（本地，取云端两个 results.json）

预注册口径（决策记录 2026-08-17）：
- 指标=ImageReach 闭环成功率（逐 rollout 成功向量，按 env seed 配对，顺序一致）；
- 判定=配对 Wilcoxon（单侧：失败注入更优）p<0.05 且 Cohen's h>0.3 → 支持；
- 预期效应=失败注入集 +15–30pp（h≥0.5）。
"""
import argparse
import json

import numpy as np
from scipy import stats


def cohens_h(p1: float, p2: float) -> float:
    import math
    p1 = max(0.0, min(1.0, p1)); p2 = max(0.0, min(1.0, p2))
    return 2.0 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("succ_json")
    ap.add_argument("fail_json")
    ap.add_argument("--out", default="hd1_verdict.json")
    args = ap.parse_args()

    s = json.load(open(args.succ_json, encoding="utf-8"))
    f = json.load(open(args.fail_json, encoding="utf-8"))
    arm = "C_light_jepa"
    s_vec = s["per_rollout_success"][arm]
    f_vec = f["per_rollout_success"][arm]
    assert len(s_vec) == len(f_vec), "rollout 数应一致（同 env seed 序列）"

    diff = np.asarray(f_vec, float) - np.asarray(s_vec, float)
    if np.all(diff == 0):
        p = 1.0
    else:
        try:
            stat, p = stats.wilcoxon(s_vec, f_vec, zero_method="wilcox", alternative="greater")
        except ValueError:
            p = 1.0
    sr_s, sr_f = float(np.mean(s_vec)), float(np.mean(f_vec))
    h = cohens_h(sr_s, sr_f)  # fail vs success（失败更高→h>0）
    # Cliff's delta：失败值 > 成功值 的比例优势
    gt = sum(1 for a in f_vec for b in s_vec if a > b)
    lt = sum(1 for a in f_vec for b in s_vec if a < b)
    n = len(s_vec) * len(f_vec)
    cliff = (gt - lt) / n if n else 0.0

    support = bool(p < 0.05 and h > 0.3)
    verdict = {
        "hypothesis": "H-D1",
        "success_rates": {"all_success": sr_s, "fail30": sr_f,
                          "diff_pp": round((sr_f - sr_s) * 100, 1)},
        "paired_wilcoxon_p_greater": float(p),
        "cohens_h_fail_vs_success": round(float(h), 4),
        "cliff_delta_fail_vs_success": round(float(cliff), 4),
        "support": support,
        "criterion": "p<0.05 and h>0.3",
        "disclaimer": "synthetic 2D env + synthetic failure-recovery semantics; chain-level evidence",
    }
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())