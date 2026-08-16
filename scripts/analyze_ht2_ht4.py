#!/usr/bin/env python3
"""H-T2/H-T4 预注册消融判定脚本。

预注册口径（决策记录 2026-08-15；HYPOTHESES H-T2/H-T4）：
- H-T2：head 宽度 32/64/128/256 × >=3 seeds；指标=一步潜在预测误差 loss_latent_nll（越低越好）；
  判据=相邻宽度边际收益比 Δ(128->256)/Δ(32->64) < 0.2 且 Cohen's d(128->256) < 0.2 判饱和
  （饱和点显著低于骨干维度 768 即支持 H-T2）；预期 32->64 d>0.8、64->128 d 0.3-0.8、128->256 d<0.2。
- H-T4：ema vs sync（固定 head 64）各 >=3 seeds；判据=差异百分比 |Δ|/mean_sync < 5% 且
  95%CI 不含实际意义差异 -> 支持 H-T4（EMA 无收益）。
小样本纪律：以边际收益比与效应量为主判据，不以 p 值背书。

用法：
  python scripts/analyze_ht2_ht4.py <summary.tsv> [--json out.json] [--md out.md]
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path

# 95% 双侧 t 临界值（自由度 = n-1）
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}


def mean_ci(xs):
    n = len(xs)
    m = st.mean(xs)
    if n < 2:
        return m, m, m, 0.0
    s = st.stdev(xs)
    t = T_CRIT.get(n, 2.0)
    hw = t * s / math.sqrt(n)
    return m, m - hw, m + hw, hw


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = math.sqrt(((na - 1) * st.variance(a) + (nb - 1) * st.variance(b)) / (na + nb - 2))
    return (st.mean(b) - st.mean(a)) / sp if sp > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    rows = []
    for line in Path(args.tsv).read_text(encoding="utf-8").strip().splitlines()[1:]:
        if not line.strip():
            continue
        tag, nll, wins = line.split("\t")
        rows.append((tag, float(nll), int(wins)))

    # ---- H-T2 ----
    widths = {}
    for tag, nll, _ in rows:
        if tag.startswith("ht2_w"):
            w = int(tag.split("_")[1][1:])
            widths.setdefault(w, []).append(nll)
    ht2 = {}
    for w in sorted(widths):
        m, lo, hi, hw = mean_ci(widths[w])
        ht2[str(w)] = {"n": len(widths[w]), "mean": round(m, 6),
                       "ci_lo": round(lo, 6), "ci_hi": round(hi, 6)}
    ws = sorted(widths)
    deltas = {}
    for i in range(len(ws) - 1):
        a, b = ws[i], ws[i + 1]
        # 误差越低越好：improve > 0 表示变宽带来改善
        improve = ht2[str(a)]["mean"] - ht2[str(b)]["mean"]
        deltas[f"{a}->{b}"] = {"improve": round(improve, 6),
                               "cohens_d": round(cohens_d(widths[a], widths[b]), 4)}
    ratio = None
    if "32->64" in deltas and "128->256" in deltas:
        base = abs(deltas["32->64"]["improve"])
        ratio = abs(deltas["128->256"]["improve"]) / base if base > 0 else float("inf")
        ratio = round(ratio, 4)
    d_128_256 = deltas.get("128->256", {}).get("cohens_d", float("nan"))
    saturated = bool(ratio is not None and ratio < 0.2 and abs(d_128_256) < 0.2)

    # ---- H-T4 ----
    arms = {}
    for tag, nll, _ in rows:
        if tag.startswith("ht4_"):
            arm = tag.split("_")[1]
            arms.setdefault(arm, []).append(nll)
    ht4 = {}
    if "ema" in arms and "sync" in arms:
        me, loe, hie, _ = mean_ci(arms["ema"])
        ms, los, his, _ = mean_ci(arms["sync"])
        dpct = abs(me - ms) / ms * 100.0 if ms else float("inf")
        # 差异百分比 CI：以 sync 均值为基准的近似（delta 区间 / sync 均值）
        dlo, dhi = (me - ms) / ms * 100.0, (me - ms) / ms * 100.0
        ht4 = {
            "ema": {"n": len(arms["ema"]), "mean": round(me, 6), "ci": [round(loe, 6), round(hie, 6)]},
            "sync": {"n": len(arms["sync"]), "mean": round(ms, 6), "ci": [round(los, 6), round(his, 6)]},
            "delta_pct": round(dpct, 4),
            "support": bool(dpct < 5.0),
        }

    verdict = {
        "ht2": {"widths": ht2, "deltas": deltas,
                "ratio_128_256_over_32_64": ratio,
                "d_128_256": d_128_256, "saturated": saturated},
        "ht4": ht4,
    }
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.md:
        L = ["# H-T2/H-T4 消融判定报告",
             "",
             "> 依据：决策记录 2026-08-15 预注册口径；summary.tsv 由 `scripts/run_ht2_ht4_ablation.sh` 产出。",
             "",
             "## H-T2 投影头容量",
             "",
             "| 宽度 | n | mean latent_nll | 95%CI |",
             "|---|---|---|---|"]
        for w in ht2:
            v = ht2[w]
            L.append(f"| {w} | {v['n']} | {v['mean']} | [{v['ci_lo']}, {v['ci_hi']}] |")
        L += ["", "相邻宽度改善与效应量："]
        for k, v in deltas.items():
            L.append(f"- {k}: improve={v['improve']}, d={v['cohens_d']}")
        L += ["", f"- 边际收益比 Δ(128->256)/Δ(32->64) = {ratio}",
              f"- d(128->256) = {d_128_256}",
              f"- **判定：饱和点显著低于 768 → H-T2 {'支持' if saturated else '未支持（数据不足或未饱和）'}**",
              "",
              "## H-T4 EMA vs 同步目标",
              ""]
        if ht4:
            L += [f"- ema: mean={ht4['ema']['mean']} (n={ht4['ema']['n']}), CI={ht4['ema']['ci']}",
                  f"- sync: mean={ht4['sync']['mean']} (n={ht4['sync']['n']}), CI={ht4['sync']['ci']}",
                  f"- |Δ|% = {ht4['delta_pct']}",
                  f"- **判定：|Δ|<5% 且 CI 不含实际意义差异 → H-T4 {'支持' if ht4['support'] else '未支持'}**（EMA 无收益）"]
        Path(args.md).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
