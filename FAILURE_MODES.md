# FAILURE_MODES

What the recorded negative results prove, and what they do not prove.
Numbers below are copied from files in this clone. This document does not add a run.

STATUS: WORK-IN-PROGRESS. These results do not close H1/H0.

## What is actually in the tree

| Record | File | What it is |
|---|---|---|
| H-T2 / H-T4 summary | `experiments/ht2_ht4_2026-08-16/verdict.json` | Means over n=3 seeds, synthetic windows (800 per row in `summary.tsv`) |
| H-D1 verdict | `experiments/hd1_2026-08-17/hd1_verdict.json` | `C_light_jepa` only, synthetic ImageReach |
| H-D1 raw rates | `experiments/hd1_2026-08-17/hd1_results_success.json`, `hd1_results_fail30.json` | Per-baseline rates; checkpoints are not in the repo |
| Chain-live demo | `experiments/p2_chain_live_2026-08-17/results.json` | ImageReach demo, n=15; not a preregistered hypothesis test |
| Pusht loss curve | prose in `STATUS.md` and `最小真实数据集方案_2026-08-17.md` | Host narrative. Dataset, windows, and checkpoint are not in the clone |

## H-T2 (projection-head capacity)

Preregistered support, from `HYPOTHESES.md`: the marginal-gain ratio Δ(128→256)/Δ(32→64) < 0.2 and d < 0.2, read as saturation below the backbone width 768.

`verdict.json` records ratio `4.8014`, d(128→256) `9.925`, `saturated: false`. Means (lower latent NLL treated as better in `HYPOTHESES.md`): width 32 `-3.505237`, 64 `-3.299737`, 128 `-2.718929`, 256 `-1.732249`, each with n=3.

That does prove, on this synthetic protocol, that the support criterion was not met, and that the recorded means got worse as the head widened.

That does not prove the head is never a bottleneck, that latent dim 64 is the right default on real video, or anything about H1/H0.

## H-T4 (EMA versus a synced target)

Preregistered support: the absolute relative gap |Δ| < 5% (EMA almost no benefit under a frozen backbone).

`verdict.json` records EMA mean `-3.299737`, sync mean `0.204362`, `delta_pct` `1714.6553`, `support: false`.

That does prove the "almost no benefit" criterion was not met on this synthetic frozen-backbone run. The recorded sync mean is much worse than the EMA mean.

That does not prove EMA is required for V-JEPA in general, for an unfrozen backbone, or on real data.

`experiments/ht2_ht4_2026-08-16/verdict.md` ends the H-T4 section with "H-T4 未支持（EMA 无收益）". The parenthesis restates the hypothesis that failed. It does not mean the run found EMA useless. The JSON means say the opposite.

## H-D1 (failure trajectories versus all-success)

Preregistered support: paired p < 0.05 and Cohen's h > 0.3, fail-injection better.

`hd1_verdict.json` records all-success `0.05`, fail30 `0.05`, paired p `1.0`, h `0.0`, Cliff's delta `0.0`, `support: false`, disclaimer "synthetic 2D env + synthetic failure-recovery semantics; chain-level evidence".

`scripts/analyze_hd1.py` scores only `C_light_jepa`. The raw JSON does not say both models were at 0.05. `B_frozen_cosine` is `0.2` in the all-success file and `0.25` in the fail30 file. Those two rates were not the preregistered comparison. Older prose that said both models were at 5% was wider than the script.

The p=1.0 in the verdict matches the script's branch for an all-zero paired difference. This commit does not recompute the vectors.

Two direction bugs in that script did not affect this tie, and are fixed for a later run:

- `wilcoxon` was called as `(success, fail)` with `alternative="greater"`, which tests success > fail. The preregistration tests fail > success. The tie branch returns before that call.
- `cohens_h` was called as `(success, fail)` while the field name and the support rule want a positive h when the fail rate is higher. Equal rates give h=0 either way, so `hd1_verdict.json` is unchanged.

A floor of 0.05 on the scored arm means a difference was not detected. It does not prove failure trajectories are worthless, and it does not prove the hypothesis is false on real robot data.

## Chain-live and the mock harness

`experiments/p2_chain_live_2026-08-17/results.json` records n=15, success rates A `0.0`, B `0.2`, C `0.06666666666666667`, and `stats_C_vs_B.image_reach.significant` `false`. The checkpoint field is null because the file is not in the repo. This is an engineering demo. It is not an H1 test.

`m3_mpc_eval.py` `smoke()` asserts that its own stub controllers order C > B > A on `MockReachEnv`. A rate printed by that smoke is a property of those stubs. It is not a measurement of a trained world model. There is no separate results file for that smoke under `experiments/`.

## What passing the offline contract does not prove

`bash scripts/verify_committed.sh` checks claim drift and the local chaos contract (ledger, cache, failpoint, and config assertions, plus the unit tests that script names). It does not prove NCCL, RDMA, a Kubernetes apply, robot control, or a route verdict. Weight, CUDA, and Docker groups in `verify_all.ps1` are absent from that command.

## Why this is still WORK-IN-PROGRESS

1. H1/H0 has no real-data run and no M4 verdict.
2. The three negatives above are synthetic-chain results with the limits just stated.
3. Real manipulation data and the official weights are not in the clone.
4. The pusht pipeline numbers are not re-checkable from this tree.
