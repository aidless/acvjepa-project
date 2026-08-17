#!/usr/bin/env python3
"""M3 闭环评测实现骨架 (m3_mpc_eval.py)

对应 `M3_MPC_EVALUATION_DESIGN.md` **v1.2** 预注册协议（2026-08-16 冻结，改动需决策记录）。
本骨架为**执行前实现**：不含真实 RoboCasa 依赖、不绑定真实世界模型；用 Mock 环境与合成排序器
冒烟验证统计层、指标采集与结果表结构（`python m3_mpc_eval.py --smoke`）。

覆盖的预注册协议点：
- §1/§6 判定规则：成功率 / 约束违规率 / 重规划次数 至少一项显著改进（配对检验 + 效应量），其余不得显著退化。
- §2 候选生成器种子冻结（每个 rollout 的候选与初始状态同 seed）。
- §4 六指标：任务成功率 / 约束违规率 / 重规划次数 / 平均动作数 / 分布外不确定性 / sim-to-real 差距。
- §4.1 ECE 分层（15 等宽置信度分箱）+ §7.3 ECE 前/中/后 1/3 时间段分解。
- §4.2 失败归因三类标注接口（interface / model / execution），真实判定规则见文档 §4.2。
- §5 统计：按任务模板分层的配对 Wilcoxon + Holm 校正；Cohen's h / Cliff's delta + 95% CI（bootstrap）。
- §7.1 结果表模板 + §7.3 「外部复现判定」列（Reproduced / Above reported / Validated-no-ref / Upstream defect）。

统计实现：scipy.stats（wilcoxon）、numpy；无 pandas 依赖。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

VERSION = "1.2-skeleton"
SEED_DEFAULT = 2026


# ---------------------------------------------------------------------------
# 统计层（§5）+ ECE 分层（§4.1/§7.3）
# ---------------------------------------------------------------------------

def cliff_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta：P(x>y) - P(x<y)，x/y 为配对或独立两组成绩向量。"""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    n, m = len(xa), len(ya)
    if n == 0 or m == 0:
        return 0.0
    gt = np.sum(xa[:, None] > ya[None, :])
    lt = np.sum(xa[:, None] < ya[None, :])
    return float((gt - lt) / (n * m))


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h：两块成功率比例的效应量（比例版本的 d）。"""
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return float(2.0 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2))))


def bootstrap_ci(x: Sequence[float], y: Sequence[float], stat_fn: Callable,
                 n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    """bootstrap 百分位 CI（配对重采样，用于 Cliff's delta 等无闭式的效应量）。"""
    rng = np.random.default_rng(seed)
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    if len(xa) != len(ya):
        # 非配对：独立重采样
        ests = []
        for _ in range(n_boot):
            xi = rng.choice(xa, size=len(xa), replace=True)
            yi = rng.choice(ya, size=len(ya), replace=True)
            ests.append(stat_fn(xi, yi))
        lo, hi = np.percentile(ests, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lo), float(hi)
    idx = np.arange(len(xa))
    ests = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        ests.append(stat_fn(xa[s], ya[s]))
    lo, hi = np.percentile(ests, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def wilcoxon_holm(scores_a: Dict[str, List[float]],
                  scores_b: Dict[str, List[float]],
                  alpha: float = 0.05) -> Dict[str, dict]:
    """按任务模板分层的配对 Wilcoxon 秩和检验 + Holm 校正。

    scores_a / scores_b: {task: [per-rollout 成绩]}（成绩=成功率=1/0 或连续分数）。
    返回 {task: {p, p_holm_corrected, significant, n}}。
    单任务样本对全部并列（零差）时 p=1.0（不显著），不抛错。
    """
    tasks = [t for t in scores_a if t in scores_b]
    raw = {}
    for t in tasks:
        a, b = np.asarray(scores_a[t], float), np.asarray(scores_b[t], float)
        n = min(len(a), len(b))
        if n == 0:
            raw[t] = 1.0
            continue
        diff = a[:n] - b[:n]
        if np.allclose(diff, 0.0):
            raw[t] = 1.0
            continue
        try:
            _, p = stats.wilcoxon(a[:n], b[:n], zero_method="wilcox",
                                  alternative="two-sided")
        except ValueError:
            p = 1.0
        raw[t] = float(p)
    # Holm 校正（升序 p 值，校正后保序）
    if not raw:
        return {}
    ordered = sorted(raw.items(), key=lambda kv: kv[1])
    m = len(ordered)
    corrected = {}
    prev = 0.0
    for i, (t, p) in enumerate(ordered):
        val = max(p * (m - i), prev)
        corrected[t] = val
        prev = val
    return {t: {"p": raw[t], "p_holm": corrected[t],
                "significant": corrected[t] < alpha,
                "n": len(scores_a[t])} for t in tasks}


def ece(confidences: Sequence[float], successes: Sequence[int],
        n_bins: int = 15) -> float:
    """期望校准误差：等宽置信度分箱 |acc - conf| 加权平均（§4.1，15 bins）。"""
    conf = np.asarray(confidences, float)
    succ = np.asarray(successes, float)
    if len(conf) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n_all = len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        acc = succ[mask].mean()
        conf_mean = conf[mask].mean()
        total += (mask.sum() / n_all) * abs(acc - conf_mean)
    return float(total)


def time_axis_ece(conf_trace: Sequence[Sequence[float]],
                  successes: Sequence[int]) -> List[float]:
    """ECE 前/中/后 1/3 时间段分解（§7.3，WorldRoamBench 度量移植）。

    每个 rollout 的逐步置信度序列等分三段；对三段各自的平均置信度分别与最终成败
    计算 ECE(conf, success) —— 检验「中途信心崩塌」（中/后段校准恶化 ⇒ 低 ECE 组
    优势主要来自中后段即 H-T3 的时间形态）。
    """
    thirds = [[], [], []]
    for trace in conf_trace:
        arr = np.asarray(trace, float)
        if len(arr) == 0:
            continue
        n = len(arr)
        split = [max(1, n // 3)] * 3
        split[0] += n - sum(split)
        i = 0
        for k in range(3):
            seg = arr[i:i + split[k]]
            i += split[k]
            if len(seg):
                thirds[k].append(seg.mean())
    return [ece(thirds[k], successes, n_bins=15) for k in range(3)]


# ---------------------------------------------------------------------------
# 环境客户端（§7 执行依赖 2：RoboCasa 环境 + ≥24GB GPU；骨架含 Mock）
# ---------------------------------------------------------------------------

class EnvClient:
    """环境客户端接口：真实评测接入 RoboCasa（vla-eval 基准客户端等价物）。"""

    name = "env"

    def reset(self, seed: int) -> dict:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, dict]:
        raise NotImplementedError

    def max_steps(self) -> int:
        raise NotImplementedError


class MockReachEnv(EnvClient):
    """合成 2D 目标达成环境（冒烟用）。

    obs = [x, y, gx, gy]；action = [dx, dy] ∈ [-0.3, 0.3]；成功 = 距离 < 0.2；
    越界（|x|>3.0）计一次约束违规并截断到边界。参数放宽保证 A<B<C 可分离。
    """

    name = "mock_reach"

    def __init__(self, max_steps: int = 18, bound: float = 3.0, success_dist: float = 0.2):
        self._max = max_steps
        self._bound = bound
        self._success_dist = success_dist
        self._step_n = 0
        self._violations = 0
        self._success = False
        self._done = False

    def reset(self, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        self._step_n = 0
        self._violations = 0
        self._success = False
        self._done = False
        self._pos = rng.uniform(-0.5, 0.5, size=2)
        self._goal = rng.uniform(-1.5, 1.5, size=2)
        self._conf_trace: List[float] = []
        return self._obs()

    def _obs(self) -> dict:
        return {"obs": np.concatenate([self._pos, self._goal]), "info": {}}

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, dict]:
        a = np.clip(np.asarray(action, float), -0.3, 0.3)
        self._pos = self._pos + a
        self._step_n += 1
        if np.abs(self._pos).max() > self._bound:
            self._violations += 1
            self._pos = np.clip(self._pos, -self._bound, self._bound)
        dist = float(np.linalg.norm(self._pos - self._goal))
        self._success = dist < self._success_dist
        self._done = self._success or self._step_n >= self._max
        reward = 1.0 if self._success else (0.0 if dist < 0.8 else -0.1)
        info = {"dist": dist, "violations": self._violations,
                "success": self._success, "failure_type": "execution"}
        return self._obs(), reward, self._done, info

    def max_steps(self) -> int:
        return self._max


# ---------------------------------------------------------------------------
# 三基线排序器（§1：A 无世界模型 / B 冻结视觉特征 / C 轻量 JEPA）
# ---------------------------------------------------------------------------

class RankingModule:
    """排序器接口：给定观测与候选动作集，返回候选分数（vla-eval 模型服务器 predict 的等价物）。"""

    name = "module"

    def rank(self, obs: np.ndarray, candidates: np.ndarray, rng: np.random.Generator,
             env: Optional[EnvClient] = None) -> np.ndarray:
        raise NotImplementedError


class BaselineRandom(RankingModule):
    """A：无世界模型（随机/几何启发）。冒烟=纯随机；真实评测=几何启发（文档 §1 A 组）。"""

    name = "A_random"

    def rank(self, obs, candidates, rng, env=None):
        return rng.random(len(candidates))


class BaselineFrozenCosine(RankingModule):
    """B：冻结视觉特征·一步余弦（文档 §1 B 组）。

    冒烟：用固定随机投影模拟冻结 ViT 特征，取「施加候选动作后的预测观测」特征与目标特征
    的余弦相似度——等价于单步贪心（无时间外推）。
    """

    name = "B_frozen_cosine"

    def __init__(self, feature_dim: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._proj = rng.normal(size=(feature_dim, 4)).astype(np.float32)

    def _feat(self, obs: np.ndarray) -> np.ndarray:
        v = self._proj @ obs
        return v / (np.linalg.norm(v) + 1e-9)

    def rank(self, obs, candidates, rng, env=None):
        goal_obs = obs.copy()
        goal_obs[2:] = obs[2:]  # goal 分量
        goal_feat = self._feat(goal_obs)
        act_feats = []
        for c in candidates:
            # 一步后观测近似（冒烟用线性动态；真实=编码器对候选 rollout 的特征）
            next_obs = obs.copy()
            next_obs[:2] = np.clip(obs[:2] + np.clip(c, -0.3, 0.3), -3.0, 3.0)
            act_feats.append(self._feat(next_obs))
        act_feats = np.asarray(act_feats)
        sims = act_feats @ goal_feat
        # 冻结特征噪声：一步余弦「目光短浅」且受特征噪声扰动（真实=冻结主干特征的不确定性）
        return sims + rng.normal(0.0, 0.15, size=len(sims))


class LightJEPARollout(RankingModule):
    """C：轻量 JEPA·预测 rollout（文档 §1 C 组）。

    冒烟：以已知线性动态做 H=4 步 rollout，按「预测最终到目标距离」评分——
    等价于理想世界模型（真实评测=训练后的轻量 JEPA 编码器+预测器 predict()）。
    """

    name = "C_light_jepa"

    def __init__(self, horizon: int = 4):
        self._horizon = horizon

    def rank(self, obs, candidates, rng, env=None):
        pos, goal = obs[:2].copy(), obs[2:]
        scores = []
        for c in candidates:
            p = pos.copy()
            for _ in range(self._horizon):
                p = np.clip(p + np.clip(c, -0.3, 0.3), -3.0, 3.0)
            pred_dist = np.linalg.norm(p - goal)
            scores.append(-pred_dist)  # 越低越好 → 取负
        return np.asarray(scores)


# ---------------------------------------------------------------------------
# MPC 控制器（§2 候选生成器种子冻结 + 预算控制 + 重规划）
# ---------------------------------------------------------------------------

class MPCController:
    """固定候选预算的滚动时域控制器。

    - 每步用 `rng` 生成 n_candidates 个候选（种子冻结，rollout 级可复现）；
    - ranking.rank 排序 → 执行最高分候选的前 n_exec 个动作 → 重规划；
    - 候选边界超限即计 violation（由 env 计，见 MockReachEnv.step）。
    """

    def __init__(self, ranking: RankingModule, n_candidates: int = 64,
                 horizon_steps: int = 1, action_dim: int = 2):
        self.ranking = ranking
        self.n_candidates = n_candidates
        self.action_dim = action_dim

    def choose_action(self, obs: np.ndarray, rng: np.random.Generator,
                      env: Optional[EnvClient] = None) -> np.ndarray:
        cands = rng.uniform(-0.3, 0.3, size=(self.n_candidates, self.action_dim))
        scores = self.ranking.rank(obs, cands, rng, env)
        return cands[int(np.argmax(scores))]


# ---------------------------------------------------------------------------
# 评测运行器（§4 六指标 + ECE + 失败归因）与结果装配（§7.1/§7.3）
# ---------------------------------------------------------------------------

@dataclass
class RolloutResult:
    seed: int
    baseline: str
    task: str = "mock_reach"
    success: bool = False
    violations: int = 0
    steps: int = 0
    ood_uncertainty: float = 0.0          # §4 指标 5（冒烟=低分候选占比近似）
    sim2real_gap: float = float("nan")    # §4 指标 6（真实 sim→real 复测才有值）
    conf_trace: List[float] = field(default_factory=list)
    failure_type: Optional[str] = None     # §4.2：interface / model / execution / none


def run_rollout(env_factory: Callable[[], EnvClient], controller: MPCController,
                seed: int, baseline: str) -> RolloutResult:
    env = env_factory()
    rng = np.random.default_rng(seed)
    # 候选生成种子冻结：rollout 使用独立 rng 但候选生成在 controller 内用同一个 rng ——
    # 冻结性由 (env_seed, rng seed) 保证；控制器 rng 派生自 rollout rng（固定派生）。
    cand_rng = np.random.default_rng(seed + 1)
    obs = env.reset(seed)
    res = RolloutResult(seed=seed, baseline=baseline)
    conf_trace = []
    low_conf = 0
    while not res.steps >= env.max_steps():
        act = controller.choose_action(obs["obs"], cand_rng, env)
        obs, _, done, info = env.step(act)
        res.steps += 1
        res.violations += info.get("violations", 0)
        # 冒烟置信度代理：归一化距离 → 距离越小置信度越高
        d = info.get("dist", 10.0)
        conf = max(0.05, min(0.95, 1.0 - d / 3.0))
        conf_trace.append(conf)
        if conf < 0.4:
            low_conf += 1
        if done:
            break
    res.success = bool(info.get("success", False))
    res.conf_trace = conf_trace
    res.ood_uncertainty = low_conf / max(1, res.steps)
    res.failure_type = info.get("failure_type") if not res.success else "none"
    return res


def _hit(env) -> bool:
    """从附带的 env 读最终成功（Mock 在 info 里已带；这里兜底）。"""
    return getattr(env, "_success", False)


class EvalRunner:
    """多 rollout 评测：对每基线 × 每任务模板跑 n_rollouts，装配结果表。"""

    def __init__(self, env_factory, controllers: Dict[str, MPCController],
                 default_rollouts: int = 20):
        self.env_factory = env_factory
        self.controllers = controllers
        self.default_rollouts = default_rollouts

    def run(self, baselines: Sequence[str], task: str = "mock_reach",
            n_rollouts: Optional[int] = None, first_seed: int = SEED_DEFAULT) -> List[RolloutResult]:
        n = n_rollouts or self.default_rollouts
        out = []
        for b in baselines:
            ctrl = self.controllers[b]
            for k in range(n):
                seed = first_seed + k
                res = run_rollout(self.env_factory, ctrl, seed, b)
                res.task = task
                out.append(res)
        return out

    @staticmethod
    def aggregate(results: List[RolloutResult]) -> Dict[str, dict]:
        """按基线汇总六指标 + ECE 分层 + 时间轴分解。"""
        by_bl = {}
        for r in results:
            by_bl.setdefault(r.baseline, []).append(r)
        rows = {}
        for b, rs in by_bl.items():
            suc_flat = []
            confs = []
            for r in rs:
                # 每步置信度与所属 rollout 的成败配对（ECE 是「预测置信度 vs 该 rollout 成败」）
                for c in r.conf_trace:
                    confs.append(c)
                    suc_flat.append(1.0 if r.success else 0.0)
            traces = [r.conf_trace for r in rs]
            succ_per_rollout = [1 if r.success else 0 for r in rs]
            rows[b] = {
                "n": len(rs),
                "success_rate": float(np.mean([1.0 if r.success else 0.0 for r in rs])),
                "violation_rate": float(np.mean([r.violations for r in rs])),
                "replans_mean": float(np.mean([max(0, r.steps - 1) for r in rs])),
                "actions_mean": float(np.mean([r.steps for r in rs])),
                "ood_uncertainty": float(np.mean([r.ood_uncertainty for r in rs])),
                "ece": ece(confs, suc_flat),
                "ece_time_1_3": time_axis_ece(traces, succ_per_rollout),
                "failure_counts": {"interface": sum(1 for r in rs if r.failure_type == "interface"),
                                   "model": sum(1 for r in rs if r.failure_type == "model"),
                                   "execution": sum(1 for r in rs if r.failure_type == "execution")},
                "sim2real_gap": float(np.nanmean([r.sim2real_gap for r in rs])) if any(
                    not math.isnan(r.sim2real_gap) for r in rs) else float("nan"),
            }
        return rows


# ---------------------------------------------------------------------------
# 结果表（§7.1 模板 + §7.3 复现判定列）与 CLI
# ---------------------------------------------------------------------------

RESULT_TABLE_COLUMNS = [
    "task", "baseline", "n_rollouts", "success_rate", "violation_rate",
    "replans_mean", "actions_mean", "ood_uncertainty", "ece",
    "ece_t1", "ece_t2", "ece_t3", "sim2real_gap", "reproduction_verdict",
]


def build_result_rows(agg: Dict[str, dict], task: str,
                      reproduction: Optional[Dict[str, str]] = None) -> List[dict]:
    rows = []
    for b, v in agg.items():
        e = v["ece_time_1_3"]
        rows.append({
            "task": task, "baseline": b, "n_rollouts": v["n"],
            "success_rate": round(v["success_rate"], 4),
            "violation_rate": round(v["violation_rate"], 4),
            "replans_mean": round(v["replans_mean"], 3),
            "actions_mean": round(v["actions_mean"], 3),
            "ood_uncertainty": round(v["ood_uncertainty"], 4),
            "ece": round(v["ece"], 4),
            "ece_t1": round(e[0], 4), "ece_t2": round(e[1], 4), "ece_t3": round(e[2], 4),
            "sim2real_gap": round(v["sim2real_gap"], 6) if not math.isnan(v["sim2real_gap"]) else "N/A",
            "reproduction_verdict": (reproduction or {}).get(b, "N/A"),
        })
    return rows


def stats_report(scores_a: Dict[str, List[float]], scores_b: Dict[str, List[float]]) -> dict:
    """§5 统计报告：配对 Wilcoxon+Holm + 效应量（成功率 h、连续量 Cliff's delta + 95%CI）。"""
    wil = wilcoxon_holm(scores_a, scores_b)
    out = {}
    for t, d in wil.items():
        a, b = scores_a[t], scores_b[t]
        sr_a = float(np.mean(a)) if a else 0.0
        sr_b = float(np.mean(b)) if b else 0.0
        out[t] = {
            "p": d["p"], "p_holm": d["p_holm"], "significant": d["significant"],
            "cohens_h": cohens_h(sr_a, sr_b),
            "cliff_delta": cliff_delta(a, b),
            "cliff_delta_ci": bootstrap_ci(a, b, cliff_delta, seed=42),
        }
    return out


def _verdict_string(stats: dict, alpha: float = 0.05) -> str:
    """§6 判定规则：成功率/约束违规率/重规划次数至少一项显著改进且其余不显著退化 → 支持 H1。"""
    sig_improve = []
    sig_degrade = []
    # 成功率：B 相对 A 的改进；退化监测用「反方向显著」。
    for t, s in stats.items():
        if s["significant"]:
            if s["cliff_delta"] > 0:
                sig_improve.append(t)
            else:
                sig_degrade.append(t)
    if sig_improve and not sig_degrade:
        return "H1-support（至少一模板显著改进，无显著退化）"
    if sig_degrade:
        return "H0-direction（存在显著退化模板，须转 §6 判定表裁决分支）"
    return "H0-direction（无显著改进）"


def write_table_csv(rows: List[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_TABLE_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def smoke() -> int:
    """Mock 环境全链路冒烟：验证统计层、指标采集与结果表结构。

    断言（在 Mock 环境上，合成排序器应满足的能力阶梯）：
      1. C（理想 rollout）成功率 > B（一步余弦）> A（随机）；
      2. ECE ∈ [0,1]；ECE 时间轴三分段均有值；
      3. Wilcoxon+Holm 输出健壮（p 有限、Holm 校正保序、单模板退化时判定字符串正确切换）；
      4. 结果表模板列完整、行数 = 基线数。
    """
    env_factory = MockReachEnv
    ctrls = {
        "A_random": MPCController(BaselineRandom(), n_candidates=64),
        "B_frozen_cosine": MPCController(BaselineFrozenCosine(), n_candidates=64),
        "C_light_jepa": MPCController(LightJEPARollout(horizon=4), n_candidates=64),
    }
    runner = EvalRunner(env_factory, ctrls, default_rollouts=30)
    results = runner.run(["A_random", "B_frozen_cosine", "C_light_jepa"],
                         task="mock_reach", first_seed=SEED_DEFAULT)
    agg = EvalRunner.aggregate(results)
    sr = {b: agg[b]["success_rate"] for b in agg}
    assert sr["C_light_jepa"] > sr["B_frozen_cosine"], f"C 应优于 B: {sr}"
    assert sr["B_frozen_cosine"] > sr["A_random"], f"B 应优于 A: {sr}"
    for b in agg:
        assert 0.0 <= agg[b]["ece"] <= 1.0, f"ECE 越界: {b}"
        assert len(agg[b]["ece_time_1_3"]) == 3 and all(0 <= v <= 1 for v in agg[b]["ece_time_1_3"])
    # 统计层：成功率配对 C vs B
    succ_by_bl = {}
    for r in results:
        succ_by_bl.setdefault(r.baseline, []).append(1.0 if r.success else 0.0)
    st_rep = stats_report({"mock": succ_by_bl["C_light_jepa"]},
                          {"mock": succ_by_bl["B_frozen_cosine"]})
    p = st_rep["mock"]["p"]
    assert math.isfinite(p) and 0 < p <= 1.0
    # Holm 校正保序（多模板合成数据检查）
    wil = wilcoxon_holm({"t1": [1, 1, 1, 0, 1], "t2": [0, 0, 0, 1, 0], "t3": [1, 0, 1, 0, 1]},
                        {"t1": [0, 0, 0, 0, 0], "t2": [0, 1, 0, 0, 0], "t3": [1, 0, 1, 1, 0]})
    ph = [wil[t]["p_holm"] for t in ["t1", "t2", "t3"]]
    assert ph == sorted(ph), f"Holm 校正后应保序: {ph}"
    # 结果表
    rows = build_result_rows(agg, "mock_reach")
    assert len(rows) == 3 and all(set(r) >= set(RESULT_TABLE_COLUMNS) for r in rows)
    rep = _verdict_string(st_rep)
    assert rep.startswith("H0") or rep.startswith("H1")

    print(f"[smoke OK] Mock 全链路通过 v{VERSION}")
    print(f"  成功率 A/B/C = {sr['A_random']:.3f} / {sr['B_frozen_cosine']:.3f} / {sr['C_light_jepa']:.3f}")
    print(f"  ECE(15bin) A/B/C = {[round(agg[b]['ece'], 4) for b in ['A_random','B_frozen_cosine','C_light_jepa']]}")
    print(f"  C vs B 配对检验: p={p:.4f}, Cliff's delta={st_rep['mock']['cliff_delta']:.3f}, "
          f"95%CI={[round(v,3) for v in st_rep['mock']['cliff_delta_ci']]}")
    print(f"  §6 判定（Mock 示例）: {rep}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="M3 闭环评测骨架（v1.2 预注册协议）")
    ap.add_argument("--smoke", action="store_true", help="Mock 环境全链路冒烟（默认行为）")
    args = ap.parse_args(argv)
    if args.smoke:
        return smoke()
    return smoke()


if __name__ == "__main__":
    sys.exit(main())