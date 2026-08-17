#!/usr/bin/env python3
"""M3 评测包 → vla-evaluation-harness 对接骨架（A22 结论落地为代码）

背景：A22 评估结论（决策记录 2026-08-16）——`allenai/vla-evaluation-harness` 已集成
RoboCasa（待首复现），M3 评测包可注册为其「模型服务器（三基线排序器）+ RoboCasa 基准客户端」
集成；统计层（ECE 分层 / Wilcoxon+Holm / 失败归因 / M3 六指标）**仍自建**（「分层在 harness 之上」）。

本骨架实现两件「对接面」：
1. `M3ModelServer` —— vla-eval 模型服务器契约的等价物：单文件、实现 `predict()`，
   内部把 A/B/C 三基线排序器（复用 `m3_mpc_eval`）包装成统一接口；
   （vla-eval 实际要求「模型服务器 = 单文件 uv 脚本 + 实现 predict」；本骨架即为该脚本的雏形。）
2. `RoboCasaClientStub` —— vla-eval 基准客户端的四方法契约 + RoboCasa 环境映射说明
   （真实 RoboCasa 需独立环境：Python≤3.10 + IsaacGym/OmniIsaac + 磁盘 ≥50GB，见 A27）。

统计层（M3 §4/§5/§7.3）不在此封装：由 `m3_mpc_eval.py` 在 harness 之上运行。

冒烟：`python scripts/m3_vla_eval_dock.py --smoke` —— 用 Mock/ImageReach 风格的合成环境
验证 predict() 契约与结果表输出（无需 GPU、无需 RoboCasa）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import m3_mpc_eval as M3


# ---------------------------------------------------------------------------
# 1. 模型服务器（vla-eval predict 契约）
# ---------------------------------------------------------------------------

class M3ModelServer:
    """vla-eval 模型服务器雏形：predict(obs, candidates) -> scores。

    契约对齐 vla-eval「模型服务器 = 单文件 uv 脚本 + predict()」；
    内部选择基线（A/B/C 之一），obs/candidates 为 numpy；返回逐候选分数（越高越好）。
    """

    def __init__(self, baseline: str = "C", ranking: Optional[M3.RankingModule] = None,
                 rng_seed: int = 0):
        if baseline not in ("A", "B", "C"):
            raise ValueError(f"baseline 须为 A/B/C，got {baseline}")
        self.baseline = baseline
        self._rng = np.random.default_rng(rng_seed)
        self.ranking = ranking if ranking is not None else self._default_ranking(baseline)

    @staticmethod
    def _default_ranking(baseline: str) -> M3.RankingModule:
        if baseline == "A":
            return M3.BaselineRandom()
        if baseline == "B":
            return M3.BaselineFrozenCosine()
        return M3.LightJEPARollout(horizon=4)

    def predict(self, obs: np.ndarray, candidates: np.ndarray,
                env: Optional[object] = None) -> np.ndarray:
        """返回与 candidates 等长的分数数组（vla-eval 模型侧只要求 predict）。"""
        return self.ranking.rank(obs, candidates, self._rng, env)

    def model_config(self) -> dict:
        return {"baseline": self.baseline, "m3_version": M3.VERSION,
                "note": "statistics layer runs above the harness (M3 4/5/7.3)"}


# ---------------------------------------------------------------------------
# 2. 基准客户端（vla-eval 基准四方法契约 + RoboCasa 映射说明）
# ---------------------------------------------------------------------------

class RoboCasaClientStub:
    """vla-eval 基准客户端契约 + RoboCasa 映射。

    vla-eval 基准侧需实现四个方法（见 allenai/vla-evaluation-harness docs）：
      reset(task_cfg, seed) -> obs
      step(action) -> obs, reward, done, info
      get_metrics() -> dict
      (per-episode artifact 记录由 harness 负责)

    RoboCasa 映射（M3 §3 任务子集：早餐台整理相关拾取/门抽屉/桌面整理，10–30 模板）：
      任务模板 + 布局 + 对象实例隔离（DATA_MANIFEST 纪律）；候选动作集与 MPC 预算同 M3 §2。
    注意：真实 RoboCasa 需要独立环境（Python≤3.10 + IsaacGym + 磁盘≥50GB）——见 BACKLOG A27。
    """

    name = "robocasa"

    def __init__(self, task_templates: Optional[List[str]] = None):
        self.task_templates = task_templates or ["place_on_shelf", "open_drawer"]

    def reset(self, task_cfg: dict, seed: int) -> np.ndarray:
        raise NotImplementedError(
            "真实 RoboCasa 环境未部署（A27 硬阻塞：Python≤3.10 + IsaacGym + 磁盘）。"
            " 本 stub 仅承载契约；冒烟用 M3.MockReachEnv 代替。"
        )

    def step(self, action: np.ndarray):
        raise NotImplementedError

    def get_metrics(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# 冒烟：验证 predict 契约 + 三基线在合成环境上的可跑性（无 GPU/RoboCasa）
# ---------------------------------------------------------------------------

def smoke() -> int:
    import m3_mpc_eval as M3E
    env_factory = M3E.MockReachEnv

    servers = {b: M3ModelServer(baseline=b) for b in ("A", "B", "C")}
    env = env_factory()
    obs = env.reset(2026)["obs"]
    cands = np.random.default_rng(1).uniform(-0.3, 0.3, size=(16, 2))
    for b, srv in servers.items():
        scores = srv.predict(obs, cands, env)
        assert scores.shape == (16,), f"{b} 分数形状错误: {scores.shape}"
        assert np.isfinite(scores).all(), f"{b} 含非有限分数"

    # 结果表契约：server 本身不产统计层；统计由 m3_mpc_eval 跑（这里演示其可用性）
    ctrl = {b: M3E.MPCController(srv.ranking, n_candidates=16) for b, srv in servers.items()}
    runner = M3E.EvalRunner(env_factory, ctrl, default_rollouts=10)
    results = runner.run(["A", "B", "C"], task="mock_reach", first_seed=2026)
    agg = M3E.EvalRunner.aggregate(results)
    rows = M3E.build_result_rows(agg, "mock_reach")
    print(f"[dock smoke OK] predict 契约 + 三基线合成环境可跑 v{M3E.VERSION}")
    for r in rows:
        print(" ", r["baseline"], "success=", r["success_rate"], "ece=", r["ece"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="M3 → vla-eval 对接骨架（A22 落地）")
    ap.add_argument("--smoke", action="store_true", help="合成环境契约冒烟（默认行为）")
    args = ap.parse_args()
    return smoke()


if __name__ == "__main__":
    sys.exit(main())