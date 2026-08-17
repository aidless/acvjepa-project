#!/usr/bin/env python3
"""M3 评测骨架 × 真实权重「链路上真」集成演示（云端执行）

把训练好的 AC-V-JEPA（P2 动作条件后训练，结构化合成帧）作为 C 排序器接入
`m3_mpc_eval.py` 的评测骨架，在**同一渲染语义**的 ImageReach 环境中做 MPC 对比：

  A = 随机排序（无世界模型基线）
  B = 冻结特征·一步余弦（几何/运动学一步传播 + 学生编码器余弦；非学习式 WM）
  C = 轻量 JEPA（AC 模型：student 编码器 + 动作条件 GRU 预测器 rollout，
                 方差加权的「预测未来潜变量 → 目标状态潜变量」距离评分）

⚠️ 本演示是**工程验证**（机制/链路、真实权重、合成图像环境），不构成任何
预注册假设检验，结果不进入 HYPOTHESES 判定；仅报告给 M3 执行的「评测管线已通」。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m3_mpc_eval as M3
from ac_vjepa_core import ActionConditionedVJEPA
from vjepa_backbone import install_hf_vjepa2_encoder
import make_structured_p2_data as datagen  # 复用渲染语义（_render/_proprio/IMG/STEP）

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GOAL_DIST = 24.0
MAX_STEPS = 12


# ---------------------------------------------------------------------------
# ImageReachEnv（与训练数据同渲染语义；obs 按模型合约 [B,T,C,H,W]）
# ---------------------------------------------------------------------------

class ImageReachEnv:
    name = "image_reach"

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._pos = self._rng.uniform(60, datagen.IMG - 60, size=2)
        self._goal = self._rng.uniform(60, datagen.IMG - 60, size=2)
        while np.linalg.norm(self._goal - self._pos) < 120:
            self._goal = self._rng.uniform(60, datagen.IMG - 60, size=2)
        self._dist = np.stack([self._rng.uniform(60, datagen.IMG - 60, size=2)
                               for _ in range(3)])
        self._history = self._make_history()
        self._step_n = 0
        self._goal_img = None

    def _make_history(self) -> np.ndarray:
        """回溯 3 步反方向抖动 + 当前位置，生成 4 帧上下文（与数据生成一致的构造）。"""
        hist = [self._pos.copy()]
        p = self._pos.copy()
        for _ in range(3):
            p = np.clip(p - self._rng.normal(0.0, 6.0, size=2), 24, datagen.IMG - 24)
            hist.insert(0, p)
        return np.asarray(hist)

    def reset(self, seed: int) -> dict:
        self.__init__(seed)
        return self.reset_obs()

    def step(self, action: np.ndarray) -> tuple:
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        self._pos = np.clip(self._pos + a * datagen.STEP, 24, datagen.IMG - 24)
        self._step_n += 1
        self._history = np.vstack([self._history[1:], self._pos.copy()])
        dist = float(np.linalg.norm(self._pos - self._goal))
        success = dist < GOAL_DIST
        done = success or self._step_n >= MAX_STEPS
        info = {"dist": dist, "success": success, "violations": 0,
                "failure_type": "execution"}
        return self.reset_obs(), 1.0 if success else 0.0, done, info

    def reset_obs(self) -> dict:
        """返回 {obs: [1,4,3,384,384], proprio: [1,4,8]}（与训练批次合约一致）。"""
        fr = np.stack([datagen._render(p, self._goal, self._dist) for p in self._history])
        obs = torch.from_numpy(fr.transpose(0, 3, 1, 2))[None].float().to(DEVICE)
        prop = torch.from_numpy(np.stack([datagen._proprio(p) for p in self._history]))[
            None].float().to(DEVICE)
        return {"obs": obs, "proprio": prop}

    def goal_image(self) -> tuple:
        """{goal_img: [1,1,3,384,384], goal_prop: [1,1,8]}（agent 置于 goal 的目标图像）。"""
        if self._goal_img is None:
            fr = datagen._render(self._goal, self._goal, self._dist)
            self._goal_img = torch.from_numpy(fr.transpose(2, 0, 1))[None, None].float().to(DEVICE)
            # 与数据生成 future_proprio 一致：goal 处位置归一化
            self._goal_prop = torch.from_numpy(datagen._proprio(self._goal))[
                None, None].float().to(DEVICE)
        return self._goal_img, self._goal_prop

    def max_steps(self) -> int:
        return MAX_STEPS

    def success_dist(self) -> float:
        return GOAL_DIST


# ---------------------------------------------------------------------------
# 排序器（真实权重；候选批量编码）
# ---------------------------------------------------------------------------

def _goal_latent(module, env: ImageReachEnv):
    gi, gp = env.goal_image()
    return module.student_encoder.encode_future(gi, gp)[0, 0]  # [D]


class RealA(M3.RankingModule):
    name = "A_random"

    def rank(self, obs, candidates, rng, env=None):
        return rng.random(len(candidates))


class RealB(M3.RankingModule):
    """B：冻结特征·一步余弦（几何/运动学一步传播 + 学生编码器；非学习式 WM）。"""

    name = "B_frozen_cosine"

    def __init__(self, module: ActionConditionedVJEPA):
        self.module = module

    def rank(self, obs, candidates, rng, env: ImageReachEnv = None):
        goal = _goal_latent(self.module, env)
        n = len(candidates)
        frames = np.stack([
            datagen._render(np.clip(env._pos + np.clip(c, -1, 1) * datagen.STEP,
                                    24, datagen.IMG - 24), env._goal, env._dist)
            for c in candidates
        ])
        fr = torch.from_numpy(frames.transpose(0, 3, 1, 2))[:, None].float().to(DEVICE)  # [N,1,3,H,W]
        prop = torch.from_numpy(np.stack([datagen._proprio(np.clip(
            env._pos + np.clip(c, -1, 1) * datagen.STEP, 24, datagen.IMG - 24))
            for c in candidates]))[:, None].float().to(DEVICE)  # [N,1,8]
        lat = self.module.student_encoder.encode_future(fr, prop)[:, 0]  # [N,D]
        return torch.nn.functional.cosine_similarity(lat, goal, dim=-1).cpu().numpy()


class RealC(M3.RankingModule):
    """C：轻量 JEPA——动作条件预测器 rollout（GRU），方差加权目标距离评分。"""

    name = "C_light_jepa"

    def __init__(self, module: ActionConditionedVJEPA, horizon: int = 3):
        self.module = module
        self.horizon = horizon

    def rank(self, obs, candidates, rng, env: ImageReachEnv = None):
        state0 = self.module.student_encoder.encode_context(
            obs["obs"], obs["proprio"])  # [1,D]
        goal = _goal_latent(self.module, env)
        n = len(candidates)
        acts = np.zeros((n, self.horizon, 20), dtype=np.float32)
        acts[:, :, :2] = candidates[:, None, :]
        return _rollout_scores(self.module, state0, goal, acts)


def _rollout_scores(module, state0, goal, acts: np.ndarray) -> np.ndarray:
    """把所有候选作为 batch 维度（N,3,20）一次 rollout，返回逐候选评分 [N]。"""
    n = acts.shape[0]
    state = state0.expand(n, -1).contiguous()  # [N,D]
    action_tokens = module.action_tokenizer(torch.from_numpy(acts).to(DEVICE))
    hidden0 = state.unsqueeze(0)
    out, _ = module.rollout(action_tokens, hidden0)  # out: [N,T,D]
    future = state[:, None, :] + module.latent_delta(out)  # [N,T,D]
    logvar = module.log_variance_head(out).clamp(-8.0, 6.0)  # [N,T,D]
    goal_b = goal.unsqueeze(0).expand(n, -1)
    err = (future[:, -1] - goal_b) ** 2            # [N,D]
    var = logvar[:, -1, :].exp()                   # [N,D]
    score = -(err / var).sum(-1)                   # [N] 逐维方差加权距离
    return score.cpu().numpy()


class LiveController:
    """MPC 控制器：候选种子冻结 + 排序 + 执行最优动作后重规划。"""

    def __init__(self, ranking, n_candidates: int = 32, scale: float = 0.9):
        self.ranking = ranking
        self.n_candidates = n_candidates
        self.scale = scale

    def choose_action(self, obs, rng, env):
        cands = rng.uniform(-self.scale, self.scale, size=(self.n_candidates, 2))
        scores = self.ranking.rank(obs, cands, rng, env)
        return cands[int(np.argmax(scores))]


# ---------------------------------------------------------------------------
# 评测循环（复用 m3_mpc_eval 统计/聚合/结果表）
# ---------------------------------------------------------------------------

def run_live(module, n_rollouts: int, n_candidates: int, seed0: int = 7) -> dict:
    ctrls = {
        "A_random": LiveController(RealA(), n_candidates=n_candidates),
        "B_frozen_cosine": LiveController(RealB(module), n_candidates=n_candidates),
        "C_light_jepa": LiveController(RealC(module), n_candidates=n_candidates),
    }
    results = []
    for base, ctrl in ctrls.items():
        for k in range(n_rollouts):
            seed = seed0 + k
            env = ImageReachEnv(seed)
            cand_rng = np.random.default_rng(seed + 1)
            obs = env.reset(seed)
            conf_trace = []
            res = M3.RolloutResult(seed=seed, baseline=base, task=env.name)
            info = {"success": False, "dist": 10.0}
            while res.steps < env.max_steps():
                act = ctrl.choose_action(obs, cand_rng, env)
                obs, _, done, info = env.step(act)
                res.steps += 1
                d = info.get("dist", 10.0)
                conf = max(0.05, min(0.95, 1.0 - d / (3 * datagen.STEP)))
                conf_trace.append(conf)
                if done:
                    break
            res.success = bool(info.get("success", False))
            res.conf_trace = conf_trace
            res.ood_uncertainty = sum(1 for c in conf_trace if c < 0.4) / max(1, len(conf_trace))
            res.failure_type = "none" if res.success else "execution"
            results.append(res)
    agg = M3.EvalRunner.aggregate(results)
    succ = {}
    for r in results:
        succ.setdefault(r.baseline, []).append(1.0 if r.success else 0.0)
    # H-D1 配对需要：按 rollout 顺序的逐成功向量（不同模型同 env seed 序列）
    per_rollout = {b: [1.0 if r.success else 0.0 for r in results if r.baseline == b]
                   for b in succ}
    stats_cb = M3.stats_report({"image_reach": succ["C_light_jepa"]},
                               {"image_reach": succ["B_frozen_cosine"]})
    stats_ca = M3.stats_report({"image_reach": succ["C_light_jepa"]},
                               {"image_reach": succ["A_random"]})
    rows = M3.build_result_rows(agg, "image_reach")
    return {
        "success_rates": {b: float(np.mean(succ[b])) for b in succ},
        "per_rollout_success": per_rollout,
        "agg": agg,
        "rows": rows,
        "stats_C_vs_B": stats_cb,
        "stats_C_vs_A": stats_ca,
        "verdict_C_vs_B": M3._verdict_string(stats_cb),
        "verdict_C_vs_A": M3._verdict_string(stats_ca),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="M3 评测骨架 × 真实权重链路上真集成演示")
    ap.add_argument("--checkpoint", required=True, help="P2 last.pt")
    ap.add_argument("--weights-dir", required=True, help="vjepa2.1-vitb-fpc64-384 权重目录（含 model.safetensors）")
    ap.add_argument("--n-rollouts", type=int, default=15)
    ap.add_argument("--n-candidates", type=int, default=32)
    ap.add_argument("--out", required=True, help="结果 json 输出路径")
    args = ap.parse_args()

    print(f"[live] loading checkpoint {args.checkpoint}", flush=True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = payload["config"]
    module = ActionConditionedVJEPA(
        image_channels=cfg["image_channels"],
        proprio_dim=cfg["proprio_dim"],
        action_dim=cfg["action_dim"],
        latent_dim=cfg["latent_dim"],
        event_dim=cfg["event_dim"],
        max_horizon=cfg["max_horizon"],
        ema_momentum=cfg["ema_momentum"],
    )
    report = install_hf_vjepa2_encoder(
        module,
        latent_dim=cfg["latent_dim"],
        ckpt_path=str(Path(args.weights_dir) / "model.safetensors"),
        model_id=args.weights_dir,  # 本地 config.json，避免 HF 联网重试
        mode="frozen",
        img_size=cfg.get("init_img_size", 384),
    )
    print(f"[live] backbone load keys={report.loaded}", flush=True)
    module.load_state_dict(payload["model"], strict=True)
    module.eval().to(DEVICE)
    print(f"[live] checkpoint epoch={payload.get('epoch')} step={payload.get('global_step')}")

    with torch.no_grad():
        out = run_live(module, args.n_rollouts, args.n_candidates)

    print("\n=== 成功率 A/B/C ===", {k: round(v, 4) for k, v in out["success_rates"].items()})
    print("=== ECE A/B/C ===", {k: round(out["agg"][k]["ece"], 4) for k in out["agg"]})
    print("=== C vs B ===", json.dumps(
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in
         out["stats_C_vs_B"]["image_reach"].items()}, ensure_ascii=False))
    print("=== C vs A ===", json.dumps(
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in
         out["stats_C_vs_A"]["image_reach"].items()}, ensure_ascii=False))
    print("verdict C vs B:", out["verdict_C_vs_B"])
    print("verdict C vs A:", out["verdict_C_vs_A"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload_out = {
        "demo": "m3_live_demo",
        "checkpoint": args.checkpoint,
        "n_rollouts": args.n_rollouts,
        "n_candidates": args.n_candidates,
        "success_rates": out["success_rates"],
        "per_rollout_success": out["per_rollout_success"],
        "agg": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                    for kk, vv in v.items() if kk != "ece_time_1_3"}
                for k, v in out["agg"].items()},
        "ece_time_1_3": {k: [round(x, 6) for x in v["ece_time_1_3"]]
                         for k, v in out["agg"].items()},
        "stats_C_vs_B": out["stats_C_vs_B"],
        "stats_C_vs_A": out["stats_C_vs_A"],
        "verdicts": {"C_vs_B": out["verdict_C_vs_B"], "C_vs_A": out["verdict_C_vs_A"]},
        "disclaimer": "engineering chain validation only; NOT a preregistered hypothesis test",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload_out, f, ensure_ascii=False, indent=2)
    print(f"[live] results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())