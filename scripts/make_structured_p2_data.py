#!/usr/bin/env python3
"""生成结构化合成 B 层风格窗口数据集（P2 动作条件后训练用，工程验证用途）。

⚠️ 用途声明：本数据集是**工程验证链路**数据（结构化合成帧 + 可解释动作），
**不是**任何预注册假设检验的实验数据；链路验证结果不进入 HYPOTHESES 判定。

场景语义（与 `docs` 保持一致，供评测侧 mock 环境复用）：
- 384×384 RGB：深色背景；**红色方块=agent**；**绿色方块=goal**；3 个灰色干扰方块（静止）；
- 动作 = agent 位移方向（归一化 [-1,1]，幅度×18px/步；越界投影回画布）；
- 数据策略 = 带噪声的直线趋向 goal（模型可学到「动作→agent 移动」的映射）。

窗口格式（与 `train_ac_vjepa_ddp.py` 的 REQUIRED_KEYS 完全一致）：
  context_video [4,3,384,384] / context_proprio [4,8] / future_video [3,3,384,384]
  future_proprio [3,8] / executed_actions [3,20] / future_events [3,4]
动作 2 维 pad 到 20（ActionTokenizer 的 action_dim=20 默认）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

IMG = 384
BLOCK = 28
STEP = 18


def _render(pos: np.ndarray, goal: np.ndarray, distractor: np.ndarray) -> np.ndarray:
    """渲染一帧 [384,384,3] uint8。pos/goal/distractor 为 (x, y) 像素坐标。"""
    frame = np.full((IMG, IMG, 3), (28, 30, 52), dtype=np.uint8)

    def put_rect(center: np.ndarray, size: int, color) -> None:
        x, y = int(round(center[0])), int(round(center[1]))
        x0, x1 = max(0, x - size // 2), min(IMG, x + size // 2)
        y0, y1 = max(0, y - size // 2), min(IMG, y + size // 2)
        frame[y0:y1, x0:x1] = color

    for d in distractor:  # 灰色干扰物（静止）
        put_rect(d, 20, (96, 96, 96))
    put_rect(goal, BLOCK, (34, 177, 76))    # 绿色 goal
    put_rect(pos, BLOCK, (227, 66, 52))     # 红色 agent（最上层）
    return frame


def _proprio(pos: np.ndarray) -> np.ndarray:
    v = np.zeros(8, dtype=np.float32)
    v[:2] = pos / IMG * 2.0 - 1.0
    return v


def make_windows(n: int, seed: int = 2026) -> list[dict]:
    rng = np.random.default_rng(seed)
    windows = []
    for _ in range(n):
        pos = rng.uniform(60, IMG - 60, size=2)
        goal = rng.uniform(60, IMG - 60, size=2)
        while np.linalg.norm(goal - pos) < 120:
            goal = rng.uniform(60, IMG - 60, size=2)
        distractor = rng.uniform(60, IMG - 60, size=(3, 2))

        # 未来 3 步动作（带噪声直线趋向 goal；幅度 STEP 像素）
        acts = []
        for _ in range(3):
            delta = np.clip(goal - pos, -STEP, STEP)
            if np.linalg.norm(delta) > 1e-6:
                unit = delta / np.linalg.norm(delta)
                noisy = unit + rng.normal(0.0, 0.25, size=2)
                noisy = noisy / (np.linalg.norm(noisy) + 1e-9)
            else:
                noisy = rng.normal(0.0, 0.3, size=2)
                noisy = noisy / (np.linalg.norm(noisy) + 1e-9)
            acts.append(noisy)

        # 回溯生成 4 帧上下文（从 pos 往前推 4 步的逆运动）
        conj = [pos]
        p = pos.copy()
        for a in reversed(acts):
            p = np.clip(p - a * STEP, 24, IMG - 24)
            conj.insert(0, p)
        ctx_positions = conj[:4]
        assert len(ctx_positions) == 4

        # 未来 3 帧（正运动）
        fut_positions = [pos]
        p = pos.copy()
        for a in acts:
            p = np.clip(p + a * STEP, 24, IMG - 24)
            fut_positions.append(p)
        fut_positions = fut_positions[1:]

        context_video = torch.from_numpy(
            np.stack([_render(q, goal, distractor) for q in ctx_positions]).transpose(0, 3, 1, 2)
        ).to(torch.uint8)
        future_video = torch.from_numpy(
            np.stack([_render(q, goal, distractor) for q in fut_positions]).transpose(0, 3, 1, 2)
        ).to(torch.uint8)

        ctx_prop = np.stack([_proprio(q) for q in ctx_positions])
        fut_prop = np.stack([_proprio(q) for q in fut_positions])

        # 动作 2 维→20 维 pad
        act2 = np.asarray(acts, dtype=np.float32)
        actions = np.zeros((3, 20), dtype=np.float32)
        actions[:, :2] = act2

        windows.append({
            "context_video": context_video,
            "context_proprio": torch.from_numpy(ctx_prop),
            "future_video": future_video,
            "future_proprio": torch.from_numpy(fut_prop),
            "executed_actions": torch.from_numpy(actions),
            "future_events": torch.from_numpy(rng.integers(0, 2, (3, 4)).astype(np.float32)),
        })
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="p2_structured_data")
    parser.add_argument("--n", type=int, default=600)
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    entries = []
    windows = make_windows(args.n)
    for i, w in enumerate(windows):
        path = root / f"window_{i:04d}.pt"
        torch.save(w, path)
        entries.append({"path": str(path.resolve())})
    manifest.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(windows)} windows to {root}")


if __name__ == "__main__":
    main()