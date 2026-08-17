#!/usr/bin/env python3
"""LeRobot 数据集 → AC-VJEPA 窗口格式转换器（真实数据管线）。

把 LeRobot v3 格式（parquet + mp4 视频）转为 `train_p1_domain_adapt.py` /
`train_ac_vjepa_ddp.py` 需要的窗口 .pt（context/future video+proprio、executed_actions、
future_events）。首个目标：`lerobot/pusht`（96×96、state/action 2 维）。

用途：**真实（非合成）数据链路冒烟**——证明真实视频能流过 AC-VJEPA 训练管线。
注意：pusht 为 2D 推块域，非厨房域——适合管线验证，不适合 H-T5 厨房域判定。

窗口契约（与 make_demo_ddp_data.py / WindowEpisodeDataset 一致）：
  context_video   [4, 3, H, W]   （已归一化 float32：(x/255 - mean)/std，ImageNet 常数）
  context_proprio [4, 8]         （pusht state 2 维 → pad 到 8）
  future_video    [3, 3, H, W]
  future_proprio  [3, 8]
  executed_actions[3, 20]        （pusht action 2 维归一化到 [-1,1] → pad 到 20）
  future_events   [3, 4]         （0；P1 无事件）

用法：
  python scripts/lerobot_to_windows.py \
    --data-root /root/autodl-tmp/realdata/pusht \
    --out /root/autodl-tmp/pusht_windows \
    --max-episodes 30 --img-size 384 --stride 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# ImageNet 归一化常数（V-JEPA 骨干输入约定；与合成 torch.randn 量级一致）
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

CONTEXT_T = 4
FUTURE_T = 3
PROP_dim = 8
ACTION_dim = 20


def decode_video_frames(mp4_path: str, max_frames: int) -> np.ndarray:
    """用 PyAV 解码 mp4，返回 [N, H, W, 3] uint8（最多 max_frames 帧）。"""
    import av

    frames = []
    container = av.open(mp4_path)
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= max_frames:
            break
    container.close()
    return np.stack(frames)  # [N,H,W,3]


def upscale_and_normalize(frames: np.ndarray, img_size: int) -> torch.Tensor:
    """[N,H,W,3] uint8 → [N,3,img_size,img_size] 归一化 float32。"""
    from PIL import Image

    out = []
    for f in frames:
        img = Image.fromarray(f).resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0          # [H,W,3] in [0,1]
        arr = (arr - MEAN) / STD                                  # ImageNet 归一化
        out.append(arr.transpose(2, 0, 1))                        # [3,H,W]
    return torch.from_numpy(np.stack(out))                        # [N,3,H,W]


def normalize_action(action: np.ndarray) -> np.ndarray:
    """pusht action（绝对目标位置，约 [0,512]）→ [-1,1]。"""
    return np.clip(action / 512.0 * 2.0 - 1.0, -1.0, 1.0)


def pad_to(vec: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    out[: len(vec)] = vec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-episodes", type=int, default=30)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.data_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 读 parquet
    import pyarrow.parquet as pq

    table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    state = np.stack([np.asarray(s, dtype=np.float32) for s in table.column("observation.state")])
    action = np.stack([np.asarray(a, dtype=np.float32) for a in table.column("action")])
    ep_idx = np.asarray(table.column("episode_index").to_pylist())
    n_frames_total = len(state)

    # 需要多少帧：覆盖前 max_episodes 个 episode
    ep_ids = np.unique(ep_idx)[: args.max_episodes]
    last_ep = ep_ids[-1]
    frame_end = int(np.searchsorted(ep_idx, last_ep, side="right"))
    need_frames = min(frame_end, n_frames_total)
    print(f"episodes: {len(ep_ids)} | frames needed: {need_frames}")

    # 解码视频帧
    mp4 = root / "videos/observation.image/chunk-000/file-000.mp4"
    video = decode_video_frames(str(mp4), need_frames)  # [N,H,W,3]
    print(f"decoded frames: {len(video)}")
    n = min(len(video), need_frames)

    # 预归一化所有需要的帧（为省内存，按窗口惰性处理；这里全量上采样可能大，故按窗口）
    manifest_entries = []
    widx = 0
    window_len = CONTEXT_T + FUTURE_T
    for ep in ep_ids:
        mask = ep_idx[:n] == ep
        ep_frames = np.where(mask)[0]
        if len(ep_frames) < window_len:
            continue
        start0 = int(ep_frames[0])
        ep_len = len(ep_frames)
        for off in range(0, ep_len - window_len + 1, args.stride):
            g = ep_frames[off: off + window_len]  # 全局帧下标
            if g[-1] >= n:
                break
            ctx_frames = video[g[:CONTEXT_T]]
            fut_frames = video[g[CONTEXT_T:]]
            ctx_video = upscale_and_normalize(ctx_frames, args.img_size)  # [4,3,H,W]
            fut_video = upscale_and_normalize(fut_frames, args.img_size)  # [3,3,H,W]
            ctx_prop = np.stack([pad_to(state[i], PROP_dim) for i in g[:CONTEXT_T]])
            fut_prop = np.stack([pad_to(state[i], PROP_dim) for i in g[CONTEXT_T:]])
            acts = np.stack([pad_to(normalize_action(action[i]), ACTION_dim)
                             for i in g[CONTEXT_T:]])
            events = np.zeros((FUTURE_T, 4), dtype=np.float32)
            window = {
                "context_video": ctx_video,
                "context_proprio": torch.from_numpy(ctx_prop),
                "future_video": fut_video,
                "future_proprio": torch.from_numpy(fut_prop),
                "executed_actions": torch.from_numpy(acts),
                "future_events": torch.from_numpy(events),
            }
            path = out / f"window_{widx:05d}.pt"
            torch.save(window, path)
            manifest_entries.append({"path": str(path)})
            widx += 1

    manifest = out / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(e) for e in manifest_entries) + "\n",
                        encoding="utf-8")
    print(f"wrote {widx} windows -> {out} | manifest: {manifest}")


if __name__ == "__main__":
    main()