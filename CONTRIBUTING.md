# Contributing

感谢关注。本仓库是研究性质的可运行工程体系（AC-VJEPA / JEPA 世界模型路线验证），欢迎 issue、PR 与实验复现。

## 环境与验证

```sh
# Fresh-clone contract (no weights, no GPU, no Docker).
# Needs PyYAML and prometheus-client; see .github/workflows/failpoint-chaos-ci.yml for the pins.
bash scripts/verify_committed.sh
```

`verify_all.ps1` is the historical Windows host runner. Groups that need weights, CUDA, or Docker SKIP when those inputs are absent. A SKIP is not a pass.

注意事项：
- 本机验证基线使用 `python -m unittest`（全局 pytest 的 deepeval 插件在本机损坏，见 VERIFY_RESULTS.md 备注）；
- torch 2.5.x Windows wheel 无 libuv：torchrun/Gloo 需 `USE_LIBUV=0`；
- 官方 V-JEPA 2 权重不入库：按 `DATA_MANIFEST.md` 从 Hugging Face 下载并遵守 Meta 条款。

## 提交规范

- 每个 commit 只做一件事；PR 描述关联 PROJECT_PLAN 里程碑或 BACKLOG 条目；
- 改动代码必须通过 `bash scripts/verify_committed.sh`。若另外跑了 `verify_all.ps1`，写明哪些组是 SKIP，不要写成全绿。
- 涉及数据/训练语义的改动，同时在 `决策记录.md` 追加条目（决策/依据/证据/未决影响四列）；
- 涉及评测指标的改动，需先修订 `M3_MPC_EVALUATION_DESIGN.md`（预注册设计）再改代码。

## 安全边界

本项目包含分布式训练故障注入与机器人数据管线的**离线逻辑实现**。所有 PR 必须遵守：
- 不含任何真实网络扰动、机器人控制、集群/云端操作命令（守卫/校验脚本除外，且其执行器为受控基础设施职责）；
- 合成数据仅用于链路验证，禁止声称其为物理有效训练数据；
- 高风险数据补丁遵循 HITL 双重审核语义（见 `hitl_quarantine_review.py`）。

发现安全/正确性问题请通过 GitHub Security 通道或 issue 报告，勿公开未修复的利用细节。
