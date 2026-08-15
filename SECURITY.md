# Security Policy

## 范围

本仓库是**离线研究与参考实现**：分布式训练故障注入、数据游标、混沌守卫均为逻辑/仿真级代码，不含真实网络扰动、机器人控制或云端操作命令（执行器属受控基础设施职责）。以下边界内的问题欢迎报告：

- 训练/恢复逻辑的正确性缺陷（游标、checkpoint、EMA、UpdatePlan、混合精度）；
- 数据管线（切窗、装配、隔离）的泄漏或不一致；
- 缓存/账本/守卫实现的完整性破坏；
- 依赖与构建问题。

## 上报

发现安全或正确性问题，请通过 GitHub 的 Security Advisory（Private vulnerability reporting）或邮件联系维护者；**不要**在公开 issue 中披露未修复的利用细节。

## 免责

参考实现不构成生产系统：部署前必须完成 `CLUSTER_VALIDATION_RUNBOOK.md` 与 `AC-V-JEPA 双臂部署：实机时延与控制频率验收协议.md` 所述的隔离验证与审批。
