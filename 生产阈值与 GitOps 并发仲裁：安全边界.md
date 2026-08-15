# 生产阈值与 GitOps 并发仲裁：安全边界

1. Prometheus 阈值是容量保护和人工升级信号，而不是“准确性已证明”的绝对常数。任何具体 p95/p99、hit ratio、队列长度与 timeout 必须先按集群规格、checkpoint 体积、GPU/NIC 拓扑、正常恢复基线和 error budget 校准；本项目交付的数值只作为初始保护栏。
2. 告警自动动作仅允许：冻结新 UpdatePlan、标记恢复/部署为 suspect、保留 checkpoint/pointer/cursor/epoch/trace 证据、通知指定负责人。告警不得直接执行 Kubernetes apply/rollback、删 pod、修改网络策略、重启节点、切换 checkpoint pointer 或提升 canary。
3. Rendezvous 重建、训练恢复和 GitOps 变更必须共享一个线性一致的 `RecoveryDeploymentEpoch` 记录。每个写入都须携带其 generation/fencing token、checkpoint hash、cursor commit ID、precision contract、topology epoch、Git revision 与 payload digest；任何陈旧 generation 只能读取或失败，不能提交或发布。
4. 仅在一致记录显示 `RECOVERY_READY` 且全状态验证通过时，GitOps controller 才可创建/更新训练工作负载。出现分区、epoch ambiguity、pointer/cursor mismatch、precision mismatch、未提交 attempt 或 Git revision drift 时，状态应为 `FROZEN`，并等待人工仲裁。
5. 真正的生产部署和故障演练必须在受保护环境、批准变更和预先测试的 rollback path 下进行。本文方案不授予 CI、Alertmanager、dashboard 或训练 worker 直接高权限集群控制能力。
