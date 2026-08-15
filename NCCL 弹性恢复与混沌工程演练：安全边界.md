# NCCL 弹性恢复与混沌工程演练：安全边界

1. 所有网络分区、节点失联、进程终止、带宽/延迟扰动和 rendezvous 故障演练仅能在专用测试节点池、独立测试 VLAN/命名空间、可丢弃数据副本和无机器人控制负载的变更窗口中执行。
2. 本文设计验证目标、故障模型、观察点、停止条件与恢复判定；实际网络策略、节点隔离、GPU reset、进程组杀伤或控制平面变更必须由拥有集群授权的基础设施人员按既定变更流程执行。不得在共享网络、共享 rendezvous 或其他团队作业上注入故障。
3. 每轮演练开始前必须确认一个已验证且可读取的原子 checkpoint、数据集提交哈希、模型/优化器/EMA 血缘、当前 topology/work manifest digest 和允许的最大重启次数。不得从部分 update 或未确认 checkpoint 恢复。
4. 出现 ECC/Xid、GPU 过热、RDMA/交换机告警、无法停止的 watchdog、重复重启、状态比较失败、错误的样本账本或证据缺失时，立即停止演练并保留日志、flight recorder、环境和 checkpoint 工件；不得自动重试掩盖根因。
5. 网络分区恢复后，旧 UpdatePlan、旧 rank 映射、旧 world size 和旧 topology epoch 一律失效。只能在 torchrun/rendezvous 形成新 worker group 后重新采集可信拓扑与 telemetry，广播新计划并从确认 checkpoint 开始。
6. 压测与故障恢复的结果仅用于离线训练正确性和基础设施可靠性评估，不得直接触发机器人动作、放松安全门控、跳过 HITL 审核或促进生产模型发布。
