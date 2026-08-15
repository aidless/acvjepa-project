# 全状态一致性验证与跨节点拓扑感知 UpdatePlan：安全边界

1. 全状态对照仅用于离线训练正确性验证。它比较学生模型、EMA 目标、buffer 和 optimizer state，不生成机器人动作、不修改安全阈值，也不能作为生产发布的唯一依据。
2. 跨节点拓扑采集仅读取可信的节点清单、GPU/NIC/NUMA 映射和基础设施健康遥测。训练进程不得基于自报主机名或未经认证的拓扑文件直接切换网络接口、修改路由、设置 HCA 或执行节点操作。
3. `UpdatePlan` 必须在完整 optimizer/checkpoint 边界产生，并携带 world size、成员/拓扑纪元、数据集和 checkpoint 血缘。检测到 epoch、成员或 plan 摘要不一致时，当前 update 必须失败关闭；不得让任一幸存 rank 单独 step。
4. 拓扑感知仅影响未来 update 的数据成本分桶和本地微批预算，不改变 DDP 的同步 collective 顺序或世界成员。网络/节点故障恢复必须由 torchrun/rendezvous 与集群变更控制完成，再在新 process group 上重新采集遥测和计划。
5. 生产密钥、节点地址、NIC 名称、拓扑 XML、NCCL trace 与 checkpoint 可能属于敏感基础设施工件，必须存入受控工件库，不写入公开数据集或 HITL 审核内容。
