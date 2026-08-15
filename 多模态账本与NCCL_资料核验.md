# 多模态账本与 NCCL 动态同步：资料核验摘要

- NIST SP 800-12 第 18 章将审计轨迹定位为支持个人责任、事件重建、入侵检测和问题分析的技术控制，并指出数字签名可保护审计轨迹免遭未被察觉的修改；同时明确，签名本身不阻止日志删除或修改。因此本方案把哈希链、Merkle 证据根和数字签名用于篡改可见性，并以独立对象存储版本保留、访问控制与外部锚定处理删除/重放威胁，而不声称“单一数据库不可篡改”。
- PyTorch 2.13 的 Distributed 文档提供 `broadcast_object_list`、`broadcast`、`all_reduce` 和 `barrier` 等 collective。DDP 在过程组内执行同步梯度归约；所有 rank 必须以相同的 collective 次序参与。方案采用：rank 0 在每个更新边界广播结构化 UpdatePlan；各 rank 再广播对计划的确定性摘要并执行 MIN/MAX 校验；仅在最后一个本地微批启用 DDP 同步 backward，使所有 rank 共同触发一次梯度 AllReduce。
- NCCL collective 文档定义 Broadcast 为由 root rank 向所有 rank 复制缓冲区、AllReduce 为跨所有 rank 归约结果。因此 `UpdatePlan` 控制面先通过 `broadcast_object_list` 传递，而 DDP 的梯度数据面保持 NCCL AllReduce；该代码仅支持稳定成员集合内的异构工作量，成员弹性必须在 checkpoint 边界重启进程组。

## 参考链接

1. NIST, *SP 800-12, Chapter 18: Audit Trails*：https://csrc.nist.rip/publications/nistpubs/800-12/800-12-html/chapter18.html
2. PyTorch, *torch.distributed 2.13 documentation*：https://docs.pytorch.org/docs/2.13/distributed.html
3. NVIDIA, *NCCL Collective Operations*：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
