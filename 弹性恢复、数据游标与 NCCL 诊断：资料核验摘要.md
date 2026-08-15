# 弹性恢复、数据游标与 NCCL 诊断：资料核验摘要

1. PyTorch `torchrun` 2.13 文档指出：worker failure 会停止并重启所有 worker；节点离开/加入会停止现存 worker、形成新的 worker group，并以新的 `RANK` 和 `WORLD_SIZE` 启动。`RANK` 不稳定，elastic 场景不可硬编码 `WORLD_SIZE`，并要求作业有 checkpoint。来源：https://docs.pytorch.org/docs/2.13/elastic/run.html
2. NCCL communicator 文档指出：网络等异步通信错误通常不再 progress；应 abort/destroy communicator 后才可能重建。其 fault tolerance 部分也说明，在任意 rank 失败时健康 rank 需要 abort 自身 communicator；`ncclCommShrink`/grow 是原生 API 路线，使用前需重建更高层训练状态。来源：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html
3. PyTorch ProcessGroupNCCL 文档列出 `TORCH_NCCL_ASYNC_ERROR_HANDLING`、timeout dump、trace buffer、desync debug、per-collective timing 和 monitoring/heartbeat 等观测与错误处理变量。来源：https://docs.pytorch.org/docs/2.13/torch_nccl_environment_variables.html
4. NCCL RAS 文档说明 RAS 可报告 unresponsive/missing process、communicator collective-count mismatch 和异步错误；短暂 mismatch 只要 counters 继续进展不必判故障。来源：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html
5. 关于全局游标的 prepare/commit/abort 与 at-least-once 重读语义，是在上述 worker-group 失败/重启语义下做出的工程设计推断，不是 PyTorch 内建 exactly-once DataLoader 承诺。
