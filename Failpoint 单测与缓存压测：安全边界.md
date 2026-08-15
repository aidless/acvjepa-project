# Failpoint 单测与缓存压测：安全边界

1. 五类 failpoint 的测试仅使用临时目录、SQLite、内存 cache/KV、逻辑 identity 和 mock 告警控制面。禁止在单测或 CI 中调用真实 torchrun、NCCL、网络、RDMA、节点、交换机、对象存储、Kubernetes 或云 API。
2. 高并发压测的默认实现是离散事件/令牌桶模拟器，而非向真实 KV、对象存储或 RDMA 服务施加不受限流量。任何真实压测都要在独立容量窗口、专用租户/namespace、预算上限和人工终止开关下运行。
3. 防雪崩措施（single-flight、negative cache、TTL jitter、分层令牌桶、并发配额、熔断、指数退避）只控制请求速率和回退路径；它们不能使缓存成为恢复真相，也不能将 KV watch 当作线性化 commit 读取。
4. 压测只能使用合成 checkpoint shard、虚拟 payload/时延和脱敏 metrics 标签。严禁使用生产数据、真实 checkpoint 内容、用户身份或高基数 work/hash label。
5. 任一 cache/KV/对象存储 overload、完整性异常、CAS 冲突或状态验证失败，都必须降低 admission、freeze new plan 或回退 durable read；不得放宽 hash/precision/cursor/plan 验证以追求吞吐。
