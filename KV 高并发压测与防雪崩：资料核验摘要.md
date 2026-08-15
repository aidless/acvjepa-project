# KV 高并发压测与防雪崩：资料核验摘要

1. etcd 性能受吞吐与延迟共同约束；线性化读要经 quorum，一般比可返回陈旧数据的 serializable read 更昂贵。官方建议在新环境先运行 benchmark，因为性能对环境差异敏感。来源：https://etcd.io/docs/v3.5/op-guide/performance/
2. etcd 需要管理 keyspace history、compaction、defragmentation 和 space quota；quota 达到会产生 cluster-wide NOSPACE alarm 并限制操作。故 checkpoint 系统只应保存小指针/manifest metadata，不应将大 shard 写入 etcd。来源：https://etcd.io/docs/v3.5/op-guide/maintenance/
3. Redis client-side cache 的 invalidation/TTL 有陈旧和带宽/内存权衡；失效通知连接丢失时客户端应 flush local cache；缓存条目应有 max TTL 和内存上限。来源：https://redis.io/docs/latest/develop/reference/client-side-caching/
4. Envoy adaptive concurrency 基于 minRTT 与采样延迟动态调整允许 outstanding request 数；jitter 可防止所有 host 同时进入低并发测量窗口。这为每 tier/rail/worker 的恢复 load token 自适应控制提供可借鉴模式。来源：https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/adaptive_concurrency_filter
5. 工程推断：恢复路径的 COMMITTED pointer 读取必须线性化；cache warming/availability hint 可接受低成本的异步/陈旧通知，但绝不可用于决定恢复事实。采用 single-flight、分层 token bucket、TTL jitter、negative cache、熔断、指数退避和 admission wave，以保护 KV、对象存储和 RDMA cache 不被 cold-start/mass-preemption 雪崩击穿。
