# 存储损坏演示与 Docker Compose：安全边界

1. 演示只使用合成 checkpoint shard、临时目录与进程内模拟的 durable store/cache/KV；不会读取、写入、上传、下载或变更任何生产 checkpoint、对象存储、etcd/Redis、RDMA 内存、GPU、节点或网络。
2. Docker Compose 只启动本地离线演练、Prometheus 和 Grafana。它没有宿主网络特权、host PID、privileged、Docker socket、设备映射、CAP_NET_ADMIN、真实凭据或外部服务 endpoint。
3. “缓存雪崩防护”指 single-flight、并发令牌、TTL jitter、negative cache、熔断和分波 admission 的逻辑验证；不包括、也不授权任何真实 traffic flood、网络限速或交换机/rail 操作。
4. 存储层损坏时，唯一允许的自动动作是隔离 cache entry、限制/熔断 suspect cache tier、从已确认 durable manifest 读取并验证、冻结新 UpdatePlan 的意图和生成审计报告；不得重写 committed pointer、推进 cursor、放宽 hash/precision 验证或发布模型。
5. Compose 的 Grafana/Prometheus 默认仅用于本机演示。若要长期运行监控，应依据组织的身份、数据保留、网络、镜像供应链和变更控制单独部署；不得把本地演示配置直接作为生产配置。
