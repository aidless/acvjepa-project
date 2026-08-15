# 异构微批混沌注入与 RDMA 缓存：安全边界

1. 自动化混沌框架默认使用进程内 failpoint、逻辑网络分区和 mock 缓存；它不包含 kill、iptables、tc、RDMA、交换机、云调度或 SSH 命令。真实节点抢占/rail 分区仅允许由基础设施团队在批准的隔离资源池执行。
2. 注入器必须一次只启用一个故障，具有确定 experiment seed、TTL、停止条件、双人批准引用和 finally rollback。任何失败测试不得使 cursor 前移、optimizer step 提交、EMA 更新、模型发布或机器人控制发生。
3. RDMA 外置缓存、远程内存池或分布式 KV 仅是只读的 performance cache 和并发加载协调层；它不能成为数据游标、checkpoint 提交或模型状态的唯一事实来源。唯一恢复锚仍是 durable ledger 中的 COMMITTED checkpoint manifest/hash。
4. 缓存条目必须绑定 checkpoint content hash、precision contract、optimizer/FP8 backend version、state shard ID、加密/访问策略和 TTL。缓存命中后仍须验证分片 hash；miss、过期、版本不匹配、KV quorum 不足或 RDMA 健康异常必须回退到对象存储/持久 checkpoint，而非使用旧版本。
5. 大规模恢复中采用分波 admission、缓存读并发配额、每机/每 rail 限速与熔断；不得为了缩短 RTO 而使所有节点并发拉取全量 checkpoint 或允许跨租户/跨作业读取缓存。
