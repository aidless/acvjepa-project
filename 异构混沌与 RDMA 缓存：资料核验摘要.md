# 异构混沌与 RDMA 缓存：资料核验摘要

1. PyTorch Distributed Checkpoint 支持多 rank 并行保存/加载，并支持 load-time resharding，使得 checkpoint 可在一种集群拓扑保存、在另一种拓扑加载。各 rank 在加载时应只读满足本地 shard 所需的最少数据。来源：https://docs.pytorch.org/docs/2.13/distributed.checkpoint.html
2. DCP async checkpoint 先将状态安全 stage 到 CPU buffer；异步 checkpoint 会增加 CPU/pinned memory 压力，官方建议通常限制为一次并发异步 checkpoint。来源：https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html
3. NVIDIA GPUDirect RDMA 支持 GPU 与第三方 PCIe peer（例如 NIC、存储适配器）直接交换数据，但硬件拓扑、root complex、IOMMU、BAR、memory pin/unpin、driver callback 和资源耗尽均是实际约束。注册/针住缓存可减少 pin/unpin 开销，但需要严格容量、失效和回调处理。来源：https://docs.nvidia.com/cuda/gpudirect-rdma/
4. etcd KV API 提供 durability 与 strict serializability；线性化请求经 Raft 有性能成本。Watch 有序/唯一/可恢复，但不保证线性化，消费者必须检查 revision。来源：https://etcd.io/docs/v3.5/learning/api_guarantees/
5. 由此推导：durable checkpoint manifest/commit pointer 适合使用强一致 KV/账本做小元数据的 compare-and-swap，而 GB/TB 级 tensor shard 不应塞进一致性 KV；大状态数据应在对象存储/并行文件系统及只读 RDMA/节点本地缓存之间分层分发，缓存命中仍要按 manifest/hash 验证。
