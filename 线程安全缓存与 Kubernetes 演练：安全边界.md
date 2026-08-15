# 线程安全缓存与 Kubernetes 演练：安全边界

1. 现有 `SingleFlight`、`TokenBudget` 与 `CacheAdmissionController` 是确定性离线合同模型；它们不应被误认为已具备多线程或多进程线程安全性。生产实现必须以共享锁/condition、进程间协调或外部一致性租约替换其内存集合和计数器。
2. 每个 cache load 的 leader 只负责读取并验证 immutable shard；它不得更新 committed pointer、推进 cursor、提交 optimizer step、改变 UpdatePlan 或发布模型。任何 verification/CAS 失败都必须唤醒 waiter 为失败/回退状态而非返回未验证 bytes。
3. Kubernetes CI 的 PR 阶段仅运行本地 Compose/容器离线合同、镜像扫描、清单静态校验和 unit/integration tests，绝不获取 kubeconfig、集群凭据、production namespace 权限、`pods/exec`、`pods/delete`、NetworkPolicy 写入或 CRD 控制权限。
4. 真实演练只可在独立、受保护的演练 namespace 中由人工批准的工作流启动。实验身份的 RBAC 必须仅覆盖该 namespace，且默认不含 Secrets 读取、节点/命名空间操作、cluster-scoped RBAC、hostNetwork、privileged、hostPath 或设备权限。
5. 故障演练自动化只允许选择预注册且风险等级已批准的逻辑/应用级 profile；网络、RDMA/rail、节点抢占及任何 Kubernetes disruption 注入均由平台团队通过单独变更流程实施，完成后必须收集 cursor/pointer/state/metrics 证据并显式回滚。
