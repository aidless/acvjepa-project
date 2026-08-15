# 生产阈值与 GitOps 仲裁：资料核验摘要

1. Prometheus alerting rule 支持 `for` 使条件持续一段时间后才 firing，支持 `keep_firing_for` 降低短暂恢复/数据缺失导致的告警抖动；Prometheus 把 pending/firing alert 以 `ALERTS` time series 暴露，Alertmanager 负责通知聚合、限频、静默和依赖关系。来源：https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/
2. Kubernetes Lease（`coordination.k8s.io`）用于锁定共享资源和协调活动，官方控制面用它实现 leader election；自定义 controller 也可使用 Lease 来确保同一组件仅一个主动 leader。来源：https://kubernetes.io/docs/concepts/architecture/leases/
3. Argo CD auto-sync 会在 Git desired manifests 与 live state 有差异时同步，因此 CI 可仅提交 Git 而不持有 Argo CD API 权限。其自动同步按 unique commit SHA + application parameters 尝试一次；retry 可以配置有上限的指数退避。source：https://argo-cd.readthedocs.io/en/stable/user-guide/auto_sync/
4. 工程推断：Lease 只能选出 controller leader，不能单独保证训练 cursor/checkpoint/precision/plan 与 Git desired revision 的跨系统原子性。需要在强一致恢复记录上写入 epoch、fencing token 与全部输入 digest，并用 admission/GitOps pre-sync gate 拒绝不匹配状态。
5. 工程推断：具体告警数值应由成功恢复的分层基线、集群拓扑、checkpoint 体积与 error budget 推导；初始绝对阈值必须与相对基线/多窗口条件、`for`、`keep_firing_for` 联用，避免把偶发慢恢复或单个坏节点误当作全局事故。
