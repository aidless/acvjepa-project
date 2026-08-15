# 线程安全缓存与 Kubernetes CI：资料核验摘要

1. Python `threading` 在线程间共享内存；Lock/Condition 等同步原语用于协调共享状态。线程适合 I/O-bound 并发，不应把 GIL 视为数据结构并发安全保证。来源：https://docs.python.org/3/library/threading.html
2. Kubernetes SecurityContext 可约束 UID/GID、privileged、Linux capabilities、`allowPrivilegeEscalation` 与 `readOnlyRootFilesystem`；推荐演练 Job 使用非 root、只读根文件系统、drop ALL capabilities、RuntimeDefault seccomp。来源：https://kubernetes.io/docs/tasks/configure-pod-container/security-context/
3. Kubernetes RBAC 的 Role 是 namespace 范围，RoleBinding 只在绑定 namespace 生效；权限是纯加法，使用 resource/verb wildcard 会造成过度授权风险，应按资源和 verb 最小授权。来源：https://kubernetes.io/docs/reference/access-authn-authz/rbac/
4. GKE CI/CD 最佳实践建议区分快速/完整 CI，测试容器与镜像结构，早期引入安全检查，使用 GitOps，构建一次并按环境 promote 而非重建，区分开发/预生产/生产集群，并准备监控与回滚。来源：https://docs.cloud.google.com/kubernetes-engine/docs/concepts/best-practices-continuous-integration-delivery-kubernetes
5. 工程推断：在内存单进程中用 `Condition` 将 immutable shard 的 leader/result/error/expiry 原子绑定；跨 pod/节点时必须用带租约和 fencing token 的一致性协调面取代 Python 内存锁，且仍需要 pointer/hash/precision/cursor gate。
