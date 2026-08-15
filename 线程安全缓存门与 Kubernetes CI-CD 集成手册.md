# 线程安全缓存门与 Kubernetes CI/CD 集成手册

**作者：Manus AI**  
**日期：2026-08-15**  
**适用范围：** 2:1 异构微批训练的 checkpoint 恢复、Failpoint 5 完整性损坏回退、离线混沌合同，以及将本地 Compose 演练逐步纳入 Kubernetes 隔离演练与生产发布门控。

> **关键澄清：** 早期 `SingleFlight`、`TokenBudget` 与 `CacheAdmissionController` 是确定性逻辑模型，用于证明正确性契约；它们本身并不是多线程生产实现。新 `threadsafe_checkpoint_load_gate.py` 是**同一 Python 进程内**的线程安全参考实现。跨 pod、跨节点或跨进程时，内存锁必须替换为具备租约、fencing token 与一致读语义的外部协调面；但 pointer、manifest、hash、precision 和 cursor 的验证合同不得改变。

## 1. 当前线程安全实现的职责边界

`ThreadSafeVerifiedShardGate` 只解决“如何安全、节流地读取一个 immutable checkpoint shard”。它明确**不**拥有训练提交权：它不写 committed pointer、不执行 KV CAS、不推进 data cursor、不提交 optimizer step、不更改 `UpdatePlan`，也不发布模型。训练恢复仍必须在 checkpoint 加载、全状态 fingerprint、cursor/plan/epoch 验证完成后，经过既有 `TrainingReady` 门控。

| 状态 | 受哪把锁保护 | 可变动作 | 被禁止的动作 |
|---|---|---|---|
| verified cache map | `ThreadSafeVerifiedShardGate._condition` | 读取、完整性校验、驱逐、验证后回填 | 返回未验 bytes；修改 pointer/cursor |
| per-key flight map | 同一 `Condition` | leader 选举、follower 等待、错误/结果发布、generation fencing | 两个 leader 同时发布同一 key |
| durable admission | `ThreadSafeAdmission._condition` | token acquire/release、limit 变更、通知 waiter | 超过全局 durable 并发预算 |
| committed pointer/cursor | 不在此模块中 | 无 | 任意 cache gate 写入 |

Python 的线程共享同一进程内存；`Lock`/`Condition` 用于协调共享状态。对于 I/O-bound shard 加载，线程仍可有效重叠等待；但不能把 GIL 当作 map、计数器或多步检查—修改操作自动安全的理由。[1]

## 2. 单点并发控制：逐步解读

### 2.1 第一把锁：按 key 选出唯一 leader

每个 shard 用不可变 key 标识，至少应绑定 namespace、checkpoint hash、precision contract、component、shard id 与 shard SHA-256。进入 `load()` 后，线程先持有 `_condition`：若 cache bytes 存在，先校验 SHA-256；通过则立即返回 cache hit，失败则仅在仍是同一 bytes 时删除该 entry，防止删除另一个线程刚刚验证回填的新版本。

若没有有效 cache，线程在 `_flights[key]` 下做 leader 选举。第一个线程写入 `_Flight(generation, deadline)` 并成为 leader；同 key 后续线程看到同一 flight 后，只增加 follower 计数并在 `Condition.wait(remaining)` 中等待。这里必须使用 `while not flight.done`，而不是 `if`，因为等待允许被无关通知或虚假唤醒打断。

```python
with self._condition:
    flight = self._flights.get(key)
    if flight is None:
        self._next_generation += 1
        flight = _Flight(generation=self._next_generation, deadline=deadline)
        self._flights[key] = flight
        leader = True
    else:
        leader = False

    if not leader:
        while not flight.done:
            self._condition.wait(remaining)
        if flight.error is not None:
            raise LoadGateError(...) from flight.error
        return LoadResult(flight.payload, "singleflight_follower")
```

因此，**single-flight 的粒度是 immutable shard key，而不是整个 checkpoint**。32 个 reader 请求同一个 optimizer shard 时只合并为一个 I/O；model、EMA 与不同 optimizer shard 仍可各自有 leader，从而在有足够 durable budget 时并行加载。

### 2.2 第二把锁：跨 key 的 durable admission token

single-flight 不能防止多个**不同** shard 同时 fallback。`ThreadSafeAdmission` 使用独立 `Condition` 维护 `_inflight < _limit`：检查可用 token、递增计数和 waiter 唤醒都在同一临界区，因此两个 leader 不会同时观察到“还有一个 token”并双重占用。

```python
with self._condition:
    while self._inflight >= self._limit:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LoadTimeout(...)
        self._condition.wait(remaining)
    self._inflight += 1
```

leader 在临界区外执行慢 I/O 与 SHA-256 校验，避免一个大 shard 读取阻塞其他 key 的 cache hit、leader 选举或 follower 取消。无论 loader 成功、完整性失败还是抛异常，`finally` 都归还 token；否则异常路径会永久耗尽预算并形成自我放大的恢复死锁。

| 情形 | per-key single-flight | global durable token | 预期 |
|---|---|---|---|
| 32 个线程请求同一坏 shard | 1 leader、31 follower | leader 占 1 token | 1 次 durable read，31 个共享结果 |
| 两个不同坏 shard，limit=1 | 2 个 key leader | 第 2 个 leader 等待 | 最大 durable inflight=1 |
| leader hash 失败或 loader 抛错 | waiter 获得相同终态 error | `finally` 归还 token | 不返回不可信 bytes，不泄漏容量 |
| follower 超时 | follower 退出等待 | 不影响 leader token | 返回超时，调用方走恢复/冻结策略 |

### 2.3 第三道防线：lease deadline 与 generation fencing

没有 deadline 的 single-flight 会因 leader 卡死而永远锁住一个 key。当前实现将 `deadline` 与单调递增 `generation` 放进 `_Flight`。新请求发现当前 leader lease 过期时，会在持锁状态下把旧 flight 标记失败、从 map 移除并创建新 generation。旧 leader 即使晚些时候完成 I/O，也会在发布前发现 `_flights[key] is not flight`，从而被 fence，不能覆盖新 leader 已回填的 cache。

> **正确的抢占条件不是“任何线程等得不耐烦”，而是已批准的 recovery deadline 已过。** 太短的 lease 会造成不必要双重 I/O；太长则拉高 `RTO_trusted`。阈值应根据 checkpoint shard p99、admission queue、Rendezvous window 与观测告警动态配置，而不是固定猜测。

跨进程/跨节点时，Python 的 flight map 不可见；实现应迁移为 `lease(key, generation, expiry, owner)` 的一致性记录。lease 取得/续约/失效必须由线性一致 CAS 或等价事务实现，并把 generation/fencing token 传递到缓存回填服务；存储服务只能接受最新 token 的写入。即便如此，cache 仍是加速层：提交仍由 durable checkpoint 与 cursor ledger 控制。

### 2.4 Failpoint 5 如何触发防雪崩

Failpoint 5 的 cache corruption 路径遵循下图：

```text
bad cache payload
   │ SHA/长度不匹配
   ▼
驱逐 entry + 记录 integrity_failed + suspect cache tier circuit
   │
   ├── 同 key follower ──► Condition 等待 leader（零重复 durable I/O）
   │
   └── leader ──► admission token ──► durable read ──► SHA verify
                                                │
                         failure ──► error/冻结/取证
                                                │
                              success ──► cache rewarm ──► notify_all
                                                           │
                                            31 follower verified cache hits
```

`checkpoint_integrity_corruption_demo.py` 与线程安全 gate 的联合回归给出两个层面的证明。逻辑 demo 中，损坏 cache shard 导致 `cache_corruption_events=1`、恢复期仅 `durable_fallback_reads=1`、`coalesced_waiters=31`、`follower_verified_cache_hits=31`，且 pointer revision 和 cursor 均未变化。真实线程 gate 使用 32 个线程与 start Event 同步抢占：断言 `durable_loader_calls=1`、leader=1、followers=31、same-key max durable inflight=1；再以两个不同 key 验证共享 token limit=1 时 `distinct_key_max_durable_inflight=1`。

## 3. 从本地 Compose 到 Kubernetes：同一合同，分层权限

为避免测试漂移，本项目将测试序列收敛到 **`scripts/run_offline_chaos_contract.sh`**。Docker Compose 的 `chaos-ci`、GitHub PR CI、Kubernetes 隔离 Job 都调用同一脚本；该脚本运行独立五类 failpoint、聚合 chaos、线程安全 gate、cache load-shedding、corruption demo、Prometheus/Grafana 配置和静态安全检查。三者差异应仅在运行隔离级别、镜像 digest 与工件采集方式，而不应改变断言逻辑。

| 路线 | 运行内容 | 权限/风险 | 适合阶段 |
|---|---|---|---|
| **A. 本地 Compose / PR CI** | 同一离线脚本与容器配置 | 无 kubeconfig、无集群操作、无真实故障 | 每个 PR；快速反馈；默认选择 |
| **B. 受保护的 GitOps 清单产物** | 手动审批后仅渲染 immutable-digest Kubernetes Job | CI 仍无集群凭据；部署由独立受控 GitOps 身份完成 | 合并后、隔离 namespace 前的审批门 |
| **C. 隔离 Kubernetes contract Job** | 使用相同镜像/脚本的非特权 Job | namespace-only deployment identity；无真实网络/节点/RDMA fault | 预生产回归与运行时政策验证 |
| **D. 平台团队真实故障演练** | 仅预注册 profile 的真实 data-plane/node 变更 | 最高风险；独立资源池、双人批准、变更窗口 | 不作为常规 CI；人工发起 |

路线 A 和 B 是轻量且可审计的默认路径；路线 C 只验证“在 Kubernetes 受限策略下，合约镜像仍能运行”；路线 D 不能由 CI 自动升级触发。

### 3.1 PR 阶段：无需任何集群访问

现有 `.github/workflows/failpoint-chaos-ci.yml` 维持 `contents: read`，固定 action SHA，运行共享脚本并上传脱敏工件。新增 `.github/workflows/kubernetes-chaos-contract.yml` 的 PR job 同样只运行共享脚本；它不含 kubeconfig、`kubectl`、Helm、secret、OIDC cluster identity 或部署步骤。

这一分层符合“快速 CI 与完整检查分离、尽早进行镜像/安全检查”的原则。[4] 在此阶段应额外接入组织自己的镜像签名、SBOM、漏洞扫描和 dependency policy，但不要将任何集群 credential 放入 PR runner。

### 3.2 受保护渲染：构建一次，提升而非重建

手动 `workflow_dispatch` 阶段要求一个已批准的专用 `acvjepa-chaos-contract@sha256:<64 hex>` 镜像。`render_kubernetes_chaos_contract.py` 只接受该路径模式和 immutable digest，替换模板 sentinel 后输出 Job manifest artifact；它不加载 kubeconfig、不调用 registry 或 Kubernetes API。

这样，CI 不会在不同环境重新 build 混沌镜像。测试通过的同一 digest 被 promote 到隔离 namespace，符合“build once, promote rather than rebuild”的交付原则。[4]

```bash
python render_kubernetes_chaos_contract.py \
  --image registry.example.com/team/acvjepa-chaos-contract@sha256:<approved-digest> \
  --output rendered/offline-chaos-contract.yaml
```

### 3.3 隔离 namespace contract Job

`k8s/chaos-lab/offline-chaos-contract.yaml` 包含四类资源：restricted `Namespace`、`automountServiceAccountToken: false` 的 ServiceAccount、默认拒绝 ingress/egress 的 NetworkPolicy，以及生成唯一名的 `batch/v1 Job`。

Job 使用 non-root UID/GID、`allowPrivilegeEscalation: false`、`readOnlyRootFilesystem: true`、`capabilities.drop: [ALL]` 与 `seccompProfile: RuntimeDefault`。这些设置对应 Kubernetes SecurityContext 的权限控制点。[2] NetworkPolicy 的意义是阻止 pod 运行时与任何外部端点通信；镜像拉取由节点运行时完成，生产集群还应以 registry policy、镜像缓存和 admission policy 管理其供应链。

RBAC 层面，Job 运行所用 ServiceAccount 不挂载 token，因此运行时代码没有 Kubernetes API 权限。部署身份应与 Job 身份分离：由受控 GitOps controller 或已审批的 deployment runner 使用 namespace-scoped Role 创建/读取该 namespace 的 Job 与日志；不要使用 ClusterRole/ClusterRoleBinding、通配符 verb/resource 或 node 权限。Kubernetes Role 的权限仅限一个 namespace，而 wildcard 会造成过度授权风险。[3]

还需注意：纯 RBAC 不能用 `resourceNames` 限制顶层 `create` 请求，因为创建时对象名可能尚未已知。[3] 因此真实集群必须另外使用 admission policy（例如组织认可的策略控制器）约束：仅允许 `acvjepa-chaos-contract` 已签名 digest、指定 labels、restricted securityContext、目标 namespace 和允许的 Job 参数。RBAC 负责“谁可申请创建”，admission policy 负责“可创建什么”。

### 3.4 GitOps 部署与回滚策略

建议将 render artifact 提交到一个受保护的 GitOps input 路径，或由批准系统把已审查 artifact 提供给 GitOps controller；controller 仅同步 `acvjepa-chaos-lab`。Git 评审、immutable manifest artifact、controller reconcile event、Job UID、run id、Prometheus report 与 cursor/pointer state 应一起写入审计记录。

| 门控 | 必须通过的证据 | 失败动作 |
|---|---|---|
| 源码 PR | shared offline script、6 个独立 failpoint、thread gate、配置 validators | 拒绝合并 |
| 镜像 | 专用 image path、批准 digest、扫描/签名（组织策略） | 不渲染/不 promote |
| 清单 | restricted namespace、NetworkPolicy、token disabled、non-root/read-only/cap drop | admission 拒绝 |
| 隔离 Job | zero real-fault capability、工件完整、指标/日志正常、cursor/pointer 不变 | 删除/TTL 回收 Job，标记演练失败 |
| 真正故障演练 | 平台双人批准、资源池隔离、rollback plan、observer 值班 | 不启动或立即回滚，不自动升级到生产 |

## 4. 已交付工件与验证范围

| 工件 | 用途 | 本地验证状态 |
|---|---|---|
| `threadsafe_checkpoint_load_gate.py` | 同进程多线程 single-flight、admission、lease/fencing | 32 same-key + 2 distinct-key smoke test 通过 |
| `scripts/run_offline_chaos_contract.sh` | Compose/CI/Kubernetes 共用测试序列 | 生成 13 类脱敏工件，联合回归通过 |
| `k8s/chaos-lab/offline-chaos-contract.yaml` | restricted namespace 与 offline Job 模板 | 静态安全验证通过 |
| `render_kubernetes_chaos_contract.py` | immutable digest Job manifest artifact | 合成 digest 渲染验证通过；cluster operation=0 |
| `.github/workflows/kubernetes-chaos-contract.yml` | PR offline + 手动受保护 render | 最小权限静态验证通过；cluster operation=0 |
| `validate_kubernetes_chaos_lab.py` / `validate_kubernetes_chaos_ci.py` | 清单与工作流防漂移 | 均通过 |

当前环境未连接 Kubernetes 集群，也未运行 Docker build/run、真实 storage/KV、RDMA、NCCL、节点抢占或网络分区。因而这些结果证明的是**线程安全局部并发、脚本一致性、清单/工作流安全合同和离线恢复不变量**，不是生产集群的性能、可用性或真实故障恢复验收。

## 参考资料

[1] [Python `threading` Documentation](https://docs.python.org/3/library/threading.html)。

[2] [Kubernetes: Configure a Security Context for a Pod or Container](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/)。

[3] [Kubernetes: Using RBAC Authorization](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)。

[4] [GKE: Best Practices for Continuous Integration and Delivery to Kubernetes](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/best-practices-continuous-integration-delivery-kubernetes)。
