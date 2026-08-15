# 存储损坏安全回退与 Docker Compose 本地演练手册

**作者：Manus AI**  
**日期：2026-08-15**  
**目标：** 演示 Failpoint 5（`cache_corruption_during_restore`）在 2:1 异构微批训练恢复期间的可信回退路径，并给出一套本地、离线、无特权的一键 Docker Compose 演练环境。

> **边界声明：** 所有 demo 数据均为进程内合成 bytes 与临时文件。Docker Compose 不接触真实 checkpoint、对象存储、etcd/Redis、RDMA、NCCL、GPU、节点调度、网络设备或机器人；它也不具备任何此类凭据或控制能力。

## 1. 触发场景与不可妥协的不变量

Failpoint 5 的输入是：当前 committed checkpoint 的 optimizer shard 已被 verified cache 预热，但 cache copy 的 payload 在内存/NVMe/RDMA 缓存层被篡改、截断或损坏。它**不是** durable checkpoint 损坏；如果 durable source 同样无法通过 manifest hash/长度/precision contract 验证，恢复必须 block，而不是继续训练。

| 不变量 | 原因 | 演示中的可执行断言 |
|---|---|---|
| cache 永远不是事实来源 | acceleration copy 可丢失、过期、被污染 | committed pointer 的 revision 前后均为 `1` |
| 损坏 cache entry 必须隔离/删除 | 相同坏 shard 不能反复服务 | `cache_corruption_events == 1`，entry 被 `fetch_verified()` 驱逐 |
| fallback 先验证再回填 | durable bytes 也要匹配内容寻址 descriptor | leader 返回 `source="durable"` 且 payload hash/字节数匹配 |
| 未确认恢复不推进 data cursor | cache 修复不是新训练 update | `cursor_next_offset` 前后都为 `0` |
| 一个冷/坏 key 只允许一个 durable leader | 防止 32 节点把一个坏 cache 同时转化为 32 次 durable read | `coalesced_waiters == 31`，recovery durable fallback reads=`1` |
| pointer 不可借此“修复” | 不能以 cache 事件覆盖最后已确认 checkpoint | 不执行 KV CAS；pointer revision 恒定 |

这种模式符合“三层验证”原则：小而强一致的 KV 只保存 immutable manifest 的 committed pointer；durable store 保存内容寻址 shard；node-local/RDMA cache 只是 verified read-through acceleration。PyTorch 分布式 checkpoint 应同样保留原子 checkpoint/manifest 作为恢复真相，而不是以缓存状态替代它。[1]

## 2. 缓存损坏到可信恢复：逐步演示

`checkpoint_integrity_corruption_demo.py` 完整实现了下面的时序。

```text
T0  committed pointer = (revision=1, checkpoint_hash, manifest_digest)
    cursor.next_offset = 0
    cache contains optimizer/rank-shard-0 (previously verified)

T1  test-only corruption replaces cached payload with "TAMPERED:..."

T2  leader reads cache:
    verify(pointer ↔ manifest ↔ descriptor)
    verify(len(payload), SHA-256(payload)) → FAIL
    cache entry evicted; metrics outcome=integrity_failed
    cache tier is marked OPEN_SUSPECT_READS_BLOCKED

T3  leader acquires single-flight(shard-key) + one durable token
    31 followers join the same immutable shard key; they do not open durable reads

T4  leader reads durable shard → verifies bytes/hash/precision-bound cache key
    only after success: rewarm verified cache; metrics outcome=fallback

T5  followers wake → each repeats cache integrity verification → 31 cache hits

T6  verify: pointer revision unchanged; cursor unchanged; no new UpdatePlan/commit
    only then emit TrainingReady for the *recovery path*.
```

### 2.1 为什么这能防止缓存雪崩

当许多节点同时恢复时，错误做法是每个 reader 在 cache miss/损坏后无协调地直读 durable object store。此时一次 cache 故障会放大成 `N` 次热点 shard I/O，导致 durable p99、重建时延和 retry 进一步恶化。

本演示组合五道控制：single-flight 对同一 immutable shard 合并请求；global/per-node admission token 限制 durable 并发；TTL jitter 分散后续无关 shard 的重热；negative cache 抑制确认缺失的 metadata 反复探测；integrity failure 打开 suspect cache tier circuit，并维持 durable manifest/pointer 为唯一真相。这些措施降低读放大，但**不**放松每次缓存/durable read 的 hash、长度、namespace、precision contract 或 manifest 绑定校验。

| 控制 | 防护对象 | 成功条件 | 失败时动作 |
|---|---|---|---|
| content-addressed verification | 被篡改、截断、错版本 shard | hash 与 byte_count 均匹配 | evict/隔离 cache entry |
| single-flight | 同一热点 shard 的并行恢复 | 单 leader 取得 shard key | followers 等待，不重复 I/O |
| durable token budget | 多 key 同时 fallback | inflight 不超过 global/per-node budget | admission 延迟或分波，而非扩容重试 |
| TTL jitter | 同步过期导致的波峰 | key expiry 分布展开 | 将即时 retry 错开 |
| integrity circuit | 不可信 cache tier 反复返回坏数据 | tier 健康检查通过后才关闭 | freeze 新计划、durable verified fallback、取证/升级 |
| pointer/CAS fence | stale/错误恢复覆盖已提交版本 | revision 与 manifest 对齐 | 拒绝写入，重新从 committed pointer 读取 |

本地结果为：`cache_corruption_events=1`、`durable_fallback_reads=1`、`coalesced_waiters=31`、`follower_verified_cache_hits=31`、`ttl_buckets=33`、pointer revision 保持 `1`、cursor 保持 `0`。这些是确定性合成实验结果，不能解释为真实对象存储、RDMA 或 GPU 集群性能数据。

## 3. 观测与告警链路

demo 通过既有 Prometheus 指标合同导出下列低基数事件：

```text
acvjepa_training_failpoint_stage_duration_seconds{
  fault_class="cache_corruption_during_restore",
  phase="trigger_to_training_ready"
}

acvjepa_training_checkpoint_cache_fetches_total{
  cache_tier="node_local", component_class="optimizer", outcome="integrity_failed"
}

acvjepa_training_checkpoint_cache_fetches_total{
  cache_tier="durable", component_class="optimizer", outcome="fallback"
}

acvjepa_training_checkpoint_cache_fetches_total{
  cache_tier="node_local", component_class="optimizer", outcome="hit"
}
```

Prometheus 使用 `scrape_config` 抓取内网 demo endpoint 并加载 recording/alert rules；其配置文件负责定义 scrape targets 与 rule files。[2] Grafana 使用 file provisioning 加载 datasource 和版本化 dashboard JSON，且 `allowUiUpdates: false`，避免本地 UI 修改掩盖源码定义。[3]

有意义的 PromQL 包括：

```promql
# failpoint-5 端到端 p95
histogram_quantile(
  0.95,
  sum by (le, cluster, job, environment) (
    rate(acvjepa_training_failpoint_stage_duration_seconds_bucket{
      fault_class="cache_corruption_during_restore",
      phase="trigger_to_training_ready"
    }[30m])
  )
)

# optimizer request hit ratio（只算正常 hit/miss/fallback）
sum(rate(acvjepa_training_checkpoint_cache_fetches_total{
  component_class="optimizer", outcome="hit"
}[5m]))
/
clamp_min(sum(rate(acvjepa_training_checkpoint_cache_fetches_total{
  component_class="optimizer", outcome=~"hit|miss|fallback"
}[5m])), 0.001)

# integrity failure 必须是非零即处置，而非等待大比例失败
sum(rate(acvjepa_training_checkpoint_cache_fetches_total{
  outcome="integrity_failed"
}[5m]))
```

## 4. Docker Compose 本地一键演练

### 4.1 服务构成

| 服务 | 本地端口 | 角色 | 安全设计 |
|---|---:|---|---|
| `metrics-demo` | `127.0.0.1:18000` | 启动一次合成 corruption/fallback demo，暴露 `/metrics`、`/healthz`、`/report` | 非 root、read-only、`cap_drop: ALL`、无外部 client |
| `prometheus` | `127.0.0.1:19090` | 仅 scrape `metrics-demo:8000`，加载本项目 rules | internal network、read-only、临时 TSDB |
| `grafana` | `127.0.0.1:13000` | 本地预置 datasource/dashboard | internal datasource、read-only provisioning、禁用注册 |
| `chaos-ci` | 不发布端口 | 显式运行离线 unittest/cache/chaos/metric/config contract | profile=`ci`，不接入任何外部系统 |

Compose 的 long-form `depends_on` 可等待依赖的 healthcheck 成功后再启动 Prometheus；本配置据此等待 `metrics-demo` 的 `/healthz`。[4]

### 4.2 前置条件

需要已安装 Docker Engine 与 Docker Compose v2。当前开发沙箱未安装 Docker CLI，因此本交付已完成 Python endpoint、YAML、JSON、静态安全和配置回归验证，但**未在该沙箱实际 build/pull/run 容器**。首次在本机运行前，建议先执行 `docker compose -f docker-compose.local-chaos.yml config`，并根据组织镜像供应链策略将 `python`、Prometheus 与 Grafana tag 固定为已批准的 digest。

### 4.3 一键运行

```bash
cd /home/ubuntu/lecun_analysis
./docker/run_local_chaos_demo.sh
```

脚本依次 build `metrics-demo` 与 `chaos-ci` 镜像，启动 metrics/Prometheus/Grafana，等待 metrics healthcheck，然后显式运行 `chaos-ci`。成功后可访问：

```text
http://127.0.0.1:18000/report
http://127.0.0.1:19090
http://127.0.0.1:13000
```

Grafana 的本地演示账号为 `local-demo / local-demo-not-for-production`；它只适用于 loopback demo，绝不能迁移到生产。离线 CI 工件会保存在 `artifacts-compose/`。

停止环境：

```bash
docker compose -f docker-compose.local-chaos.yml down --volumes --remove-orphans
```

### 4.4 验收结果

| 检查 | 预期 |
|---|---|
| `GET /healthz` | `ok` |
| `GET /report` | `cache_corruption_events: 1`、`durable_fallback_reads: 1`、pointer revision 不变、cursor 不变 |
| Prometheus `/targets` | `local-chaos-metrics-demo` 为 UP（等待 scrape interval 后） |
| Grafana | 自动加载 14-panel AC-VJEPA dashboard，并显示 failpoint/cache 查询 |
| `chaos-ci` | 五类独立 failpoint、cache load-shedding、corruption demo、观测演练及 config validators 全通过 |
| `artifacts-compose/` | 有脱敏 JSON、Prometheus exposition 和文本测试日志；不应包含 checkpoint/凭据/哈希明细 |

## 5. 已完成的验证与生产化限制

已在当前环境完成：Python `py_compile`、存储损坏演示、Prometheus/Grafana 配置验证、Compose 静态安全验证，以及短暂本地 HTTP endpoint smoke test。由于 Docker CLI 缺失，Compose build/run 必须由具备 Docker 的本地开发机或隔离 CI runner 完成。

生产化时，不能直接复用本地 Compose 的镜像 tag、匿名本地账号、临时 TSDB、plain HTTP 或单节点配置。需要替换为经审批准入的镜像 digest、mTLS/身份、受限 metrics ingress、持久化审计/告警、真实 KV quorum 监控、对象存储健康与同一组织隔离策略。任何 real RDMA/rail、NCCL、network partition 或 scheduler 故障演练仍必须在独立资源池、变更窗口和双人授权下实施。

## 参考资料

[1] [PyTorch Distributed Checkpoint Documentation](https://docs.pytorch.org/docs/2.13/distributed.checkpoint.html)。

[2] [Prometheus Configuration](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)。

[3] [Grafana Provisioning Documentation](https://grafana.com/docs/grafana/latest/administration/provisioning/)。

[4] [Docker Compose Services Reference](https://docs.docker.com/reference/compose-file/services/)。
