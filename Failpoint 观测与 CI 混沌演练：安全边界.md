# Failpoint 观测与 CI 混沌演练：安全边界

1. CI 仅执行 offline logical failpoint、缓存调度模拟、配置/schema 校验和 CPU/Gloo 语义测试；不得调用真实 NCCL 多机、RDMA、网络控制、节点调度、对象存储、etcd/Redis 或机器人接口。
2. 观测指标不得携带 work ID、checkpoint hash、客户数据、错误正文、审批人、GPU UUID 或其他高基数/敏感标签。允许标签只限于稳定的 `fault_class`、`phase`、`cache_tier`、`component_class`、`outcome`、`cluster/job/environment`。
3. failpoint 时延以单调时钟记录；跨节点真实时钟不用于端到端相减。生产系统应从协调器签发的 fault/recovery trace ID 对齐事件，而不假定节点 wall clock 完全同步。
4. CI 告警联动仅可以生成 freeze/evidence/notify 意图和测试报告；没有生产凭据、部署 token、基础设施 mutation token 或模型发布权限。
5. 任何断言失败、指标 schema 漂移、cache integrity failure、cursor/plan 状态异常都应使流水线失败并归档诊断工件；不得自动重试以掩盖确定性恢复错误。
