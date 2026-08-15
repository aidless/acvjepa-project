# Failpoint 观测与 CI：资料核验摘要

1. Prometheus histogram 支持按时间窗口聚合并计算分位数；histogram 可在 Prometheus 侧聚合，而 summary 的预计算分位数不可聚合。来源：https://prometheus.io/docs/practices/histograms/
2. Prometheus 对 offline/batch 作业建议记录 major stage 时长、成功时间和运行时长；cache 关键指标包括总查询、hit、cache 前端服务的查询/错误/时延。标签集会增加时序成本，应避免高基数。来源：https://prometheus.io/docs/practices/instrumentation/
3. GitHub Actions 安全准则要求最小化 `GITHUB_TOKEN` 权限、避免不可信 PR 在 privileged trigger 下运行、审计第三方 action、优先固定 action 到完整 commit SHA；artifact 也应视为不可信输入。来源：https://docs.github.com/en/actions/reference/security/secure-use
4. upload-artifact 支持唯一工件名称、失败时 `if-no-files-found: error`、保留期设置和 SHA-256 digest 输出；默认隐藏文件不上传。来源：https://github.com/actions/upload-artifact
5. 工程推断：failpoint 恢复延迟以协调器/同一进程的单调时钟记录分段 span，生产跨节点关联以 trace ID 而非 wall-clock 差值；Prometheus 只承载低基数 phase/fault/tier/outcome 聚合，具体 experiment/attempt/hash 进入 audit artifact。
