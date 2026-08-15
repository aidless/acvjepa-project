# Docker Compose 本地演练：资料核验摘要

1. Docker Compose 服务可通过 `depends_on` 长语法的 `condition: service_healthy` 等待依赖健康；`healthcheck` 用于声明容器健康状态。来源：https://docs.docker.com/reference/compose-file/services/
2. Prometheus 配置文件定义 scrape job、targets 与 rule files；`scrape_config` 可用 static_configs，规则可通过配置文件加载。来源：https://prometheus.io/docs/prometheus/latest/configuration/configuration/
3. Grafana 支持用版本控制的 provisioning 文件定义 datasource 和 dashboard；dashboard provider 可从本地文件路径载入 JSON，配置可禁用 UI 更新。来源：https://grafana.com/docs/grafana/latest/administration/provisioning/
4. 工程选择：本地 Compose 可用单独的 demo metrics service 暴露离线演练生成的低基数 Prometheus 指标，Prometheus 仅 scrape 该服务，Grafana 通过只读 provisioned datasource/dashboard 展示；无真实集群、RDMA、KV、GPU 或故障网络控制。
