# DATA_MANIFEST — 四层数据金字塔清单（M1，PROJECT_PLAN L4）

> 对应 `轻量级 V-JEPA：突破数据与算力限制的技术路线与实验室实施方案.md` §4「四层数据金字塔」。
> 本文件是数据资产的**登记清单**：声明每层的目的、规模起点、采集/许可约束与就绪状态；实际文件不入库（体积与许可）。
> 数据纪律（所有层）：按**布局/对象实例/任务组合**隔离训练/验证/测试集；时间戳/相机标定/动作坐标系全链路一致。

## 四层总览

| 层 | 目的 | 起始规模 | 动作标签 | 状态 | 登记路径/来源 |
|---|---|---|---|---|---|
| **A 通用预训练层** | 获取通用视觉时空先验 | 官方 V-JEPA 2.1 ViT-B/16（80M）权重 | 不需要 | 待获取（网络/许可前置） | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)；HF `transformers.models.vjepa2` |
| **B 本地无标签视频层** | 域适配（厨房/台面/双臂遮挡等） | 20–50 h | 不需要 | 待采集/许可确认 | 本地固定+手持相机；公开视频需许可 |
| **C 仿真交互层** | 受控反事实、稀有布局、长程组合 | 10–30 任务模板起步 | 仿真器自动生成 | 待装配 | [RoboCasa](https://robocasa.ai/)（RoboCasa365）子集 |
| **D 真实动作层** | 校准真实相机/夹爪/接触/延迟 | 5–10 h 受监督采集 | **需要严格时间同步** | 待实机（M5 前置） | 受控遥操作；需双臂硬件 |

## A 层：预训练权重（M1 已建适配接口）

- 适配器：`vjepa_backbone.py`（两条加载路径——原生格式键重映射 + **HF Transformers 真实权重路径** `HFVJEPA2Backbone`）。
- 加载入口：`train_ac_vjepa_ddp.py --init-from vjepa2:<path>[:frozen|last_k|lora|finetune]`（原生格式）；真实 HF 权重用 `install_hf_vjepa2_encoder`（frozen/finetune）。
- 冒烟：`python vjepa_backbone_smoke.py --safetensors <model.safetensors>`（真实权重，已 PASS）；`--checkpoint <official.pt>`（原生格式）；无权重走同构随机初始化路径。

### A 层登记：V-JEPA 2.1 ViT-B/16（80M）官方权重（已下载并验证）

| 字段 | 值 |
|---|---|
| 就绪日期 | 2026-08-15 |
| 来源 | HuggingFace `davevanveen/vjepa2.1-vitb-fpc64-384`（Meta V-JEPA 2.1 ViT-B 的 HF 转换版，distilled from ViT-G；模型卡见 [HF](https://huggingface.co/davevanveen/vjepa2.1-vitb-fpc64-384)） |
| 原仓库 | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)；论文 [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) |
| 本地路径 | `weights/vjepa2.1-vitb-fpc64-384/model.safetensors`（不入 git，见 .gitignore） |
| 大小 / SHA-256 | 438,855,416 bytes / `77D2D1166D26F1434A116E537E9B5E7B41AA72DC8212ECC3D7B9C46CC19D6035` |
| 结构 | ViT-B/16，384px crop，64 帧 clip，hidden 768，12 层 12 头；时空 patch embed + modality embeds + distillation norms |
| 加载验证 | `HFVJEPA2Backbone`：**391/395 键加载（encoder 全覆盖）**；12 个 skip = predictor 蒸馏头(1664)与辅助键（无需）；`[1,3,384,384]→[1,latent_dim]` forward 契约 PASS；安装进 AC-VJEPA + EMA 训练步 PASS |
| 许可 | 遵循 Meta V-JEPA 2 条款（HF 模型卡标注 `other`，原作者 Meta AI；商用需自行确认） |
| 用途 | M2 域适配的冻结骨干（frozen）或轻适配（finetune）；predictor 头不使用 |

## B 层：本地无标签视频（待采集）

- 内容：多厨房布局、不同光照、门开合、常见餐具、手/机械臂遮挡；去除隐私/不安全内容。
- 格式要求：`timestamp`、相机内外参、RGB/RGB-D；与训练窗口契约对齐（见 `make_demo_ddp_data.py` 的 tensor 布局）。
- 采集完成后的登记项：时长、相机型号与标定文件、场景清单、许可声明。

## C 层：RoboCasa 仿真子集（待装配）

- 任务模板选择：与「早餐台整理」相关的拾取放置、门/抽屉、桌面整理；跨布局/纹理/对象随机化。
- 每模板产出：RGB-D、位姿、动作、事件标签、失败/不可达案例（反事实）。
- 工具链：`sim2real_pointcloud_video_pipeline.py`、`generate_pointcloud_pairs_ddp.py`、`resumable_simjob_ledger.py` 已就绪（M0 验证通过）。

## D 层：真实双臂轨迹（M5 前置，待实机）

- 严格时间同步：RGB-D + 腕相机 + 关节 + 本体 + 夹爪 + 力/触觉 + 动作 + 结果 + 失败类别。
- 采集纪律：只在仿真/受控夹具安全制造区分性反事实轨迹；严禁为数据制造危险。
- 验收门槛：`AC-V-JEPA 双臂部署：实机时延与控制频率验收协议.md`。

## 版本与登记规范

- 每层就绪后在此追加「就绪日期 + 内容摘要 + SHA/大小 + 许可 + 存放路径」条目。
- 数据集版本号与 `dataset_commit_to_acvjepa_windows.py` / `elastic_data_cursor_ledger.py` 的 dataset_commit 字段保持一致（训练可追溯）。
- 禁止：将原始视频/点云文件本身提交到本 git 仓库（仅登记清单）。

## 当前状态

| 日期 | 事项 | 状态 |
|---|---|---|
| 2026-08-15 | M1：`vjepa_backbone.py` 适配器 + `vjepa_backbone_smoke.py`（随机初始化路径） | 完成（smoke PASS） |
| 2026-08-15 | A 层官方权重下载（438.9 MB，SHA 见上） | **完成**（下载 + HF 真实加载验证 PASS，391/395 键） |
| 2026-08-15 | B 层采集、C 层 RoboCasa 装配 | 待执行（M2 前置） |
