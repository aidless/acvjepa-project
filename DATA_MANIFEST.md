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

- 适配器：`vjepa_backbone.py`（键重映射：`encoder.`/`backbone.` 前缀剥离；`attn.qkv` 合并布局；`patch_embed.proj` ↔ `patch_embed` 别名）。
- 加载入口：`train_ac_vjepa_ddp.py --init-from vjepa2:<path>[:frozen|last_k|lora|finetune]`。
- 冒烟：`python vjepa_backbone_smoke.py --checkpoint <official.pt>`（无权重时走同构随机初始化路径，已 PASS）。
- 待办：下载官方权重（数百 MB）并记录 SHA-256 于此文件；许可证确认（MIT 许可，见 [HF 模型卡](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256)）。

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
| 2026-08-15 | A 层官方权重下载 | 待执行（网络/许可前置） |
| 2026-08-15 | B 层采集、C 层 RoboCasa 装配 | 待执行（M2 前置） |
