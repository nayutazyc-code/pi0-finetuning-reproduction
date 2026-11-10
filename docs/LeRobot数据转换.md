# 香蕉演示筛选与 LeRobot 数据转换

## 版本与范围

本项目固定的 openpi 提交是 `215abfb217dbac7d5f1273282331b9b1866c0479`。其依赖锁定 LeRobot 提交 `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`，对应数据格式 **v2.1**。LeRobot Python 包自身的版本号为 0.1.0，与数据格式版本不是一回事。

转换使用该提交的真实 `LeRobotDataset.create / add_frame / save_episode`，不是仅把 NPZ 改名为 Parquet。源码提交及下载归档哈希保存在 `requirements/lerobot-source.json`。

[固定版本 LeRobot 数据 API](https://github.com/huggingface/lerobot/blob/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5/lerobot/common/datasets/lerobot_dataset.py)

## 已完成的筛选

用户明确委托助手回看第一批 24 条演示。助手查看每条原始 RGB 关键阶段总览和 8 个回放侧视阶段，并完整物理回放全部 28,575 个原记录动作，对照来源状态与物体位置。逐条结果与限制记录在：

[助手筛选报告](../data/banana_demos/20261023-213415-07fb84/assistant_review/筛选报告.md)

24 条全部接受，无剔除、无待定。`reviewed_by=assistant`、`review_status=accepted`；`human_review=pending` 表示用户未亲自回看。用户的明确委托允许这些演示进入转换，没有冒充人工亲审。

关键阶段抽帧不等于逐帧人工观看整个视频。记录中保留每条实际查看的帧索引、图像哈希、完整回放指标和接触解释。

## 独立数据环境

- `.venv`：原有 Mac MuJoCo 仿真环境。
- `.venv-data`：仅供数据转换与固定版本读取。
- `.vendor/lerobot-<commit>`：固定 LeRobot 源码，启动脚本显式设置 `PYTHONPATH`。
- `requirements/lerobot-data.in`：数据 API 所需依赖子集。
- `requirements/lerobot-data.lock.txt`：本次实际安装版本。

该环境不包含完整真机驱动或模型训练依赖。转换不加载 π0 权重、不需要 GPU、不上传 Hugging Face。

环境需要重建时执行 `bash scripts/setup_data.sh`；它使用实际依赖锁并校验固定源码。

## 运行转换

```bash
cd '/path/to/pi0-repro'
bash scripts/run_data.sh \
  --source data/banana_demos/20261023-213415-07fb84
```

输出默认在 `data/lerobot/<原始session>/`。已存在结果或 `.partial` 时拒绝覆盖；需要重新转换时用 `--output-root` 指定新目录。转换过程先写 `.partial`，全部完成并读回后才改为正式目录。

## 三份独立数据集

```text
data/lerobot/20261023-213415-07fb84/
  dataset_manifest.json
  local/
    pi0_banana_train/          # 16 条，19065 帧
    pi0_banana_validation/     # 4 条，4733 帧
    pi0_banana_test/           # 4 条，4777 帧
```

每份都有 `meta/info.json`、`meta/episodes.jsonl`、`meta/episodes_stats.jsonl`、`meta/tasks.jsonl`、逐 episode Parquet、`source_episodes.json` 和 `conversion_audit.json`。

三份 repo_id 分别是 `local/pi0_banana_train`、`local/pi0_banana_validation`、`local/pi0_banana_test`。它们只是本地命名，不表示已经发布到 Hub。

固定版本 LeRobot 在每个独立数据集的内部 metadata 中默认使用 `train` 数据段名；这里的训练/验证/测试意义由三个独立 repo_id 和 `source_split` 明确区分。训练时只指向 `local/pi0_banana_train`。

## 字段与对齐

| LeRobot 字段 | 来源 | 单帧含义 |
|---|---|---|
| `observation.images.top` | `rgb/第t帧.png` | 原始 640×480 RGB，无缩放；PNG 内容嵌入 Parquet |
| `observation.state` | `trajectory.npz/state[t]` | 14 维实际状态，float32 |
| `action` | `trajectory.npz/action[t]` | 14 维经过限幅后实际提交的关节/夹爪目标 |
| `timestamp` | 官方写入器按 `frame_index / 50` 生成 | 与原始时间戳读回比较 |
| `task_index` | 原始语言指令 | 通过 `meta/tasks.jsonl` 恢复文字 |

14 维顺序为左臂 6 个关节、左夹爪、右臂 6 个关节、右夹爪。关节是弧度位置目标，夹爪是归一化目标；不是 OpenVLA 的 7 维末端增量。

状态和图像都在对应动作执行前获得。不额外加入视频末尾的最终画面。`requested_action`、`next_state`、物体真值和阶段信息保留在原数据，不作为额外 VLA 训练输入。

图像使用 `image` 类型，PNG 嵌入逐轨迹 Parquet；不用有损 MP4 代替训练图像。移动整个输出目录即可携带图像，不依赖原始临时 PNG 路径。

## 实际转换时的核对

转换器逐轨迹读回 Parquet，比较所有状态、动作和每张图像；若 PNG 编码字节不同则解码比较像素。通过后再用同一提交的真实 LeRobot 读取器读取三份数据，构造 50×14 动作块，并检查首条轨迹的末尾 padding 不越过 episode。

结果写入 `conversion_audit.json` 和总 manifest。任一步失败都不会标记为 `complete`。

## 在云端 openpi 中读取

把完整输出目录传到服务器后，让 `HF_LEROBOT_HOME` 指向包含 `local/` 的目录，例如：

```bash
export HF_LEROBOT_HOME=/root/pi0-repro/data/lerobot/20261023-213415-07fb84
```

使用服务器现有固定版本 openpi 环境读取：

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

train = LeRobotDataset(
    "local/pi0_banana_train",
    delta_timestamps={"action": [i / 50 for i in range(50)]},
)
item = train[0]
print(item["observation.state"].shape)        # torch.Size([14])
print(item["action"].shape)                   # torch.Size([50, 14])
print(item["observation.images.top"].shape)   # torch.Size([3, 480, 640])
```

`LeRobotAlohaDataConfig` 的默认重映射正好使用 `observation.images.top / observation.state / action`。后续应采用 `adapt_to_pi=True`，并沿用原 ALOHA Sim 的 `use_delta_joint_actions=False`，除非同时明确更改训练/部署动作约定。

LeRobot 的原始统计不是 openpi 的归一化资产。必须在训练划分上，经过 ALOHA 适配后计算/选择 openpi 统计；验证和测试数据不参与拟合。

香蕉任务的 LoRA 配置已注册，数据已上传并通过全部 50 个文件的 SHA-256 校验，训练专用统计已生成，云端试训已启动。实际状态、操作和微调权重部署见 [香蕉 LoRA 训练](香蕉LoRA训练.md)。闭环模型成功率仍待评估。

## 本次实际输出

[打开结果报告](../data/lerobot/20261023-213415-07fb84/README.md)。状态为 complete，24 条、28,575 帧，三份数据正式目录均已读回。
