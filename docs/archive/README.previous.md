> Historical snapshot: this is the project README as it stood before the documentation reorganization on 2026-10-24. Original prose and code blocks are preserved; relative Markdown links are rebased for this location. Runtime statements, results, and next steps describe their original context and have not been reverified. Use the [current overview](../../README.md) and [documentation index](../README.md) for maintained guidance.

---

# 原版 π0 复现：Mac 仿真客户端 + 云端 GPU 推理

本目录只管理 **原版 π0（Flow Matching）** 的复现。Mac 负责仿真、观测处理、远程请求和结果保存；模型在 Linux NVIDIA GPU 服务器运行。

**香蕉任务已实际成功。最终配置的默认重复确认 3/3、验证布局 4/4、测试布局 4/4 均成功。模型、视频和逐次结果见 [成功复现报告](../../docs/香蕉任务成功复现.md)。**

当前 Mac 运行：

```bash
bash scripts/run_mac.sh run --config configs/banana_finetuned_pi0.toml
```

云端 ROI 服务和 Mac SSH 隧道本次已建立。模型为 `banana_v3_roi2000/1999`，正式配置包含原始图像、统一 ROI、完整 50 步动作块及速度限制。

## 1. 为什么默认用 ALOHA 仿真

原版 π0 的官方公开检查点 `pi0_aloha_sim` 与 ALOHA 双臂传递方块任务配套，可以作为第一条复现路径。

你此前选择的 LIBERO 方向仍保留在 `configs/libero.toml`，但需注意：

- 本项目中的原版 `pi0_libero` 配置需要配套的已微调检查点。
- 当前官方 `--env LIBERO` 默认指向 π0.5，不能用它来代表原版 π0 的复现。
- 此 Mac 环境没有安装旧版 LIBERO，也没有用 robosuite Lift 场景冒充 LIBERO。
- `infer-file` 可以从 Mac 发送按 LIBERO 格式保存的真实观测；完整 LIBERO 仿真和微调建议留在 Linux 环境。

如果你的目标是复现原版 π0 在 LIBERO 上的指标，后续应在 Linux 上微调 `pi0_libero`，再接入该接口；ALOHA 的结果不等于 LIBERO 的结果。

## 2. 项目目录

```text
pi0复现/
├── README.md                       # 从这里开始
├── 项目进度.md                      # 已完成事项和待办
├── pyproject.toml                  # 本地客户端包和直接依赖
├── .python-version                 # Python 3.11
├── .venv/                          # 此项目独立环境
├── .runtime/python311/             # 此项目专属 Python 运行时
├── .vscode/                        # 本目录的解释器与调试配置
├── pi0复现.code-workspace           # 用 VS Code 打开此文件
├── repos/openpi/                   # 官方仓库，Git submodule 固定提交
├── sources.lock.json               # 上游来源和提交记录
├── requirements/
│   ├── macos-arm64.in               # 直接依赖
│   ├── macos-arm64.lock.txt         # 本机完整依赖版本快照
│   ├── python-runtime.osx-arm64.explicit.txt # Python 运行时制品记录
│   └── environment.json            # 安装环境记录
├── configs/
│   ├── aloha_sim.toml              # 默认：原版 π0 + ALOHA
│   ├── banana_box.toml             # 香蕉与开口盒子的本地摆放
│   ├── banana_pi0.toml             # 香蕉任务的远程 π0 控制
│   └── libero.toml                 # 原版 π0 的 LIBERO 单次观测接口
├── src/pi0_repro/                  # 本地客户端、仿真循环、数据适配
├── scripts/
│   ├── setup_mac.sh                # 重建本地环境
│   ├── ssh_tunnel.sh               # 建立 SSH 隧道
│   └── run_mac.sh                  # 始终调用本项目解释器
├── server/
│   ├── serve_pi0.py                # 云端专用：限定原版 π0
│   └── README.md                   # 云端部署和启动步骤
├── docs/
│   ├── 香蕉入盒场景搭建.md           # 自定义场景、预览和参数说明
│   ├── 香蕉夹取与π0运行.md            # 连接模型并执行香蕉任务
│   ├── 数据接口.md
│   └── 环境与版本管理.md
├── data/                           # 自己准备的观测文件，不进入 Git
└── outputs/                        # 视频、动作、延迟和运行记录，不进入 Git
```

## 3. 环境已经安装在哪里

```text
/path/to/pi0-repro/.venv/bin/python
```

Python 3.11.16 运行时保存在本目录 `.runtime/python311`，`.venv` 基于它建立，不依赖先前学习环境里的 Python，也不继承其他项目的 site-packages。

关键版本：Python 3.11、NumPy 1.26.4、MuJoCo 2.3.7、dm-control 1.0.14、gym-aloha 0.1.1、Gymnasium 1.0.0、websockets 14.1。仿真核心版本按固定提交下官方 ALOHA 示例选择；视频编码依赖使用支持 Apple Silicon 二进制的 imageio-ffmpeg 0.6.0。所有传递依赖的实际版本见 lock 文件。

Mac 端没有安装整套 openpi 模型训练依赖，没有下载模型权重，也不在 Mac 加载大模型。

### VS Code

打开 `pi0复现.code-workspace`。如果之前手动选过其他解释器，再按 `⌘⇧P` → `Python: Select Interpreter`，选择本项目 `.venv/bin/python`。

命令行启动不依赖 VS Code 的历史选择：`scripts/run_mac.sh` 会明确使用本项目解释器。

## 4. 第一次运行：三个终端

### 终端 A：GPU 服务器

先按 [云端部署说明](../../server/README.md) 安装匹配提交的官方 openpi，并上传 `server/serve_pi0.py`。然后在服务器的 openpi 根目录运行：

```bash
source /root/pi0-repro/env.sh
cd /root/pi0-repro/openpi
JAX_PLATFORMS=cuda /root/pi0-repro/openpi/.venv/bin/python \
  /root/pi0-repro/serve_pi0.py --profile aloha_sim
```

它明确加载 `pi0_aloha_sim`，默认监听服务器 `127.0.0.1:8000`。

### 终端 B：Mac 建立 SSH 隧道

```bash
cd '/path/to/pi0-repro'
bash scripts/ssh_tunnel.sh 用户名@服务器IP SSH端口
```

替换用户名、IP 和 SSH 端口，保持此终端打开。如果平台使用默认 SSH 端口 22，可省略第二个参数。

### 终端 C：Mac 仿真

```bash
cd '/path/to/pi0-repro'
bash scripts/run_mac.sh run --config configs/aloha_sim.toml
```

默认执行 1 个 episode，最多 300 步，与 gym-aloha 的默认任务时限一致，保存视频和轨迹。想显示实时相机画面：

```bash
bash scripts/run_mac.sh run --config configs/aloha_sim.toml --display
```

`--display` 显示的是仿真相机画面，不是可拖拽操作的 MuJoCo 查看器。普通 Python 启动即可；本入口不调用 `mujoco.viewer.launch_passive`。

如果只想读取服务端元数据、不发送观测：

```bash
bash scripts/run_mac.sh connect --config configs/aloha_sim.toml
```

云端启动命令对应你本次使用的 AutoDL 目录；其他服务器需按实际路径调整。本次未替你启动仿真。

## 5. 怎样确保这次是原版 π0

1. 固定 openpi 提交：`215abfb217dbac7d5f1273282331b9b1866c0479`。
2. 云端入口提供 `pi0_aloha_sim`、`pi0_libero` 和本项目注册的 `pi0_banana_lora` 配置。
3. 加载前检查 `Pi0Config(pi05=False)`，不接受 π0.5 或 FAST 配置。
4. 云端握手声明 `model_family=pi0`、配置、动作维度和源码提交。
5. Mac 要求握手信息匹配，配置不匹配时不发送推理观测。

握手是防止误连、误配的工程检查，并非独立鉴定远端权重来源。仍需使用与配置匹配的可信检查点。

官方仓库自身含其他模型源码；保留上游原貌有利于版本固定。本项目运行入口只针对原版 π0。

## 6. 输出怎么看

每次运行使用新目录，例如 `outputs/20261022-xxxxxx-xxxxxx/`：

| 文件 | 内容 |
|---|---|
| `run.json` | 配置、Python 路径、依赖版本、源码提交、系统信息 |
| `server.json` | 远端声明的模型、配置、动作维度 |
| `episode_000.mp4` | 相机画面；固定 50fps 的仿真时间回放 |
| `episode_000.npz` | 观测状态、实际执行动作和后继奖励 |
| `episode_000_latency.jsonl` | 每次请求的往返延迟、动作块形状、服务端计时 |
| `summary.json` | 步数、最大奖励、环境成功标记、终止与异常状态 |

视频的 50fps 不表示程序在墙钟时间达到 50Hz；等待网络响应时，仿真主循环会暂停步进。保存的数据用于观察和记录，不是已经完善的训练数据集：没有完整高分辨率观测、跨设备时间同步或数据清洗流程。

## 7. 自定义场景：香蕉与开口盒子

在 Mac 运行：

```bash
cd '/path/to/pi0-repro'
bash scripts/run_mac.sh scene --config configs/banana_box.toml
```

通过 `1` / `2` 切换俯视与斜视，空格暂停，`R` 重置，`S` 保存截图。调整 `configs/banana_box.toml` 中的位置后重启预览。首次画面与组装后的 XML 自动保存在 `outputs/scene-*/`。

以上 `scene` 命令用于预览，机械臂保持初始关节目标。若要让云端原版 π0 控制香蕉场景，保持服务和 8000 端口转发，然后在 Mac 执行：

```bash
bash scripts/run_mac.sh connect --config configs/banana_pd_pi0.toml
bash scripts/run_mac.sh run --config configs/banana_pd_pi0.toml --display
```

默认传方块检查点尚未针对香蕉任务微调，运行新场景不保证抓取成功。这里使用与专家采集一致的自定义 PD/指腹接触场景；其具体参数在采集记录的 scene.xml 中保存。

详细步骤见 [香蕉入盒场景搭建](../../docs/香蕉入盒场景搭建.md) 和 [香蕉夹取与 π0 运行](../../docs/香蕉夹取与π0运行.md)。

后续微调见 [香蕉场景微调路线](../../docs/香蕉场景微调路线.md)：包含演示数据要求、固定版本 LeRobot 兼容性、JAX LoRA 候选配置及部署要求。候选配置需注册、演示需人工复核和转换后，训练命令才可用。

## 8. 无需真机采集专家演示

在 Mac 本地执行：

```bash
bash scripts/run_mac.sh collect-demo --config configs/banana_expert.toml --episodes 1 --display
```

已实际采集到固定布局下的完整受控抓放候选，包含主动释放、退避和稳定等待。结果位于 `data/banana_demos/20261023-212233-b3dfa8/episode_000/`，1190 组 RGB/状态/动作，当前等待你回看确认。

采集不需要云端 GPU。详细操作、物理参数和复核命令见 [仿真专家演示采集](../../docs/仿真专家演示采集.md)。这是脚本专家的结果，不代表 π0 已经学会该任务。

## 9. 下一步

- 实际运行 ALOHA 基线，记录多次任务结果和延迟。
- 回看并复核专家候选，扩大有效布局覆盖，再完成 LeRobot 转换与原版 π0 微调。
- 如继续 LIBERO：准备原版 π0 的 LIBERO 微调检查点，并在 Linux 上运行匹配仿真。

服务连接成功不代表已经复现论文成功率；任务执行结果与训练效果需要另行评估。

## 来源

- [openpi 官方仓库，固定提交](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)
- [该提交的 ALOHA 仿真示例](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479/examples/aloha_sim)
- [官方远程推理协议](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/docs/remote_inference.md)
- [gym-aloha](https://github.com/huggingface/gym-aloha)

你发来的 pi0_fast_deploy 用于了解远程部署思路；本目录直接使用官方 openpi 客户端协议及 ALOHA 示例，不依赖其 FAST 检查点或 Lift 演示。

## 多布局演示采集

```bash
bash scripts/run_mac.sh collect-dataset --config configs/banana_dataset.toml
```

固定种子采集 24 个局部不同布局，预先划分 16/4/4 个训练、验证、测试候选。每条保存实际场景 XML、布局、视频和同步观测/动作；失败保留，成功仍需人工回看。详见 [多布局演示数据集](../../docs/多布局演示数据集.md)。

本批实际结果：已实际采集 24 个不同布局，得到 24 条自动成功候选、28575 组同步 RGB/状态/动作；均已通过用户委托的助手筛选，并完成 LeRobot v2.1 转换。

[打开本批采集报告](../../data/banana_demos/20261023-213415-07fb84/采集报告.md)（包含逐条视频与布局清单）。

## 演示筛选与 LeRobot 转换已完成

第一批 24 条全部接受，28,575 帧已转换为独立的训练/验证/测试数据集（16/4/4 条）。审核者标记为助手，保留逐条证据。

[结果报告](../../data/lerobot/20261023-213415-07fb84/README.md) · [使用说明](../../docs/LeRobot数据转换.md)
