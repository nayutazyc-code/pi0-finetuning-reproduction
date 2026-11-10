# 香蕉入盒：原版 π0 LoRA 训练与部署

> 2026-10-24 后续更新：第三轮已成功完成任务；当前正式配置、最终 3/3、4/4、4/4 结果和运行方法见 [成功复现报告](香蕉任务成功复现.md)。本文保留对应阶段的历史记录。


更新：2026-10-24。这里记录本项目实际使用的配置和路径。首轮已正常完成 2,000 次更新，最终检查点为 `1999`，退出码 0；详见 [首轮训练结果](首轮香蕉LoRA训练结果.md)。

## 1. 本轮训练在学什么

输入顶视 RGB、当前双臂 14 维状态和任务文本，学习输出未来 50 步的动作序列。动作是 14 维 ALOHA 绝对关节/夹爪目标；不是末端位姿增量，也不是 OpenVLA 动作 token。

初始化使用本机云服务器已有的 `pi0_aloha_sim` 参数；模型保持原版 π0，`pi05=False`。官方 openpi 固定到 `215abfb217dbac7d5f1273282331b9b1866c0479`，本项目在 `server/banana_config.py` 中注册独立配置，不修改官方源码。

本批演示只有右臂执行抓放，左臂基本不动。它用于验证局部香蕉任务的微调链路；还不能代表大范围位置泛化、多物体选择或语言理解能力。

## 2. 数据与划分

数据目录：`data/lerobot/20261023-213415-07fb84`。云端根目录为 `/root/pi0-repro`，Mac 根目录为 `/path/to/pi0-repro`。

| 用途 | LeRobot repo_id | 轨迹数 | 帧数 |
|---|---|---:|---:|
| 训练 | local/pi0_banana_train | 16 | 19,065 |
| 验证 | local/pi0_banana_validation | 4 | 4,733 |
| 最终测试 | local/pi0_banana_test | 4 | 4,777 |

24 条演示经用户委托的助手回看与完整物理回放筛选，审核身份记录为助手；人工复核仍标为 pending。数据为固定版本 LeRobot v2.1，RGB 以无损 PNG 嵌入 Parquet，状态和动作为 float32。云端上传已通过全部 50 个文件的 SHA-256 校验。

训练只加载训练 repo；统计只使用训练 repo。`meta/info.json` 中的库内部 split 名称不会替代这里三份独立 repo 的划分。

## 3. 归一化

实际统计位于：

```text
assets/pi0_banana_lora/local/pi0_banana_train/norm_stats.json
assets/pi0_banana_lora/local/pi0_banana_train/normalization_report.json
```

先使用官方 `AlohaInputs(adapt_to_pi=True)` 对应的数值变换，再统计。state 的夹爪变换与 action 的夹爪变换不同，不能直接拿原始 NPZ 算一套均值方差替代。

本轮是原版 π0 的 z-score 归一化：

\[
\widetilde{x}_i=\frac{x_i-\mu_i}{\sigma_i+10^{-6}}.
\]

state 统计使用 19,065 行；actions 统计使用每帧未来 50 步，共 953,250 行。每条轨迹末尾重复自己的最后一个动作，不跨 episode 取动作。实现复用官方 ALOHA 数值变换和 RunningStats，通过列式读取避免解码全部图像；分位数直方图的更新方式可能与官方分批脚本略有差异，但此配置使用 mean/std。

左臂多个维度方差极小或为零，是演示中左臂静止造成的。报告记录了这些维度；没有擅自增大标准差。部署必须使用训练检查点携带的新统计，不要沿用传方块任务的旧统计，也不要只在推理端改标准差。

在云端重新计算的入口：

```bash
JAX_PLATFORMS=cpu /root/pi0-repro/openpi/.venv/bin/python \
  /root/pi0-repro/server/compute_banana_norm_stats.py
```

已有统计时程序拒绝覆盖。需要重算应先保留旧统计并明确建立新实验，避免训练和部署版本混淆。

## 4. 本轮超参数

配置文件：`configs/banana_finetune.toml`。

| 参数 | 本轮值 |
|---|---|
| GPU | RTX 4090 D，24 GB |
| 模型 | 原版 π0 + Gemma 2B / 300M expert 的 LoRA |
| 训练方式 | 仅名称含 lora 的参数可训练，其余模型参数冻结 |
| 输入图像 | 顶视图变换为 224×224；缺失腕部视角有 mask |
| 模型动作维度 | 32，实际 14 维经官方变换后补零 |
| 动作块 | 50 步 |
| batch size | 1 |
| 首轮更新步数 | 2,000 |
| 学习率 | warmup 100 步，峰值 1e-5，余弦衰减到 1e-6 |
| 随机种子 | 42 |
| EMA / WandB | 关闭 / disabled |
| 保存 | 每 500 步及最终一步；keep_period=1000 |

**这是首轮链路试训配置，没有宣称是最优超参数。** 严格只训练 LoRA 与官方默认 LoRA 冻结过滤器略有区别：官方默认过滤器还可能允许视觉等其他参数更新，本轮将这些参数冻结，以适应 24 GB 显存和小数据。初始化日志共含 3,288,036,112 个浮点参数元素，其中 LoRA 为 49,987,584（约 1.52%）；未发现非 LoRA 的 float32 参数数组。实际可训练集合由参数过滤器决定，精度日志用于核对。

模型训练损失是 Flow Matching 向量场回归损失。loss 下降说明它更贴合训练动作分布；是否会抓香蕉，需要闭环仿真评估。

## 5. 查看训练任务与历史日志

Mac 登录：

```bash
SSH_PORT=2222  # 示例值，请替换为服务器的实际 SSH 端口。
ssh -i ~/.ssh/your_ssh_key -p "$SSH_PORT" user@your-server.example.com
```

云端查看日志和 GPU：

```bash
tail -f /root/pi0-repro/logs/banana_pilot_v1.log
nvidia-smi
```

首轮任务已结束，`pi0-banana` 会话会随主进程退出而消失。下面的 tmux 操作用于训练仍在运行时；Mac 断网或 VS Code 关闭不会主动结束 tmux 中的训练。若要进入尚在运行的会话：

```bash
tmux attach -t pi0-banana
```

本轮输出已重定向到日志，tmux 面板可能没有实时文字；使用 `tail -f` 查看。退出 tmux 而保留任务，按 `Ctrl+B`，松开，再按 `D`。

进程结束后，退出码保存在 `logs/banana_pilot_v1.exit-code`。0 表示正常结束；还需确认日志有最终保存完成记录和完整检查点。

## 6. 新实验与断点继续

不要在当前实验仍运行时重复启动。同一个实验目录不会自动覆盖。

新实验示例（先按实际需要调整配置、实验名）：

```bash
tmux new-session -d -s pi0-banana-v2 \
  'bash /root/pi0-repro/server/run_banana_train.sh banana_pilot_v2 > /root/pi0-repro/logs/banana_pilot_v2.log 2>&1'
```

中断后，仅在存在完整检查点且原进程已退出时继续：

```bash
tmux new-session -d -s pi0-banana-resume \
  'bash /root/pi0-repro/server/run_banana_train.sh banana_pilot_v1 --resume >> /root/pi0-repro/logs/banana_pilot_v1.log 2>&1'
```

还未保存第一个检查点时，`--resume` 无法恢复已丢失的更新；先保留失败日志和目录，再用新的实验名启动。

检查点在：

```text
/root/pi0-repro/checkpoints/pi0_banana_lora/banana_pilot_v1/<step>/
```

官方循环从 0 计步，本轮 2,000 次更新的最终目录预计是 `1999`。以实际完整目录为准。检查点包含基础参数和训练状态，并非只有很小的 adapter 文件，需留足磁盘。

## 7. 训练后让 Mac 使用新策略

先确认训练已结束，检查点包含 `params`、`assets` 等完整内容；服务器显存不宜同时加载训练和另一套推理模型。

云端部署示例，替换成实际存在的检查点：

```bash
source /root/pi0-repro/env.sh
JAX_PLATFORMS=cuda /root/pi0-repro/openpi/.venv/bin/python \
  /root/pi0-repro/server/serve_pi0.py \
  --profile banana \
  --checkpoint /root/pi0-repro/checkpoints/pi0_banana_lora/banana_pilot_v1/1999
```

Mac 建立隧道，保持该终端打开：

```bash
SSH_PORT=2222  # 示例值，请替换为服务器的实际 SSH 端口。
ssh -i ~/.ssh/your_ssh_key -p "$SSH_PORT" \
  -o ExitOnForwardFailure=yes -N -L 8000:127.0.0.1:8000 \
  user@your-server.example.com
```

Mac 另一个终端：

```bash
cd '/path/to/pi0-repro'
bash scripts/run_mac.sh connect --config configs/banana_finetuned_v1_fullview.toml
bash scripts/run_mac.sh run --config configs/banana_finetuned_v1_fullview.toml
```

客户端仍使用 ALOHA 14 维协议，但要求服务端声明 `policy_config=pi0_banana_lora`。它使用演示采集时的 `position_pd` 场景，不能换回另一个控制器后直接比较结果。

这里的单次默认场景运行用于观察行为，**不是验证集或测试集成绩**。后续需要按保存的独立布局逐一还原场景、多随机种子闭环运行，记录成功率、碰撞、抓取后掉落、入盒保持等指标；用验证集选训练步数/超参数，最后在测试集报告结果。已完成两次闭环诊断，均未通过，详见 [闭环评估](香蕉Pi0闭环评估.md)；完整独立验证/测试集评估尚未完成。
