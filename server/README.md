# 原版 π0：Linux GPU 服务器准备

此目录中的程序供云端运行。Mac 的 `.venv` 只安装客户端和仿真依赖。

## 1. 安装固定版本

以下在 Linux NVIDIA GPU 服务器执行，假设已有 Conda、Git 和可用的 NVIDIA 驱动。RTX 4090 24GB 可作为起步配置，实际需求取决于模型和运行设置。

```bash
conda create -n pi0-serve python=3.11 -y
conda activate pi0-serve
python -m pip install uv

git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout --detach 215abfb217dbac7d5f1273282331b9b1866c0479

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

服务器推理入口不调用 Git 子模块里的仿真代码；如果随后使用官方 LIBERO 等示例，再按官方文档初始化相应子模块及独立仿真环境。

## 2. 上传本项目的 π0 专用服务入口

以下在 Mac 执行，替换 SSH 端口、用户名和服务器 IP：

```bash
scp -P SSH端口 \
  '/path/to/pi0-repro/server/serve_pi0.py' \
  用户名@服务器IP:~/serve_pi0.py
```

## 3. ALOHA：公开的原版 π0 检查点

在服务器 openpi 根目录内执行：

```bash
conda activate pi0-serve
uv run "$HOME/serve_pi0.py" --profile aloha_sim
```

固定使用：

```text
config:     pi0_aloha_sim
checkpoint: gs://openpi-assets/checkpoints/pi0_aloha_sim
model:      Pi0Config(pi05=False)
action:     14 维 ALOHA 双臂关节目标
listen:     127.0.0.1:8000
```

首次需要下载权重并进行 JAX 编译。服务端环境和权重应放在服务器持久磁盘上。

如你已将同一检查点下载到服务器，可指定根目录：

```bash
uv run "$HOME/serve_pi0.py" \
  --profile aloha_sim \
  --checkpoint /data/checkpoints/pi0_aloha_sim
```

检查点根目录需要有模型参数和对应的 `assets` 归一化统计。只下载 `params` 而缺少统计数据，不能当成完整部署。

## 4. LIBERO：只接受原版 π0 微调检查点

```bash
uv run "$HOME/serve_pi0.py" \
  --profile libero \
  --checkpoint /data/checkpoints/pi0_libero/实验名/训练步数
```

路径必须换成你实际训练或取得的配套检查点。该入口不会凭空提供微调权重，也不会自动退回 π0.5 或 FAST。

原版 `pi0_libero` 在本固定提交下使用默认动作块长度 50；不要照搬 FAST 示例中长度 10 的假设。输出经 LIBERO 变换后为每步 7 维动作。

## 5. 从 Mac 连接

先在 Mac 运行 `scripts/ssh_tunnel.sh`。云端默认只监听回环地址；普通使用无需开放公网 8000 端口。

Mac 客户端会要求本服务发送的模型元数据。直接使用官方 `serve_policy.py` 时其默认元数据未必包含这些字段，因此本项目客户端会提示改用 `serve_pi0.py`，不会静默跳过模型检查。

## 6. 本项目香蕉 LoRA 微调

本次已连接用户的 AutoDL 4090D，复用 `/root/pi0-repro/openpi/.venv` 与已有 `pi0_aloha_sim` 权重。数据已上传并通过 SHA-256 校验，新训练统计已生成。

当前 sidecar 文件放在云端 `server/`，包括 `banana_config.py`、`banana_finetune.py`、`compute_banana_norm_stats.py`、`run_banana_train.sh` 和扩展的 `serve_pi0.py`；根目录旧服务入口仍供原传方块流程使用。

香蕉检查点用 `server/serve_pi0.py --profile banana --checkpoint 实际检查点目录` 加载，同目录需要保留 `banana_config.py`，项目根目录需要有 `configs/banana_finetune.toml`。Mac 使用 `configs/banana_finetuned_pi0.toml`。

实际训练配置、tmux 日志、恢复与部署步骤见 [香蕉 LoRA 训练](../docs/香蕉LoRA训练.md)。

## 当前香蕉成功部署

原版 π0 最终检查点为 `checkpoints/pi0_banana_lora/banana_v3_roi2000/1999`，训练配置为 `configs/banana_finetune_roi.toml`。服务已运行在 `pi0-banana-roi-serve`，客户端使用原图 ROI 和完整 50 步块。实际结果与重启方法见 [成功报告](../docs/香蕉任务成功复现.md)。
