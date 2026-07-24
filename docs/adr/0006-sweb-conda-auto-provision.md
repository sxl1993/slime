# ADR-0006: SWE-bench conda auto-provision

## 状态

已接受

## 上下文

SWE-bench eval 在 LocalSandbox 上运行需要 miniconda3 来创建对应 Python 版本的 conda env。
当前代码在 `/opt/miniconda3` 不存在时只打 warning 然后 eval.sh 静默降级到系统 Python，
导致测试因 Python 版本不匹配大概率失败。

## 决策

1. **conda auto-provision**：在 `_ensure_conda_env` 内部检测 conda 是否可用，
   不可用时自动下载并静默安装 miniconda3（`-b -p <path>`），无需外部前置步骤。
2. **镜像源**：miniconda 安装包和 conda package channel 均走清华源
   （`https://mirrors.tuna.tsinghua.edu.cn/anaconda/`），
   不改全局 `.condarc`，每次 `conda create` / `conda install` 时通过
   `--override-channels -c <mirror>` 显式指定。
3. **安装路径**：默认 `/opt/miniconda3`，可通过 `SWEB_CONDA_ROOT` 环境变量覆盖。
4. **失败处理**：miniconda 安装或 conda env 创建失败时，打 error 日志并跳过当前
   instance，不中断整个 batch。不静默降级到系统 Python。

## 理由

- 官方源在当前网络环境不可达，必须走清华源
- 不改全局 `.condarc` 避免影响用户机器上其他 conda 环境
- instance-level fail 比 batch-level fail 更合理：500+ instance 中一个失败不应阻塞其余