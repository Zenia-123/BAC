# Lowdim FP16 本轮交付与验证

日期：2026-10-17。分支：`feature/lowdim_fp16`。远程：`git@github.com:Zenia-123/BAC.git`。

## 完成范围

完成现有 checkpoint 的 Transformer-only AMP 精度切换、完整采样审计、特征采集、ACS 调度、BUA 消融、动作误差、计时和真实 Block Pushing 闭环验证。未训练模型，未重装环境。默认精度仍为 FP32。

`diffusion_policy/` 已从子模块转换为 BAC 直接追踪的普通源码：369 个原受控文件内容哈希保持不变，保留原许可证、资产与本地七个文件的既有修改。新增来源说明 `diffusion_policy/VENDORED.md`。原子模块 Git 历史已备份为本机 `.git/lowdim-full-cy8w51br/diffusion-policy-history.bundle`，原 Git 目录也保留；远程运行不依赖这些本机元数据。

本轮开始时工作区和暂存区的 diff、源码快照及文件哈希均保存在 `.git/lowdim-full-cy8w51br/`。用户的 `lowdim_fp16.md` 和原 P2 教学文档保留并一并同步。旧 patch 与 manifest 留作历史记录，当前恢复脚本遇到普通源码会提示无需 patch，并安全退出。

## 已执行的验证

| 项目 | 结果 |
|---|---|
| 6 个 CPU 回归测试 | 通过：DP 与穷举一致、预算检查、步数变化时末步索引、完整调度检查与 BUA 并集、Threshold 1 精确 FP32 回归、FP16 统计先转 FP32。 |
| 18 种模式 × 3 个噪声 seeds | 完整采样通过；每次均检查 100 次模型调用、100 次 FP32 Scheduler 调用、有限输出及缓存实际执行次数。 |
| 单次 Transformer forward 审计 | AMP 的 Linear / SA / CA / head 输出为 FP16；参数、模型入口、部分残差与归一化路径保留 FP32。 |
| 8 种模式 × 3 个真实环境 seeds | 共 24 episodes；本轮各模式均得到 scores `[1.0, 1.0, 1.0]`。 |
| 18 种模式的同步计时 | 每种预热 3 次、测量 10 次；候选间回收循环引用后复测。旧 summary 保留在原始结果目录。 |
| 传统 `collect_activations` CLI | 实际 checkpoint / 环境运行通过，24 个 SA/CA/FFN 分支各保存 100 步 FP16 特征。 |
| 传统 `get_optimal_cache_update_steps` CLI | FP16 目录读取、FP32 相似度统计与 25 个模块的预算 5 调度保存通过，路径正确。 |
| THOP 真实 forward | FP32 / AMP 输出 dtype 正确；两者 MACs 相同；日志区分实际参数量与 THOP 部分统计。 |
| 源码保存检查 | 369 个文件哈希与迁移前一致；当前索引中不存在 160000 gitlink。 |

本机日志保存在 `.git/lowdim-full-cy8w51br/`，大体积原始数据保存在 `results/lowdim_fp16_20261017_full/`，均未上传。随 Git 保存的完整结构化摘要为 `reports/lowdim_fp16/summary.json`，不含 checkpoint 或激活 Tensor。

## 结果应怎样理解

最终复测：FP32 Original 约 557 ms，AMP Original 约 655 ms；AMP 单独使用在本机更慢。AMP Original 相对 FP32 的 action MAE（三个噪声 seeds 均值）约 `6.59e-6`。AMP 加 ACS 20、使用 FP32 调度约 224 ms，相对 FP32 Original 加速约 2.49 倍。

速度、误差和 BUA 的效果应逐项读取 [P5_lowdim_fp16_results.md](P5_lowdim_fp16_results.md)，不能把 FP16 与缓存的收益混为一谈。BUA 并不保证每组都更快或误差更小。预算 5 的模式未做闭环评估；预算 20 的三个环境 seeds 也不足以得出论文规模的泛化结论。校准只使用一个初始观测窗口，后续扩大统计量时应保留本轮结果并使用新的输出目录。

具体修改位置、作用、调用链、完整运行命令和以后如何 Git 提交，见 [LOWDIM_FP16_RUNBOOK.md](LOWDIM_FP16_RUNBOOK.md)。

## 远程保存与恢复验证

已推送至 `origin/feature/lowdim_fp16`：`a939393`（导入完整源码）和 `74b4761`（实验、修复、报告与已有学习笔记）。

对 GitHub 进行独立的浅克隆和稀疏源码检出，未依赖本地 Git 对象复制：远程 HEAD 的全部文件路径、类型和 blob 内容哈希与本地主仓库完全一致，且不存在 160000 gitlink。实际从远程取回的 178 个原始 Python 文件 SHA-256 均与迁移前一致；BACInfer、Policy 和 Transformer 的导入路径均指向新检出目录；该目录内再次通过全部 6 项测试。验证日志：本机 `.git/lowdim-full-cy8w51br/remote-verification.log`。

远程恢复不需要 patch、不需要另一个私有仓库，也不需要 `git submodule update`。模型 checkpoint 与原始激活未上传，复跑实验时仍需本机 checkpoint 和现有 `bac` 环境。本次恢复验证没有重跑已完成的全部 GPU / rollout 实验。
