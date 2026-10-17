# diffusion_policy 子模块恢复（方案 B）

> 历史说明：当前分支已将 `diffusion_policy/` 转为 BAC 直接跟踪的源码目录。
> 新克隆无需初始化子模块或应用补丁；直接编辑、测试并提交源码即可。
> 本目录保留早期 P1 的原始补丁和校验记录，只适用于迁移前的提交。
> 当前说明见 `diffusion_policy/VENDORED.md` 和 `reports/LOWDIM_FP16_RUNBOOK.md`。

主仓库保持公开子模块基准 `5ba07ac6661db573af695b419a7947ecb704690f`，修改通过本目录的完整 patch 交付，不需要第二个私有仓库。不要把仅存在本机的子模块新提交作为主仓库 gitlink 推送。

## 新目录恢复

```bash
git clone --branch feature/lowdim_fp16 --recurse-submodules git@github.com:Zenia-123/BAC.git
cd BAC
python scripts/apply_diffusion_policy_patch.py --check
python scripts/apply_diffusion_policy_patch.py
python scripts/apply_diffusion_policy_patch.py --check
```

脚本只依赖 Python 标准库和 Git，不安装环境。若 GitHub HTTPS 子模块访问受限，可在本机为这个明确地址配置可用代理 URL 后初始化；本项目不把特定网络代理写入共享 `.gitmodules`。

脚本固定校验基准提交、patch SHA-256 和 7 个目标文件 SHA-256。首次应用要求子模块工作区干净；部分应用、其他本地修改、错误基准或损坏的 patch 会被拒绝，不会自动 reset、清理或覆盖。已经完全应用时再次执行只报告文件校验通过，不重复打补丁。

应用后，子模块仍在公开基准提交，但内部会显示 7 个修改文件，这是方案 B 的预期状态。后续子模块修改需要更新完整 patch 与 manifest，并再次验证新目录恢复。不要直接 `git add diffusion_policy` 把仅存在本机的提交指针带入主仓库。

## 保存内容

`lowdim_fp16.patch` 保存以下子模块内部路径的全部现有修改：

- `diffusion_policy/policy/diffusion_transformer_lowdim_policy.py`：FP32 默认精度属性、CUDA FP16 AMP forward 与 FP32 Scheduler 边界，以及已有注释。
- `diffusion_policy/env_runner/blockpush_lowdim_runner.py`：已有 shared_memory 兼容设置。
- `diffusion_policy/gym_util/async_vector_env.py`、`sync_vector_env.py`：已有 Gym reset/concatenate 兼容适配。
- `diffusion_policy/model/diffusion/transformer_for_diffusion.py`：已有学习注释与空白。
- `diffusion_policy/workspace/train_diffusion_transformer_lowdim_workspace.py`：已有 output_dir 支持。
- `setup.py`：已有 namespace 包发现支持。

`manifest.ini` 记录准确基准与文件校验值。原有 Transformer 的 15 处行尾空白原样保留；脚本的 `--whitespace=nowarn` 用于忠实恢复原内容，没有运行格式清理。原项目 LICENSE 与版权声明保持不变。

本机另有保存相同内容的安全提交 `386b7dedb680034f666124d34f790a4fafea27d5`；它不作为本方案的远程恢复依赖。本机子模块停在该提交时，脚本也会通过目标文件校验并跳过应用；主仓库 status 可能因本地提交与固定 gitlink 不同而标记子模块修改。

## 验证范围

P1 的真实 checkpoint 验收见 `reports/P1_precision_switch.md`。本目录的校验只证明源码恢复一致，不代替模型推理测试，也不包含 checkpoint、数据或环境安装。
