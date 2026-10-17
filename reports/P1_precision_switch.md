# P1：Lowdim Transformer FP16 AMP 开关

日期：2026-10-17（Asia/Shanghai）。实现与本阶段推理验收通过；用户最终选择 B（现有 BAC 仓库保存完整 patch 与恢复脚本）。源码恢复测试通过，正在完成主仓库 Git 交付；没有进入 P2。

## 改动与调用关系

只修改两份生产源码：

- `diffusion_policy/diffusion_policy/policy/diffusion_transformer_lowdim_policy.py`
  - `__init__` 显式新增 `inference_precision='fp32'`，保存到实例并校验取值；旧 checkpoint 配置不需要增加字段。
  - `conditional_sample` 从实例属性定义局部 `use_fp16`，修复 G0 的 NameError；也校验运行时被修改的精度属性。
  - FP16 AMP 要求 CUDA 和 FP32 condition_data，防止采样状态意外变为 half。
  - autocast 仍仅包围 Transformer forward；仅 AMP 时把模型输出转换为 FP32。FP32 分支不再额外 `.float()`。
  - 顺手去除本文件原有两处行尾空白，不改注释内容与其他既有成果。
- `BACInfer/scripts/eval_fast_diffusion_policy.py`
  - Click 新增 `--precision [fp32|fp16_amp]`，默认 fp32。
  - `main` 增加同名参数，在加载权重和 deepcopy 后、缓存安装及 warmup 前，分别设置 original/fast 的精度。
  - fp16_amp 在非 CUDA/无 CUDA 时给出明确错误；拒绝对其他 Policy 类型声称启用本次 Lowdim AMP。
  - 日志、benchmark_results 和 eval_results 记录 precision。

```text
--precision fp16_amp
  -> main(..., precision)
  -> original_policy.inference_precision = precision
     fast_policy.inference_precision = precision
  -> 各自 predict_action()
  -> conditional_sample(): use_fp16 = (precision == 'fp16_amp')
  -> with torch.autocast(..., enabled=use_fp16):
       model(trajectory, t, cond)
  -> AMP 时 model_output.float()
  -> with 外执行 scheduler.step()
```

没有修改 Transformer.forward、缓存 wrapper、训练 compute_loss、checkpoint 或模型权重精度；没有增加 predict_action/forward 的参数。精度属性没有进入 self.kwargs，因此不会被展开传给 Scheduler。

## 验证环境与输入

- 真实 checkpoint：`checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt`。
- 沿用 eval 的 `workspace.model` 权重来源。
- 现有 bac 环境：Python 3.10.0，PyTorch 2.11.0+cu128；GPU 为 NVIDIA GeForce RTX 5070 Ti Laptop GPU。
- 使用 checkpoint 中的真实配置，完整去噪步数为 100；观测为 FP32 零张量 `(1,3,16)`，动作输出 `(1,1,2)`。
- 推理在 `torch.inference_mode()` 中执行；验证后移除临时 hook，Scheduler 替代观察函数通过 unittest.mock.patch 自动恢复。
- 所有运行均为功能验收，没有环境 rollout、任务成功率、稳定计时或 FP32/FP16 误差研究。

## 命令与结果

仓库根目录：`/home/zenia/Projects/BAC/BAC`。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/zenia/cimflow/miniconda3/envs/bac/bin/python -B .git/p1-validation/validate_p1.py > .git/p1-validation/validation.log 2>&1
```

退出码 **0**，末行为 `ALL P1 CHECKS PASSED`。

验证脚本和完整日志仅保存在当前仓库 Git 元数据目录：

- `.git/p1-validation/validate_p1.py`
- `.git/p1-validation/validation.log`

它们不是已版本化测试，普通 clone 不会带回这些本地证据；本报告保留验收方法和结果。P2 会按工作计划新增版本化的独立 dtype 审计入口。

| 验收项 | 结果 |
| --- | --- |
| 老 checkpoint 加载与默认精度 | PASS：默认 fp32，precision 未混入 scheduler kwargs |
| Original FP32 完整 predict_action | PASS：100 次模型调用、100 次 Scheduler step，有限输出 |
| FP32 回归 | PASS：相同种子下 action/action_pred 与原提交采样函数逐元素完全一致 |
| 单次 FP16 AMP 模型 forward | PASS：输出 torch.float16，全部有限 |
| Original FP16 AMP 完整 predict_action | PASS：100 步，有限输出 |
| Threshold 缓存 FP32 | PASS：100 步，存在缓存、有限输出 |
| Threshold 缓存 FP16 AMP | PASS：100 步，存在缓存、有限输出 |
| Scheduler 精度边界 | PASS：上述四条完整推理路径中，输入 model_output、trajectory 和输出 prev_sample 都为 FP32；step 在 CUDA autocast 外 |
| 模型输入/条件与权重 | PASS：检查模型输入、cond（若有）、参数保持 FP32 |
| 模型原始输出 | FP32 两条路径为 torch.float32；AMP 两条路径为 torch.float16 |
| CPU AMP、非法精度、half 采样状态 | PASS：明确拒绝 |
| CLI help | PASS：显示 --precision [fp32\|fp16_amp]，默认 fp32 |
| CLI 精度传播 | PASS：真实 CLI 执行加载、deepcopy、缓存安装；检查两次 warmup 调用均收到所选精度 |
| 两份修改源码的内存 compile | PASS，退出 0，无新增 pyc |
| 主仓库 git diff --check | PASS，退出 0 |
| Policy 文件 git diff --check | PASS，退出 0 |

CLI 传播测试临时替换 predict_action 来观察属性，在检查两份 Policy 后主动停止，不执行 THOP、benchmark 或环境评估。两种精度的真实完整 predict_action 已在前面的独立验收中执行；因此不能把 CLI 传播检查误称为整套 eval 全流程通过。

FP32 回归方法：从子模块基准 `5ba07ac6661db573af695b419a7947ecb704690f` 提取原始 conditional_sample 的 AST，在内存中绑定给独立 Policy 副本；不覆盖磁盘源码。使用相同 checkpoint、输入、种子 11，与当前默认 FP32 路径比较，断言 rtol=0、atol=0。这一检查验证本次改动未改变该输入下的 FP32 结果，不是 P3 的精度误差评估。

help 可单独复查：

```bash
/home/zenia/cimflow/miniconda3/envs/bac/bin/python -B -m BACInfer.scripts.eval_fast_diffusion_policy --help
```

后续正式 eval 在已有命令中增加 `--precision fp32` 或 `--precision fp16_amp` 即可。该入口会继续执行原有计时和环境 runner，P1 本轮没有启动完整 eval。

## Python / PyTorch 说明

- `inference_precision='fp32'` 是可选形参；旧配置未提供时采用默认值。放在 `**kwargs` 前显式声明，可避免误传给 Scheduler。
- `self.inference_precision` 是当前 Policy 对象保存的配置；`use_fp16 = (precision == 'fp16_amp')` 是本次调用中的局部布尔变量。原来的裸变量现在有明确来源。
- `enabled=use_fp16` 左边是 autocast 接口参数名，右边是已定义的布尔值。
- `with` 通过缩进划定 AMP 作用范围；Transformer 内部实际执行的运算继承此上下文，Scheduler 调用位于外面。
- `model_output.float()` 返回转换后的张量，需要接收返回值；它不会把整个模型或 trajectory 一起转换。
- deepcopy 后分别赋值，确保两份独立对象配置相同；对其中一份的后续赋值不会自动更新另一份。
- `inference_mode`、`eval` 与 autocast 各有职责，分别控制求导相关记录、推理层行为、运算精度。本次 AMP 没有调用 model.half。

## Git 状态、备份与尚未完成事项

- 主分支仍为 `feature/lowdim_fp16`，本阶段交付前主仓库 HEAD 为 `7eb921d`。
- 在用户选择 A 后，子模块从 detached HEAD 建立本地分支 `feature/lowdim_fp16`，创建提交 `386b7dedb680034f666124d34f790a4fafea27d5`（`feat: add opt-in lowdim AMP inference precision`）。该提交包含 P1 Policy 修复与所有 6 个其他原有修改文件，共保存 7 个文件；提交说明明确区分 P1 功能与已有兼容适配/注释，未改写既有历史。
- 修改前另存了完整原始文件、哈希和 diff：`.git/p1-backups/20261017-165304/`。G0 备份继续保留。
- `lowdim_fp16.md` 和子模块其余 6 个原有修改文件保持原内容；本轮只编辑两份目标生产源码。
- 提交前子模块整体 `git diff --check` 因 Transformer 文件的 **15 处原有行尾空白**退出 2。本轮没有清理该只读文件，原样保存用户成果；P1 修改的 Policy 自身检查通过。提交后工作树干净不等于消除了这些历史格式问题。
- 子模块提交完成后，使用 `git clone --no-local --branch feature/lowdim_fp16` 克隆到新的临时目录，核对 7/7 个文件 SHA-256、HEAD 与 LICENSE，均通过且克隆工作树干净。证据在 `.git/p1-validation/local-restore.log`。这一私有本地提交保留为安全备份，但不作为方案 B 的远程恢复依赖。
- 用户最终明确改选 B。主仓库的 `.gitmodules` 与 gitlink 保持原公开基准 `5ba07ac`，没有给子模块配置主仓库 URL、没有创建第二个远程仓库、没有推送子模块。仅推送现有 BAC 功能分支。
- 新增 `patches/diffusion_policy/lowdim_fp16.patch`、`manifest.ini`、README 和 `scripts/apply_diffusion_policy_patch.py`。完整 patch 固定公开基准，覆盖全部 7 个文件；manifest 校验 patch 与各目标文件 SHA-256。
- 恢复脚本只依赖标准库和 Git，支持 --check 与重复执行；应用前拒绝脏工作区、错误基准和损坏 patch，应用后验证全部目标文件。现有本地安全提交已含完整内容时，只校验而不再应用。
- `.gitattributes` 仅对保存原始内容的 patch 文件关闭空白格式检查，以忠实保留已有注释和行尾空白；不改变生产源码的格式检查规则。
- 全新临时目录恢复测试退出 0：干净基准 check 不改文件；应用后 7/7 哈希与实际推理验收源码一致；重复应用不改文件；脏工作区、错误基准、损坏 patch 均拒绝且保留原字节。日志在 `.git/p1-validation/patch-restore.log`。
- 原 THOP 路径直接调用 model，绕过 conditional_sample；本次 precision 字段描述 Policy 推理精度，不能据此认为 THOP 路径也启用了 AMP。该边界保留给后续性能核验。
- 尚未进行 ACS/BUA 验证、离线 FP16 特征采集、全面模块/缓存 dtype 审计或 FP16 动作误差研究；这些仍按 P2-P5 顺序推进。

方案 B 的准确恢复命令见 `patches/diffusion_policy/README.md`。本机子模块保留在安全提交 `386b7de`，所以主仓库 status 仍可能标记子模块提交不同；这不表示漏交付，远程固定 gitlink 加完整 patch 才是恢复依据。用户原有 `lowdim_fp16.md` 保留未提交。

下一步是推送主仓库功能分支，并在全新远程克隆中验证子模块初始化和 patch 恢复；完成后停止，等待确认进入 P2。
