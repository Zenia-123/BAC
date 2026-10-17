# G0：BAC Lowdim Transformer FP16 Git 与源码审计

审计日期：2026-10-17（Asia/Shanghai）。范围仅 G0；P1 尚未执行。

## 1. 审计范围与计划来源

- 工作区目录 `/home/zenia/Projects/BAC` 是外层目录，不是 Git 仓库；真正仓库根目录为 `/home/zenia/Projects/BAC/BAC`。后续 Git 命令在真正仓库中执行。
- 两个目录中都没有找到用户所述的 `BAC_Lowdim_FP16_VSCode_逐步实施工作计划.md`。本次读取的是用户明确提供的附件路径：`/home/zenia/Downloads/BAC_Lowdim_FP16_VSCode_%E9%80%90%E6%AD%A5%E5%AE%9E%E6%96%BD%E5%B7%A5%E4%BD%9C%E8%AE%A1%E5%88%92.md`，并核对了仓库现有 `lowdim_fp16.md`。没有复制或覆盖两者。
- 本次只增加此报告和 Git 元数据目录中的备份、日志；没有修改模型/评估代码、Git 分支/远程/索引，没有安装环境、commit、push、reset 或应用 patch。

## 2. 主仓库真实状态

| 项目 | 核查结果 |
| --- | --- |
| 当前分支 | `feature/lowdim_fp16`，并非 `feauture/lowdim_fp16` |
| HEAD | `7eb921dff7763c2770f175fefa2a1d0cac64718a`（`更新笔记`） |
| 跟踪分支 | `origin/feature/lowdim_fp16` |
| origin fetch/push | `git@github.com:Zenia-123/BAC.git` |
| upstream fetch/push | `https://ghfast.top/https://github.com/ky-ji/BAC.git` |
| 原有主仓库修改 | `lowdim_fp16.md`：新增 150 行；`diffusion_policy`：子模块内部有修改 |
| 暂存区 | 主仓库和子模块均无暂存修改 |
| 原有未跟踪文件 | 两个仓库的 `status --porcelain=v1 --untracked-files=all` 均未列出；忽略文件不在此结论中 |

只读执行了 `git ls-remote --symref origin HEAD refs/heads/feature/lowdim_fp16 refs/heads/main`（SSH BatchMode、严格校验已有主机密钥，退出码 0）：

```text
ref: refs/heads/main HEAD
5ef7f413d4f80abd8025d97b85afff1cb83131df HEAD
7eb921dff7763c2770f175fefa2a1d0cac64718a refs/heads/feature/lowdim_fp16
5ef7f413d4f80abd8025d97b85afff1cb83131df refs/heads/main
```

因此远程存在且非空，远程功能分支与本地 HEAD 一致；远程 main 与本地 main（`82029a6`）不同。没有 fetch、合并或改写历史。SSH 可读取并不能单独证明仓库的 Private 可见性或具备写权限；本次未调用 GitHub API 检查可见性，也未测试写入。

## 3. diffusion_policy 子模块与已有成果

三项证据一致：`.gitmodules` 声明该路径；主仓库索引模式为 `160000`（Git 用于子模块的 gitlink）；子目录拥有独立 Git 状态。

```text
gitlink / 子模块 HEAD：5ba07ac6661db573af695b419a7947ecb704690f
子模块状态：HEAD detached at 5ba07ac
.gitmodules URL：https://github.com/real-stanford/diffusion_policy.git
子模块实际 origin：https://ghfast.top/https://github.com/real-stanford/diffusion_policy.git
```

`git submodule status` 的前导空格只表示检出的提交与主仓库记录一致，不能据此判断内部工作区干净。主仓库短状态的 `m` 表示内部存在未提交修改。

子模块已有 **7 个修改文件（100 行增加、74 行删除）**：

| 子模块内部路径 | diff 中观察到的内容 |
| --- | --- |
| `diffusion_policy/env_runner/blockpush_lowdim_runner.py` | `AsyncVectorEnv(..., shared_memory=False)` |
| `diffusion_policy/gym_util/async_vector_env.py` | 覆盖 reset、调整 concatenate 参数顺序 |
| `diffusion_policy/gym_util/sync_vector_env.py` | 覆盖 reset、调整 concatenate 参数顺序 |
| `diffusion_policy/model/diffusion/transformer_for_diffusion.py` | 主要是中文讲解注释和空白改动；本次不把它们当作已验证的实验配置 |
| `diffusion_policy/policy/diffusion_transformer_lowdim_policy.py` | autocast、输出 `.float()`、讲解注释 |
| `diffusion_policy/workspace/train_diffusion_transformer_lowdim_workspace.py` | 构造器新增可选 output_dir 并传给父类 |
| `setup.py` | `find_packages` 改为 `find_namespace_packages` |

这些均属于现有现场，不能只保留 autocast 文件。对子模块源码的提交，主仓库只记录提交 ID，不会把其内部未提交内容自动上传。即使子模块先在本地 commit，若新提交没有可访问的远程或可重建 patch，新机器仍可能无法恢复。

### 本次已完成的本地备份

目录：`/home/zenia/Projects/BAC/BAC/.git/g0-backups/20261017-154542/`。

- `parent.patch`：主仓库已有修改，排除子模块目录。
- `diffusion_policy.patch`：子模块全部 7 个文件的修改。
- 两份 patch 均采用 `git diff --binary --full-index HEAD`；保存两仓库原始 status、remotes。
- `metadata.json`：两个基准 HEAD、8 个原有修改文件的 SHA-256、patch 反向校验结果。
- 分别执行 `git apply --reverse --check <patch>`，均退出 0；这里只检查 patch 与现场对应，没有实际应用或撤销修改。
- `environment_and_nameerror.log`、`cli_help.log`：本轮运行证据。

备份位于 `.git`，不会被普通主仓库提交收录；这是本地保护，不等于远程可恢复交付。恢复时应在对应基准提交的新工作区中先 `git apply --check`，确认后再应用；不要对当前已经含有修改的现场重复应用。

### 后续需要选择的远程恢复方案（本轮未执行）

| 方案 | 必要动作 | 取舍 |
| --- | --- | --- |
| A：独立 Private diffusion_policy 派生仓库 | 为子模块创建分支、保存全部既有修改；推送子模块提交到可访问的私有远程；主仓库更新 `.gitmodules` URL 与 gitlink；新目录验证递归克隆 | 工作计划推荐的长期方案；还需确认用户要使用的子模块仓库 URL、访问权限与原项目许可 |
| B：主仓库维护完整 patch | 保留公开基准 gitlink，提交覆盖全部现有子模块修改的 patch、固定基准 SHA、应用/校验脚本与 README；新目录验证 clone + patch 可以重建 | 仓库较少；此时主仓库 gitlink 本身仍指向未修改的公开基准，必须明确应用 patch 的步骤，后续 patch 也要维护 |

以上仅是方案；没有创建远程仓库、修改 `.gitmodules`、切换子模块分支或开始提交。用户本轮明确要求不提交/推送，优先于文档阶段提交示例。进入后续提交/推送前必须选定并验证恢复策略。

## 4. use_fp16 与当前可运行性

源码位置：`diffusion_policy/diffusion_policy/policy/diffusion_transformer_lowdim_policy.py:77-87`。

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16,
    enabled=use_fp16
):
    model_output = model(trajectory, t, cond)
model_output = model_output.float()
```

审计结论：`use_fp16` 只被读取，没有在方法参数、局部变量或模块全局中定义；构造器和继承链也没有精度属性。`self.use_fp16` 和裸变量 `use_fp16` 是不同表达式，仅增加属性而不改读取方式仍不能修复问题。

实测结果：

```text
Policy compile: PASS
Imported source: /home/zenia/Projects/BAC/BAC/diffusion_policy/diffusion_policy/policy/diffusion_transformer_lowdim_policy.py
Module defines use_fp16: False
Expected runtime failure: NameError: name 'use_fp16' is not defined
```

复现调用了真实类的 `conditional_sample`，仅使用 CPU 小张量和最简 scheduler/self 替身，让真实方法执行到首个去噪步。替身不修改生产类，不执行真实 Transformer，也不加载检查点。异常被审计程序捕获，所以审计命令退出 0，含义是成功确认现有故障，并非模型推理通过。

结论应区分：文件语法正确、Policy 可以导入、CLI help 可以运行；**现有 Lowdim 推理在进入首个去噪步时会失败**。即使在 eval main 中定义同名局部变量，另一个模块中的方法也不会自动读取它。

## 5. 环境与 Checkpoint

| 项目 | 实测值 |
| --- | --- |
| 工具 shell 初始 Conda 环境 | base |
| 已有 bac 环境 | `/home/zenia/cimflow/miniconda3/envs/bac` |
| 本轮验证解释器 | `/home/zenia/cimflow/miniconda3/envs/bac/bin/python` |
| Python | 3.10.0 |
| PyTorch | 2.11.0+cu128 |
| diffusers / hydra-core / click | 0.11.1 / 1.3.2 / 8.5.0 |
| CUDA build / available | 12.8 / True |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Checkpoint | `checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt` 存在，`ls -lh` 约 137M |

未重装环境。Checkpoint 被现有 `.gitignore` 排除，本轮未加载权重、未验证配置或完整采样；P1/P2 应读取实际配置，不能照抄 Transformer 注释中的示例维度。

`.gitignore` 同时忽略 `*.json`、`*.pkl` 等，后续希望版本化的小型结果 JSON 需要有针对性的例外，不能因其未出现在 status 就认为不存在。

## 6. 完整调用关系：1 个入口、2 条并列推理路径、1 个共同采样循环

```text
eval_fast_diffusion_policy.py 的 Click CLI
  main(checkpoint, output_dir, device, ...)
    torch.load -> cfg -> hydra.utils.get_class(cfg._target_)
    -> workspace = cls(cfg)
       -> hydra.utils.instantiate(cfg.policy) 构建 Policy 与 Transformer
    -> workspace.load_payload(payload) 恢复 state_dict 等
    -> original_policy = workspace.model -> .to(device) -> .eval()
    -> fast_policy = deepcopy(original_policy)
       -> FastDiffusionPolicy.apply_cache(fast_policy, ...)

    并列路径 A：original_policy.predict_action(obs_dict)
    并列路径 B：fast_policy.predict_action(obs_dict)
               -> reset_cache() -> 原 Policy.predict_action(obs_dict)

    两条路径均进入：
    predict_action
      obs -> normalizer['obs'].normalize -> nobs
      -> 构建 cond_data / cond_mask / cond
      -> conditional_sample(cond_data, cond_mask, cond=cond, **self.kwargs)
         -> torch.randn(dtype=condition_data.dtype) 初始化 trajectory
         -> scheduler.set_timesteps(num_inference_steps)
         -> for t in scheduler.timesteps:
              写回 condition_data 中受 mask 约束的值
              -> autocast 内 model(trajectory, t, cond)
                 -> nn.Module.__call__ -> TransformerForDiffusion.forward
                    时间/输入/条件 embedding -> encoder -> decoder -> ln_f -> head
              -> model_output.float()
              -> scheduler.step(...).prev_sample -> 新 trajectory
         -> 最后重新施加条件 -> 返回 nsample
      -> 取动作维度 -> unnormalize -> 截取 n_action_steps -> action 字典
```

original 和 fast 是两份独立 Policy，属于并列对照，不是 original 的输出再输入 fast。二者共享同一种嵌套调用结构。

`apply_cache` 是提前安装行为的步骤，不是每次推理重新装配：它用 `types.MethodType` 绑定替代方法。cached model.forward 先递增 `current_step`、更新缓存标志，再调用原 Transformer.forward；DecoderLayer 的 forward 则计算或复用 SA/CA/FFN 残差分支。外层 autocast 仍涵盖缓存版本在该次 forward 中实际发生的运算；直接返回的旧缓存张量保留原 dtype。`detach()` 只断开梯度关系，不转换 dtype。

源码导航：eval CLI/main 行 24/36；恢复/复制/缓存安装行 40-85；warmup 行 115-117；计时调用行 203/223；Policy 构造器行 14、采样行 55、predict_action 行 102；Transformer.forward 行 271；cache apply_cache 行 17、更新标志行 323、Decoder 替代 forward 行 360。

### 精度控制当前在哪里断开

```text
当前：CLI 无 precision 参数 -> Policy 无精度属性 -> enabled=use_fp16 未定义

P1 设计（尚未实现）：
--precision fp16_amp
  -> main 的 precision 字符串
  -> original_policy.inference_precision 和 fast_policy.inference_precision
  -> conditional_sample 中派生局部 bool use_fp16
  -> autocast(enabled=use_fp16)
```

精度配置属于 Python 状态，不需要随 cond 张量逐层传进 Transformer.forward 或 scheduler.step。

### 张量数据流（待运行验证 dtype）

设 B 为 batch、To 为观测步数、T 为 horizon、Do/Da 为观测/动作维度：

| 阶段 | shape / 类型来源 |
| --- | --- |
| eval 的零观测 | `(B, To, Do)`，显式 FP32；当前 B=1 |
| nobs | 归一化后的观测，实际 dtype 后续验证 |
| obs_as_cond=True | `cond=nobs[:, :To]`；cond_data 为 `(B,T,Da)`，dtype=self.dtype；pred_action_steps_only 时 T 替换为 n_action_steps |
| obs_as_cond=False | cond=None；cond_data 为 `(B,T,Da+Do)`，历史观测写入后 Do 维；mask 为 bool |
| trajectory | 与 condition_data 同 shape/dtype/device 的随机噪声 |
| Transformer 输出 | 同采样序列长度、输出维度由模型配置决定；AMP 中具体算子 dtype 不能猜测 |
| scheduler 输入/输出 | P1 目标：model_output 和 trajectory 均为 FP32；当前只有 model_output.float，仍须确认 condition_data/参数原本是 FP32 |
| action | 切片并反归一化后返回；shape 根据 pred_action_steps_only 与 n_action_steps 决定 |

`ModuleAttrMixin.dtype/device` 从首个参数读取；它们不是“当前 autocast 运算精度”的报告器。AMP 下参数仍可保持 FP32。

## 7. P1 最小修改方案（仅设计，不落地）

只编辑工作计划指定的两个文件。

| 文件/函数 | 新增或修改 | 原因 |
| --- | --- | --- |
| Policy `__init__` | 在 `**kwargs` 前显式增加可选 `inference_precision='fp32'`，保存到 self 并校验枚举 | 老 checkpoint 配置不含此项时仍有默认值；避免让精度参数误入 scheduler kwargs |
| Policy `conditional_sample` | 从实例属性计算局部 `use_fp16`；检查所选精度与 trajectory 实际设备；AMP 只包围 model 调用；仅 AMP 时将输出 `.float()` | 消除 NameError；CUDA 选择错误时清晰失败；尽量保留 FP32 原路径 |
| eval Click 装饰器 | 增加 `--precision`，`click.Choice(['fp32','fp16_amp'])`，默认 fp32 | 命令行可明确切换并限制输入 |
| eval `main` | 增加 precision 形参；load_payload 和 deepcopy 后、apply_cache/warmup 前为两份 Policy 显式设置相同属性；校验精度与设备 | 避免只配一份模型或被恢复/复制次序覆盖 |
| eval 结果保存 | 日志、benchmark_results 和 eval_results 增加 precision 字段 | 防止结果混淆；这仍在同一文件内 |

建议最小代码形态（示意，不是当前实现）：

```python
# __init__ 的显式可选参数，放在 **kwargs 前
inference_precision='fp32'

# conditional_sample 中，在循环外派生一次
use_fp16 = (self.inference_precision == 'fp16_amp')
if use_fp16 and trajectory.device.type != 'cuda':
    raise RuntimeError('fp16_amp requires CUDA')

# 循环中
with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_fp16):
    model_output = model(trajectory, t, cond)
if use_fp16:
    model_output = model_output.float()
# scheduler.step 在此 with 外，输入 trajectory 仍应为 FP32
```

P1 不必给 predict_action、Transformer.forward、cache wrapper 添加 precision 形参：实例属性已能跨方法共享；继承类与缓存替代方法沿用原 API。保留已有 generator 参数；固定噪声机制留到 P3。

P1 只读核对 Transformer、缓存 wrapper、BaseLowdimPolicy、ModuleAttrMixin 和其他 conditional_sample 定义。当前搜索结果显示其他 Policy 各自定义同名采样方法，不能通过修改它们来“补齐”本 Lowdim 开关。

训练 `compute_loss` 单独调用模型，新增开关仅在 conditional_sample 使用，不能把推理精度开关扩展到训练。离线 `collect_activations`/`run_policy` 是独立入口；eval CLI 的 precision 局部参数不会自动传播到它们，P4 再处理。

P1 验收：现有 CUDA 设备上，用实际 checkpoint 检查 FP32/FP16 AMP 单次模型 forward 与完整 predict_action；确认无 NameError/dtype mismatch/NaN/Inf、Scheduler 和 trajectory 为 FP32、FP32 原路径回归、CLI help 出现 precision。使用 inference_mode 运行推理验证。不要 model.half、改 checkpoint 或重新训练。通过后再按已确认的子模块恢复策略提交与推送。

## 8. Python / PyTorch 语法讲解

1. `self` 是当前对象，可以理解为“这台 Policy 自己”。`self.inference_precision` 是这台 Policy 保存的配置；`use_fp16` 是函数查找的名字，Python 不会自动在前面补 self。未找到名字会抛 NameError，编译通过不代表运行时名字存在。[Python 名称解析](https://docs.python.org/3.10/reference/executionmodel.html#resolution-of-names)
2. `enabled=use_fp16` 左侧是传给 autocast 的关键字参数名，右侧必须先求值为已有布尔值。`**kwargs` 则把字典展开成关键字参数。此项目将 Policy 的额外 kwargs 传给 scheduler.step，所以 precision 应明确声明或保存在属性中，不能不加区分地塞入 kwargs。
3. `with torch.autocast(...)` 是限定缩进范围的上下文管理器，可以理解为“这段计算临时采用自动选精度规则”。在该作用域内调用的子函数也受影响；退出后恢复上下文，但已产生的张量不会自动转回 FP32。AMP 按算子选择 dtype，并不把模型永久改成 half。[PyTorch 2.11 AMP](https://docs.pytorch.org/docs/2.11/amp.html)
4. `.float()` 返回 FP32 张量，需要接收返回值；它不会改变 shape，也不会将整个模型或其他张量一起转换。只把 model_output.float 并不能证明 trajectory 也保持 FP32，后者还由创建噪声时的 dtype 决定。
5. `model(trajectory,t,cond)` 调用 nn.Module 的调用入口，最终进入 forward，同时保留 hook 等机制。`trajectory` 是当前带噪序列，t 是 scheduler 时间步，cond 是条件观测；不是三次独立模型调用。
6. `deepcopy(original_policy)` 创建独立对象。复制前的属性通常随复制保留；复制后的赋值不会自动同步另一份对象，所以 P1 在复制后分别设置两者更容易审计。
7. `.eval()` 设置 Dropout 等层的推理行为；`torch.no_grad()`/`torch.inference_mode()` 控制自动求导相关记录；`autocast` 控制数值精度。三者解决不同问题，不能相互替代。
8. `@click.option('--precision', type=click.Choice(...))` 是装饰器，可理解为给 main 安装参数解析规则；Click 解析后将值作为 main 的 precision 参数传入。装饰器与函数签名需要一致。[Click 选项](https://click.palletsprojects.com/en/stable/options/)

## 9. 验证命令、结果与当前风险

所有命令在真实仓库根目录执行；Python 均指定已有 bac 解释器，并使用 `-B`/PYTHONDONTWRITEBYTECODE 避免生成新字节码。

| 核验 | 命令或方式 | 退出码 / 结果 |
| --- | --- | --- |
| Git 状态 | rev-parse、branch -vv、status、remote -v、submodule status、diff | 0，详情见前文 |
| Git 外层定位 | 在 `/home/zenia/Projects/BAC` 执行 rev-parse | 128：不在仓库内；已纠正执行目录 |
| 远程读取 | SSH 只读 git ls-remote | 0，功能分支 HEAD 一致，远程非空 |
| 备份一致性 | 两份 patch 的 apply --reverse --check | 均 0，没有实际应用 |
| 环境/可运行性 | bac Python 的版本、导入、compile、真实 conditional_sample + 最简替身 | 0；确认现有 NameError；日志见备份目录 |
| CLI 冒烟 | `/home/zenia/cimflow/miniconda3/envs/bac/bin/python -B -m BACInfer.scripts.eval_fast_diffusion_policy --help` | 0；尚无 --precision；日志见备份目录 |
| diff 格式 | 主仓库 git diff --check | 0 |
| 子模块 diff 格式 | git -C diffusion_policy diff --check | 2：原有 17 处行尾空白，涉及 Transformer 与 Policy；本轮不清理 |

审计复现核心调用（未修改真实类）：

```python
from types import SimpleNamespace
import torch
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy

class Scheduler:
    timesteps = [0]
    def set_timesteps(self, steps):
        pass

stub = SimpleNamespace(model=lambda *args: None,
                       noise_scheduler=Scheduler(), num_inference_steps=1)
data = torch.zeros((1, 2, 1))
mask = torch.zeros_like(data, dtype=torch.bool)
try:
    DiffusionTransformerLowdimPolicy.conditional_sample(stub, data, mask)
except NameError as error:
    print(type(error).__name__, str(error))
```

待处理风险按阶段归属：

- P1：use_fp16 未定义、CLI 未接通；FP32 trajectory/Scheduler 精度尚无完整推理证据；选择 fp16_amp 应拒绝 CPU。
- 后续 Git：子模块七处原有修改尚无远程恢复策略；当前备份仅本机。17 处原有格式问题应在后续明确范围后处理，不能把子模块 diff --check 说成通过。
- P2：尚未实测各算子/缓存 dtype，AMP 不能直接等同于“所有缓存都为 FP16”。
- P3：零观测仅冒烟输入，当前每次 predict_action 都重新抽随机噪声；THOP 路径直接 profile(fast_policy.model)，绕过 conditional_sample，未来 precision 开关不会自动覆盖该路径。它也不能作为完整采样 FLOPs。
- P4：current_step 为模型调用顺序计数，optimal 分支额外加入索引 99；缺少个别模块调度与完全缺少调度要区分（当前完全缺失会报错）；BUA 计算/加载异常可能被捕获而跳过。这里只记录边界，未进行算法复现或修复。
- 后续可比性：eval 当前选择 workspace.model；run_policy 可能根据配置选择 ema_model，比较时应统一权重来源。eval 的 CPU 计时还含无条件 torch.cuda.synchronize，不能把存在 --device 参数理解为已支持 CPU 全流程。
- 环境评估异常会被捕获跳过；本轮没有 rollout、成功率、性能或数值正确性结论。

G0 完成后停止，等待用户确认进入 P1。本轮无 commit/push；回溯基准为主仓库 `7eb921d` 与子模块 `5ba07ac`，原有未提交成果由上述两份 patch 保护。
