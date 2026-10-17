# Transformer FP32 / FP16 AMP 前向讲解与 P2 手动操作

> 最新流程：用户已授权直接完成实验并改为主仓库直接跟踪源码。
> 当前统一命令和结果见 [LOWDIM_FP16_RUNBOOK.md](LOWDIM_FP16_RUNBOOK.md)。
> 本文保留前向教学与早期操作记录；其中子模块补丁步骤、逐阶段等待确认和暂不提交的说明属于历史流程。

日期：2026-10-17。仓库根目录：`/home/zenia/Projects/BAC/BAC`。

本文是学习说明和操作指南，不是完整 P2 验收报告。独立脚本已由用户创建；随后经用户授权，助手修复了 hook 的关键字参数处理，并给 eval 的 THOP 调用补上显式 autocast。运行证据见文末，本轮不提交或推送 Git。

## 1. 当前阶段与目标

工作计划中的下一阶段是 **P2：dtype 和模型结构验证**。先完成第一项练习：同一个 Transformer、同一份输入，分别执行一次 FP32 和 FP16 AMP forward，观察模块输入、输出的 dtype 和 shape。

本练习只调用 `policy.model(...)`，不执行完整 `predict_action()`，不安装 BAC 缓存。之后再补齐 Scheduler、完整采样和缓存审计，才能完成 P2。固定噪声下的完整动作误差、正式延迟比较属于 P3。

P1 的历史验收见 [P1_precision_switch.md](P1_precision_switch.md)。其中 `.git/p1-validation/validation.log` 是以前的记录，不代表这次已经重新运行通过。

## 2. 从命令行到 Transformer 的调用关系

| 文件 | 函数或位置 | 作用 |
| --- | --- | --- |
| `BACInfer/scripts/eval_fast_diffusion_policy.py` | Click `--precision`、`main()` | 选择精度，并给原始策略和缓存策略分别赋值 |
| `diffusion_policy/diffusion_policy/policy/diffusion_transformer_lowdim_policy.py` | `__init__()` | 定义默认 `inference_precision='fp32'` |
| 同上 | `predict_action()` | 归一化观测，准备条件和轨迹，调用采样 |
| 同上 | `conditional_sample()` | 在每个去噪步，只对模型 forward 启用 AMP |
| `diffusion_policy/diffusion_policy/model/diffusion/transformer_for_diffusion.py` | `forward()` | 时间/输入/条件嵌入，编码器，解码器，输出头 |
| `BACInfer/core/diffusion_cache_wrapper.py` | `_add_cache_to_block()` | 缓存模式下替换 DecoderLayer 的执行过程，保存或复用分支输出 |

```text
eval --precision fp16_amp
  -> main()
     -> original_policy.inference_precision = 'fp16_amp'
     -> fast_policy.inference_precision = 'fp16_amp'
     -> predict_action(obs)
        -> conditional_sample(...)
           -> 每个去噪时间步：
              with autocast(...):
                  model(trajectory, t, cond)
                    -> TransformerForDiffusion.forward(...)
              model_output.float()
              scheduler.step(...)  [autocast 范围之外]
```

这些函数是嵌套调用。`model(...)` 通过 PyTorch 的模块调用机制执行 `forward()`，也会触发注册的 hook。

`original_policy` 表示不使用缓存的策略，不代表固定 FP32。eval 中选择 `fp16_amp` 时，普通策略和缓存策略都会收到同一个精度设置；eval 中的 original/cached 时间对比不是 FP32/FP16 时间对比。

## 3. FP16 从哪里开始生效

当前 `conditional_sample()` 的关键逻辑如下，已有代码不需要重复粘贴：

```python
use_fp16 = (self.inference_precision == 'fp16_amp')

with torch.autocast(
    device_type='cuda',
    dtype=torch.float16,
    enabled=use_fp16,
):
    model_output = model(trajectory, t, cond)

if use_fp16:
    model_output = model_output.float()
```

- `==` 进行比较，返回布尔值，决定是否启用 AMP。
- `with` 建立一个临时执行范围；范围内调用的子模块也继承 autocast 状态。
- autocast 按运算选择精度，不是在进入代码块时把所有变量转成 half。
- `model_output.float()` 返回 FP32 Tensor，需要接收返回值；它不会恢复 FP16 运算已经损失的数值精度。
- 不使用 `model.half()`：模型保存的参数仍保持 FP32。
- `model.eval()` 影响 Dropout 等模块；`torch.inference_mode()` 关闭推理不需要的求导记录；二者都不负责开启 FP16。

**重要边界：** 精度属性只有在 `conditional_sample()` 读取它时才发挥作用。直接调用 `policy.model(...)` 会绕过这个开关，因此本练习会显式建立 autocast。eval 中的 THOP 调用也已增加独立的 autocast 上下文，使用同一个 `precision` 参数；不再依赖 Policy 的采样上下文。补上 AMP 不意味着运算次数或 THOP FLOPs 减半。

## 4. 一次 Transformer 前向的数据流

输入含义：

- `sample`：当前去噪步的轨迹，形状 `(B, T, input_dim)`。
- `timestep`：当前去噪步编号，通常是整数 Tensor。
- `cond`：历史观测条件，形状 `(B, To, cond_dim)`；部分配置下为 `None`。
- `B` 为批大小，`T` 为动作序列长度，`To` 为观测步数，`D` 为嵌入维度。

不要将源码默认参数或注释示例中的维度当作 checkpoint 的实际配置。下面的练习会打印真实配置和输入形状。

### 4.1 条件编码器与解码器分支

```text
timestep -> time_emb --------------------+
                                        |
cond -> cond_obs_emb -> 拼接时间条件 -----+
                                        |
                                 加条件位置编码
                                        |
                                     encoder
                                        |
                                      memory
                                        |
                                        v
sample -> input_emb -> 加位置编码 ------> decoder
                                          |
                                         ln_f
                                          |
                                         head
                                          |
                                      模型输出
```

动作支路的 `input_emb` 是 Linear。AMP 下它可以输出 FP16；随后与 FP32 的 `pos_emb` 相加，结果回到 FP32。

条件支路中，时间编码通常为 FP32，观测的 Linear 嵌入在 AMP 下输出 FP16；拼接并加入 FP32 条件位置编码之后，条件编码器输入为 FP32。

`n_cond_layers == 0` 时，条件编码器是 `Linear -> Mish -> Linear`；大于 0 时是 TransformerEncoder。两种结构的 `memory` 输出类型可能不同，应以实际 hook 记录为准。

若实际模型的 `encoder_only=True`，则走源码中的另一分支：时间 token 拼入主序列，经 TransformerEncoder 后移除时间 token，再进入输出头；没有独立 decoder 或交叉注意力。下面脚本会打印该标志，不能套用不存在的 decoder 结构。

### 4.2 DecoderLayer 中的 SA、CA、FFN

模型创建 DecoderLayer 时设置 `norm_first=True`，实际结构可以概括为：

```python
x = x + self._sa_block(self.norm1(x), ...)
x = x + self._mha_block(self.norm2(x), memory, ...)
x = x + self._ff_block(self.norm3(x))
```

- SA：Self-Attention，动作序列内部交换信息。
- CA：Cross-Attention，动作序列读取条件；源码中的模块名是 `multihead_attn`，方法名是 `_mha_block`。
- FFN：逐位置非线性变换，核心为 `linear1 -> GELU -> Dropout -> linear2 -> Dropout`。
- `x + 分支输出` 为残差连接：保留主干信息，同时加入当前模块的修正量。

AMP 下的典型 FFN 数据流：

```text
主干 x [FP32] ------------------------------+
    |                                       |
  LayerNorm [FP32]                           |
    |                                       |
  Linear1 [FP16 输出]                        |
    |                                       |
  GELU / Dropout                             |
    |                                       |
  Linear2 [FP16 输出]                        |
    |                                       |
  Dropout --------------------------------> 相加
                                              |
                                          新 x [FP32]
```

LayerNorm 适合保留 FP32，Linear 等矩阵运算可使用 FP16；FP16 与 FP32 Tensor 相加会产生 FP32 结果。这些规则见 [PyTorch 2.11 AMP 文档](https://docs.pytorch.org/docs/2.11/amp.html#cuda-op-specific-behavior)。注意力可能走融合内核，不能仅凭模块输入、输出 dtype 判断内部 Softmax 或累加器精度。

### 4.3 两种精度的预期对比

以下以当前 FP32 参数和 FP32 输入为前提，是源码与 autocast 规则给出的预期，尚不是本次实测结果：

| 位置 | FP32 | FP16 AMP |
| --- | --- | --- |
| 模型原始参数 | FP32 | FP32 |
| forward 入口 sample、cond | FP32 | FP32 |
| input_emb、cond_obs_emb 的 Linear 输出 | FP32 | FP16 |
| 与 FP32 位置编码相加后 | FP32 | FP32 |
| 标准 LayerNorm 输出 | FP32 | FP32 |
| 注意力投影、FFN Linear | FP32 | 按 autocast 规则使用 FP16 |
| Decoder 残差主干 | FP32 | FP32 |
| memory | FP32 | 取决于实际 encoder 结构，待测 |
| 最终 head 输出 | FP32 | FP16 |
| Policy 中 `.float()` 后 | FP32 | FP32 |
| Scheduler 输入与轨迹 | FP32 | FP32，待完整采样审计复查 |

这里只说明 Tensor 的 dtype，不代表已测量 GPU 内部每个算子的累加精度、速度或显存。输出类型变成 FP16 也不能证明所有内部运算均为 FP16。

## 5. P2 第一项练习：独立审计脚本

### 5.1 创建位置与原因

当前独立脚本已经存在，其路径为：

```text
BACInfer/analysis/check_lowdim_fp16.py
```

下面保留完整参考代码，便于理解和恢复；当前无需重复覆盖已有文件。不要把审计逻辑插入 `TransformerForDiffusion.forward()`。eval 本轮只补齐 THOP 的 autocast，不改变 Policy 或缓存算法。

独立脚本只在运行时注册观察 hook，结束时移除，不污染正常推理输出。此版本只审计 Original 单次 forward，后续仍在 P2 内扩展完整采样和缓存观测。

### 5.2 完整参考代码（已包含关键字参数 hook 修复）

```python
"""P2 first exercise: observe one original Transformer forward per precision."""
import argparse
import json
from pathlib import Path
import sys

import dill
import hydra
from omegaconf import OmegaConf
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'diffusion_policy'))
sys.path.insert(0, str(ROOT))

from diffusion_policy.policy.diffusion_transformer_lowdim_policy import (
    DiffusionTransformerLowdimPolicy,
)


def describe(tensor):
    if tensor is None:
        return None
    finite = bool(torch.isfinite(tensor).all().item())
    if not finite:
        raise RuntimeError('NaN/Inf detected in an observed data tensor')
    return {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'finite': finite,
    }


def observe_forward(model, sample, timestep, cond, use_amp):
    rows = []
    handles = []

    def make_hook(name):
        def hook(module, inputs, kwargs, output):
            input_tensor = inputs[0] if inputs else None
            if input_tensor is None:
                for value in kwargs.values():
                    if isinstance(value, torch.Tensor):
                        input_tensor = value
                        break
            # MultiheadAttention returns (attention_output, weights_or_None).
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f'Unexpected output from {name}')
            rows.append({
                'module': name,
                'type': type(module).__name__,
                'input': describe(input_tensor),
                'output': describe(tensor),
                'cuda_autocast': torch.is_autocast_enabled('cuda'),
            })
            # No return: the hook observes but does not replace the output.
        return hook

    observed_types = (
        nn.Linear, nn.LayerNorm, nn.MultiheadAttention,
        nn.TransformerDecoderLayer, nn.TransformerEncoderLayer,
    )
    observed_names = {'time_emb', 'encoder', 'decoder'}
    try:
        for name, module in model.named_modules():
            if isinstance(module, observed_types) or name in observed_names:
                handles.append(
                    module.register_forward_hook(make_hook(name), with_kwargs=True)
                )
        with torch.inference_mode():
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                output = model(sample, timestep, cond)
            output_info = describe(output)
        expected = torch.float16 if use_amp else torch.float32
        if output.dtype != expected:
            raise RuntimeError(f'Expected model output {expected}, got {output.dtype}')
        return {'model_output': output_info, 'modules': rows}
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT / (
        'checkpoint/low_dim/block_pushing/diffusion_policy_transformer/'
        'train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This FP16 AMP exercise requires CUDA')
    if args.output.exists():
        raise FileExistsError(f'Keep the existing result; choose a new path: {args.output}')

    with args.checkpoint.open('rb') as file:
        payload = torch.load(file, pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model.to(device).eval()
    if not isinstance(policy, DiffusionTransformerLowdimPolicy):
        raise TypeError('Expected DiffusionTransformerLowdimPolicy')
    if getattr(policy, 'inference_precision', None) != 'fp32':
        raise RuntimeError('Verify that P1 is restored and defaults to fp32 first')
    model = policy.model
    parameter_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    if parameter_dtypes != ['torch.float32']:
        raise RuntimeError(f'Expected FP32 parameters, got {parameter_dtypes}')

    # Build the inputs once. Both precision runs reuse these exact tensors.
    with torch.inference_mode():
        obs = torch.zeros((1, policy.n_obs_steps, policy.obs_dim), device=device)
        nobs = policy.normalizer['obs'].normalize(obs)
        length = policy.n_action_steps if policy.pred_action_steps_only else policy.horizon
        features = policy.action_dim if policy.obs_as_cond else policy.action_dim + policy.obs_dim
        generator = torch.Generator(device=device).manual_seed(args.seed)
        sample = torch.randn((1, length, features), device=device,
                             dtype=torch.float32, generator=generator)
        cond = nobs[:, :policy.n_obs_steps] if policy.obs_as_cond else None
        if not policy.obs_as_cond:
            sample[:, :policy.n_obs_steps, policy.action_dim:] = nobs
        policy.noise_scheduler.set_timesteps(policy.num_inference_steps)
        timestep = policy.noise_scheduler.timesteps[0].to(device)

    report = {
        'scope': 'original_single_forward_only; full_P2_pending',
        'checkpoint': str(args.checkpoint.resolve()),
        'weights': 'workspace.model (same source as eval)',
        'python': sys.version.split()[0],
        'torch': str(torch.__version__),
        'gpu': torch.cuda.get_device_name(device),
        'seed': args.seed,
        'parameter_dtypes': parameter_dtypes,
        'model_config': OmegaConf.to_container(cfg.policy.model, resolve=True),
        'policy': {name: getattr(policy, name) for name in (
            'obs_dim', 'action_dim', 'horizon', 'n_obs_steps',
            'n_action_steps', 'num_inference_steps', 'obs_as_cond',
            'pred_action_steps_only')},
        'encoder_only': model.encoder_only,
        'inputs': {'obs': describe(obs), 'nobs': describe(nobs),
                   'sample': describe(sample), 'cond': describe(cond),
                   'timestep': describe(timestep)},
        'runs': {},
    }
    for precision in ('fp32', 'fp16_amp'):
        result = observe_forward(model, sample, timestep, cond,
                                 use_amp=(precision == 'fp16_amp'))
        report['runs'][precision] = result
        print(f'\n=== {precision} ===')
        for row in result['modules']:
            print(row['module'], row['input']['dtype'], '->',
                  row['output']['dtype'], row['output']['shape'])
        print('model_output:', result['model_output'])
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError('Model parameters unexpectedly changed dtype')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(f'\nJSON saved: {args.output.resolve()}')
    print('P2 FORWARD CHECK PASSED (full P2 still pending)')


if __name__ == '__main__':
    main()
```

### 5.3 代码中的语法与观察点

| 语法或变量 | 作用 |
| --- | --- |
| `Path(__file__).resolve().parents[2]` | 从脚本位置定位 BAC 根目录，避免依赖启动时的导入路径 |
| `workspace.model` | 与 eval 采用相同的权重来源，不切换到 EMA |
| `model.eval()` | 关闭 Dropout 的随机丢弃行为，使同一份输入更适合做对照 |
| `torch.Generator(...).manual_seed(11)` | 生成一份固定的初始轨迹；两次 forward 复用同一份 Tensor |
| `with torch.inference_mode()` | 推理时不记录梯度相关信息；不会启用 AMP |
| `named_modules()` | 遍历模块名称和对应模块对象 |
| `register_forward_hook(...)` | 模块执行后调用观察函数；观察函数不返回新输出 |
| `with_kwargs=True` | 将关键字参数也传给 hook；签名对应为 `(module, inputs, kwargs, output)` |
| `kwargs.values()` | 遍历关键字实参；位置参数为空时，从中找到 `tgt=x` 等输入 Tensor |
| `make_hook(name)` | 用闭包保存各个模块自己的名称，避免记录错位 |
| `finally` | 即使 forward 报错，也移除已经注册的 hook |
| `describe(...)` | 只保存 dtype、shape、device、finite 等摘要，不保存完整激活 |
| `open('x')` | 新建结果文件；若文件已经存在则拒绝覆盖 |

此练习只检查观测到的数据 Tensor；注意力 mask 中有意设置的 `-inf` 不属于要拒绝的激活 NaN/Inf，因此不对这些 mask 做全有限断言。

SA 对应日志中的 `self_attn`，CA 对应 `multihead_attn`。FFN 观察 `linear1`、`linear2`；本练习 `eval()` 下最终 Dropout 不改变结果，可据此理解 FFN 分支输出。后续缓存审计仍要直接观测真正写入缓存的分支 Tensor。

部分注意力投影在 PyTorch 内部通过 functional 运算执行，而不是调用某个 `nn.Linear` 模块的 `__call__()`，因此不能要求所有 `out_proj` 都产生 hook 日志。没有日志不等于该投影没有计算。

hook 和逐模块的 `isfinite().item()` 会增加开销，甚至影响部分融合路径选择。因此此脚本不是性能测试，禁止据此计算 FP16 加速比。

## 6. 运行命令与预期结果

### 6.1 先确认 P1 已恢复

```bash
cd /home/zenia/Projects/BAC/BAC
conda activate bac
python -B scripts/apply_diffusion_policy_patch.py --check
```

若显示 `Patch already present; 7 file checksums match`，说明已有源码与保存版本一致。

若显示 `Patch applies cleanly ... check only`，说明源码仍是原始基准。此时由你实际应用后再检查：

```bash
python -B scripts/apply_diffusion_policy_patch.py
python -B scripts/apply_diffusion_policy_patch.py --check
```

如果提示其他本地修改或校验不一致，先保留当前文件并分析差异，不执行 reset 或强行覆盖。

### 6.2 运行已有脚本

当前脚本已经存在并包含修复，可直接执行。以下输出目录每次生成一个新名称，保留过去的结果。

```bash
cd /home/zenia/Projects/BAC/BAC
conda activate bac
mkdir -p .git/p2-validation
p2_run_dir=$(mktemp -d .git/p2-validation/forward-XXXXXX)
python -B -m BACInfer.analysis.check_lowdim_fp16 --output "$p2_run_dir/forward.json"
echo $?
```

`echo $?` 必须紧接 Python 命令执行。预期退出码为 `0`，最后显示：

```text
P2 FORWARD CHECK PASSED (full P2 still pending)
```

每种精度只调用一次模型；与 P1 完整采样的 100 次模型调用不同。本轮实际运行记录另见文末，不把第一项练习通过称为完整 P2 通过。

检查终端和 JSON：

| 观察点 | FP32 预期 | FP16 AMP 预期 |
| --- | --- | --- |
| 参数、sample、cond（非 None 时） | FP32 | FP32 |
| input_emb 输出 | FP32 | FP16 |
| DecoderLayer 输入和输出（存在时） | FP32 | 通常 FP32，与残差主干对应 |
| norm1 / norm2 / norm3 输出（存在时） | FP32 | FP32 |
| self_attn / multihead_attn 输出（存在时） | FP32 | 预期 FP16，以实际记录为准 |
| linear1 / linear2 输出 | FP32 | FP16 |
| head / model_output | FP32 | FP16 |
| finite | true | true |

如果某项与预期不同，先看 checkpoint 结构和对应模块的实际路径，不为了符合表格而强制插入 `.half()`。

## 7. P2 后续项目与阶段验收

单次 forward 练习通过后，仍需在同一个独立脚本中补齐：

1. 在完整 `predict_action()` 中记录模型入口、每步模型原始输出、Scheduler 输入和输出；用前置 hook、输出 hook 及可恢复的 Scheduler 观察包装，确认 autocast 仅覆盖模型。
2. 分别检查 Original FP32、Original AMP、Threshold FP32、Threshold AMP。缓存测试用于审计机制，不代表已经完成 ACS/BUA 的 P4 复现。
3. 对每层 SA/CA/FFN 记录实际执行次数，观察 `block_cache` 的每个条目的 dtype、shape、有限性、写入次数；区分缓存写入与复用。
4. 确认观察 hook 和方法包装在结束后都已移除或恢复，完整采样无 dtype mismatch、NaN/Inf。
5. 将小型有效结果和对比表写入 `reports/P2_dtype_audit.md`，记录 checkpoint、环境、精度、步数和命令，再验收 P2。

不能将“单次 head 输出 FP16”当作全部 P2 验收；本练习没有执行 Scheduler step、缓存或完整采样。也不能用模块边界的 dtype 声称已证明融合内核每个内部算子的精度。

`collect_activations.py` 属于后续离线特征采集链路，暂不修改。仅给 eval 添加精度开关，不会自动使其他采集入口启用 AMP。

## 8. Git 保存与同步：由你操作

本轮经用户授权修改主仓库的独立审计脚本、eval 的 THOP 上下文和本文档。没有修改子模块源码或更新补丁，没有提交或推送。

本练习的独立脚本放在 BAC 主仓库中；不改子模块源码，因此不需要重新生成子模块 patch。原有 `lowdim_fp16.md` 和子模块本地修改均保留。

完整 P2 通过后，检查主仓库状态和远程：

```bash
git branch --show-current
git remote -v
git status --short
git diff --check
```

确认分支为 `feature/lowdim_fp16`，origin 指向你的 `git@github.com:Zenia-123/BAC.git`。届时仅将实际创建并验收的文件逐一加入：

```bash
git add BACInfer/analysis/check_lowdim_fp16.py \
        BACInfer/scripts/eval_fast_diffusion_policy.py \
        reports/P2_transformer_forward_guide.md \
        reports/P2_dtype_audit.md
git diff --cached
git diff --cached --check
git commit -m "test: verify AMP dtypes and scheduler stability"
git push origin feature/lowdim_fp16
```

上面的提交命令用于完整 P2 验收之后，不要在只有本文档或第一项练习时执行。不要 `git add .`，也不要把 `.git/` 中的本地 JSON 或 checkpoint 加入提交。

`.git/p2-validation/` 只存本机运行证据，不会随 clone 恢复；正式验收报告应保留可复现命令和小型结果。若后续确实修改子模块源码，必须另行更新完整 patch、manifest 并验证新目录恢复，不能只提交子模块指针。

## 9. 本轮你需要做什么

1. 阅读第 2–4 节，对照实际源码理解 autocast 的边界。
2. 阅读第 5 节的 hook 修复；已有脚本无需重新创建。
3. 查看文末本轮运行证据；需要自行复查时按第 6 节生成新的结果。
4. 根据实际结果继续完成 P2 后续审计，不提前进入 P3，不提前提交阶段完成记录。

## 10. 本轮实际修改与验证记录

### 10.1 修复内容

用户授权直接操作后，本轮修改三个主仓库文件：

1. `BACInfer/analysis/check_lowdim_fp16.py` 的 `observe_forward()`：hook 接口增加 `kwargs`，注册时传入 `with_kwargs=True`。位置参数为空时，从关键字参数寻找输入 Tensor，解决 `decoder(tgt=x, ...)` 的输入被记录为 `None`，进而在打印 `['dtype']` 时触发 TypeError 的问题。保留用户已有中文注释。
2. `BACInfer/scripts/eval_fast_diffusion_policy.py` 的 `main()`：只在 THOP `profile(...)` 外增加由 `precision` 控制的 CUDA autocast；不改变普通采样、缓存算法或 Scheduler。
3. 本文档：修正相同的 hook 示例，更新 THOP 说明并记录实测结果。

改前备份位于 `.git/p2-validation/fix-61jc8sst/before/`，并保存 `before_sha256.json`。未修改模型源码、checkpoint、用户笔记或 `.gitignore`；没有 Git 提交、推送。补丁检查仍显示 `7 file checksums match`，无需更新子模块 patch。

### 10.2 单次前向审计：通过

实际执行命令（已完成，结果文件存在；自行重跑请使用第 6 节的新目录命令）：

```bash
/home/zenia/cimflow/miniconda3/envs/bac/bin/python -B \
  -m BACInfer.analysis.check_lowdim_fp16 \
  --output .git/p2-validation/fix-61jc8sst/forward.json \
  > .git/p2-validation/fix-61jc8sst/forward.log 2>&1
```

退出码 `0`，末行为 `P2 FORWARD CHECK PASSED (full P2 still pending)`。日志和 JSON 都保存在上述本地目录，未纳入 Git。

真实配置：`obs_dim=16`、`action_dim=2`、`horizon=5`、`n_obs_steps=3`、`num_inference_steps=100`、`n_emb=256`、`n_layer=8`、`n_head=4`、`obs_as_cond=True`、`n_cond_layers=0`。实际走条件 MLP 加 TransformerDecoder 分支。

两次 forward 使用同一份输入：sample 为 `(1,5,2)`，cond 为 `(1,3,16)`，均为 FP32。时间步取 Scheduler 设置后的第一个时间步，但不执行 Scheduler step。每种精度记录 73 条模块结果；已检查所有记录的输入均非 None，输入、输出有限，autocast 状态与模式一致。

| 观测位置 | FP32 输入 -> 输出 | FP16 AMP 输入 -> 输出 |
| --- | --- | --- |
| input_emb | FP32 -> FP32 | FP32 -> FP16 |
| 条件 encoder / memory | FP32 -> FP32 | FP32 -> FP16 |
| decoder.layers.0.norm1 | FP32 -> FP32 | FP32 -> FP32 |
| decoder.layers.0.self_attn | FP32 -> FP32 | FP32 -> FP16 |
| decoder.layers.0.multihead_attn | FP32 -> FP32 | FP32 -> FP16 |
| decoder.layers.0.linear1 | FP32 -> FP32 | FP32 -> FP16 |
| decoder.layers.0.linear2 | FP32 -> FP32 | FP16 -> FP16 |
| 整个 decoder | FP32 -> FP32 | FP32 -> FP32 |
| head / 模型输出 | FP32 -> FP32 | FP32 -> FP16 |

模型输出形状均为 `(1,5,2)`；模型原始参数保持 FP32。上表展示代表性模块，全部模块记录以 JSON 为准。

### 10.3 THOP 路径：两种精度均通过

采用临时 CLI 测试：执行真实 checkpoint 加载、两份策略的 warmup 和实际 THOP `profile()`，通过临时模型 hook 观测输入、输出。测试包装器在 THOP 完成后主动停止，未继续 benchmark 或环境 rollout。没有在生产脚本中添加提前退出逻辑。

| precision | profile 内 CUDA autocast | 模型输入 | 模型输出 | 有限性 |
| --- | --- | --- | --- | --- |
| fp32 | False | FP32 | FP32 | 通过 |
| fp16_amp | True | FP32 | FP16 | 通过 |

测试进程退出码 `0`，记录位于 `.git/p2-validation/fix-61jc8sst/thop-check.log` 和 `thop-check.json`。本次使用 `threshold` 缓存模式；THOP 在两种精度下返回的计数相同，不能据此得出速度变化或完整采样 FLOPs。THOP 对注意力等模块的统计覆盖也未在本轮核实。

### 10.4 静态检查与下一步

Python 语法检查、文档参考脚本与实际脚本的 AST 一致性检查、文档 Bash 语法检查和 `git diff --check` 均通过。

**下一步仍在 P2：** 给独立审计脚本补充完整 `predict_action()` 的 Scheduler 精度边界观测，再增加 BAC 缓存 dtype、shape 和更新/复用次数审计。完成后才生成完整的 `P2_dtype_audit.md` 并进行阶段提交。当前无需再改 Transformer 或 Policy，也不进入 P3 的误差和性能结论。
