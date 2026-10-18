# Lowdim decoder 严格 FP16：修改、语法与验证

这次直接修改 `/home/zenia/Projects/BAC/BAC-original-fp32`。运行入口仍是
`python -m BACInfer.scripts.eval_fast_diffusion_policy`，增加
`--precision decoder_fp16_strict`。默认 `--precision fp32`。

当前从 ACS20 开始使用；无缓存 FP32 与严格 FP16 的同条件三任务均通过，
ACS5 在两种精度下均失败。严格 FP16 ACS20（BUA 关闭）后续新样本为 12/12，
与严格 FP16 Original 相同。这支持已测样本中精度影响较小、缓存更新预算影响更明显。
下方早期 ACS5 数据作为历史记录保留；ACS20+BUA3 的任务结果需另行评估。

日常命令及提交说明见 [分支使用说明](STRICT_FP16_RELEASE.md)，
历史协议见 [同条件对照命令与修改](STEP1_PAIRED_COMPARISON.md)。

## 1. 这次精度范围是什么

| 部分 | 精度与行为 |
|---|---|
| 观测、normalizer、初始 trajectory | 原来的 FP32 |
| input embedding、位置相加、时间及观测条件 encoder | 原来的 FP32 |
| decoder 入口的 target、memory、加性 mask | 转成 FP16 |
| 全部 8 个 decoder layer 的参数、Linear、SA、CA、FFN | FP16 |
| LayerNorm 的求和、均值、偏差、平方、方差、仿射变换 | FP16 运算，每次运算舍入到 FP16 |
| QK、PV、Linear 的乘积与树形归约 | FP16 乘积、FP16 累加 |
| Attention 缩放、mask 相加、Softmax 求和及归一化 | FP16；最大值通过半精度位模式的整数排序求出 |
| 残差相加；在线分支输出与缓存 | FP16 |
| decoder 后面的 `ln_f` | 也转换成严格 FP16 |
| 最终 `head`、scheduler、返回动作 | 保留 FP32 |
| ACS 相似度、BUA 特征变化排名 | 离线转换为 FP32 统计；DP 使用原 NumPy 数值精度 |
| ACS/BUA 调度、索引、缓存开关 | 整数、布尔值，没有 FP16 转换的需要 |

“所有 decoder 内部浮点运算都按半精度实现”是这份代码的目标。
指针、索引、线程编号和排序用整数，仍会出现 32/64 位整数指令。
查表准备、decoder 入口/出口转换以及外部模型计算不属于这段严格浮点算术。
这里验证的是软件计算及执行指令，不能据此推断 GPU 电路内部的物理位宽。

## 2. 把调用关系串起来

```text
终端 --precision decoder_fp16_strict
  |
  v
eval_fast_diffusion_policy.main
  +-- 加载原 checkpoint 的 workspace.model；to(cuda)；eval()
  +-- set_policy_precision(policy, precision)
  |     +-- model.set_inference_precision(precision)
  |           +-- strict_fp16.configure(model, precision)
  |                 +-- 只替换 decoder 子模块及 ln_f 的运算实现
  |                 +-- 只把它们的参数转 FP16
  |                 +-- 在推理前准备 FP16 查表
  +-- deepcopy -> fast_policy
  +-- FastDiffusionPolicy.apply_cache
  |     +-- 读取已经离线生成的 ACS 调度
  |     +-- 从对应 activations 计算 BUA 排名、合并更新时间
  |     +-- 包装分支 forward，并接通严格 FP16 残差相加
  +-- 预热 -> 指令审计 -> 计时 -> 环境闭环 -> 保存结果

predict_action(FP32 obs)
  +-- 原来的归一化和 conditional_sample
        +-- 100 次循环：
              model(FP32 trajectory, timestep, FP32 condition)
                +-- FP32 embedding / encoder
                +-- target.half()、memory.half()
                +-- 8 层严格 FP16 decoder
                |     +-- SA / CA / FFN 更新或复用 FP16 缓存
                |     +-- FP16 残差相加
                +-- 严格 FP16 ln_f
                +-- x.float() -> 原来的 FP32 head
              原来的 FP32 scheduler.step
  +-- 原来的反归一化、动作截取 -> FP32 action
```

没有减少外层的 100 个去噪步骤。缓存省掉的是步骤内部的分支重算。

## 3. 主要修改及 Python 语法

### 3.1 Transformer：明确设置精度边界

文件：[transformer_for_diffusion.py](../diffusion_policy/diffusion_policy/model/diffusion/transformer_for_diffusion.py)。
新增 `set_inference_precision()`；在原 `forward()` 的 decoder 前后插入转换。

```python
strict = getattr(self, 'inference_precision', 'fp32') == 'decoder_fp16_strict'
if strict:
    x, memory = x.half(), memory.half()
```

- `self` 是当前 Transformer 实例。
- `getattr(实例, 属性名, 默认值)` 读取属性；旧实例没有此属性时按 FP32 执行。
- `==` 得到布尔值，`if strict:` 决定是否进入严格路径。
- `x, memory = ...` 是同时给两个变量赋值。
- Tensor 的 `.half()` 返回 FP16 Tensor；入口仅在这里做数值类型转换。

```python
x = self.ln_f(x)
if getattr(self, 'inference_precision', 'fp32') == 'decoder_fp16_strict':
    x = x.float()
x = self.head(x)
```

先以严格半精度执行 `ln_f`，再在 head 前恢复 FP32 接口。
`.float()` 只扩展存储精度，不会恢复已经丢失的低精度信息。
Policy 的 `conditional_sample()` 和 scheduler 算法没有改写。

精度转换发生在 checkpoint 加载之后。已经舍入成 FP16 的权重无法靠 `.float()`
还原原始权重，所以对同一个已转换实例请求切回 FP32 会报错；重新执行命令加载 checkpoint 即可。

### 3.2 新增严格算子及 decoder 实现

文件：[strict_fp16.py](../diffusion_policy/diffusion_policy/model/diffusion/strict_fp16.py)。

`class StrictLinear(nn.Linear)` 表示继承原 Linear 的参数组织方式，重新定义
`forward()` 使用严格矩阵乘法。`StrictLayerNorm`、`StrictAttention`、
`StrictDecoderLayer` 同样保留原模块接口与参数名称，替换具体计算。
`StrictDecoder` 只按顺序调用各层，使用模型已经提供的 mask，避免原容器自动生成并比较 causal mask 的额外 CUDA 路径。

```python
module.__class__ = StrictLayerNorm
model.decoder.half()
model.ln_f.half()
```

`module.__class__ = ...` 在原对象上切换到兼容的子类实现，现有 weight、bias、
模块名字和 hook 位置得到保留。它没有创建新的随机权重。
这种做法适用于这里经过检查的 PyTorch 模块与纯 Python 子类；不适合随意替换任意类。

`model.decoder.half()` 是 Module 的方法，递归修改该子树参数的 dtype；
它与前面 Tensor `.half()` 返回转换后 Tensor 的用法不同。
没有调用 `policy.half()`，因此 encoder、head、normalizer 和初始 trajectory 保持 FP32。

保留 `nn.TransformerDecoderLayer` 的继承关系尤其重要：原缓存 wrapper 用
`isinstance(layer, nn.TransformerDecoderLayer)` 查找可缓存层。替换后仍能找到全部八层。
保留 `dropout1/2/3` 的调用，离线 collector 仍能在原来的位置采集分支输出。
`eval()` 下 Dropout 不随机丢弃数据。

严格 Attention 实现 Q/K/V 投影、按 head 重排、QK、Softmax、PV 和输出投影，
不调用 PyTorch 的融合 Attention 路径。`reshape`、`transpose`、数据复制只调整组织方式。
本模式限定 CUDA、eval、无梯度、`norm_first=True`、GELU、当前 lowdim 模型所用的 Attention 配置；
不支持的配置会报错，不会静默退回其他精度。

### 3.3 为什么另写 Triton/PTX

在 Python 里把输出 Tensor 标成 FP16，无法验证内部累加精度。
这份实现直接指定半精度运算指令：

```python
@triton.jit
def _add(a, b):
    return tl.inline_asm_elementwise(
        'add.rn.f16 $0, $1, $2;', constraints='=h,h,h',
        args=[a, b], dtype=tl.float16, is_pure=True, pack=1)
```

- `@triton.jit` 是装饰器，表示这个函数由 Triton 编译成 GPU 代码。
- `tl.inline_asm_elementwise` 插入逐元素 PTX 指令。
- `add` 是加法，`rn` 是就近舍入、恰好居中时取偶数，`f16` 指定半精度。
- `$0`、`$1`、`$2` 分别是输出和两个输入；`=h,h,h` 指定半精度寄存器约束。
- `_mul()` 对应 `mul.rn.f16`。

矩阵乘法先执行半精度乘积，再通过 `tl.reduce(..., _add)` 做半精度树形求和。
没有使用 `tl.dot` 或 Tensor Core 矩阵指令，也没有使用默认 `tl.sum`。
树形求和指并行地先求局部和，再归并；每次相加都舍入到 FP16，
不是将整个点积放入 FP32 寄存器、最后一次性转回 FP16。
树形顺序由实际编译布局决定；本结果不承诺与任意硬件的串行求和逐位相同。

内核通过 `kernel[grid](...)` 启动：`grid` 指定并行程序块数量。
`enable_fp_fusion=False` 关闭编译层的自动乘加融合。
SASS 中可能用 `HFMA2(..., -RZ)` 实现半精度乘法，或用乘数 1 实现半精度加法；
这种半精度机器指令仍按对应 PTX 运算的语义执行，不能仅看见 “FMA” 就判成 FP32 累加。

### 3.4 LayerNorm、Softmax 与非线性查表

LayerNorm 的实现是：

```text
mean = half_sum(x) * half(1/D)
centered = half(x - mean)
variance = half_sum(half(centered * centered)) * half(1/D)
inv = half_rsqrt(half(variance + half(eps)))
y = half(half(half(centered * inv) * weight) + bias)
```

方差用除以 D 的定义，均值和方差采用两遍计算。
每个公式中的乘加都由上述半精度内核实现。严格半精度可能发生溢出或下溢；
程序不会通过隐藏的 FP32 累加来补救。

binary16 只有 65536 种位模式。因此在推理前，CPU 以 float64 计算
`exp`、`reciprocal`、`rsqrt`、`GELU` 对所有半精度输入的结果，随后保存成 FP16 表。
四张 GPU 表合计约 512 KiB，只在准备阶段创建、传输一次。
这部分离线准备允许较高精度；decoder 运行中读取的表项只有 FP16。

```python
index = x.to(tl.uint16, bitcast=True).to(tl.int32)
value = tl.load(table + index)
```

`bitcast=True` 表示重新解释同一组比特，不做浮点到整数的数值转换。
例如半精度输入的 16 位模式直接成为表索引，得到一个半精度输出。
运行时没有 float32 的 exp、除法或平方根调用。
表项来自 float64 近似计算后舍入，包含 Inf、NaN 和有符号零的处理；
这不是声称实现了对所有实数的符号级精确超越函数。

Softmax 先按半精度位模式的整数顺序找最大值，减最大值、查 exp、
半精度累加、查倒数、再半精度相乘。Attention 的缩放发生在 QK 归约之后。
加性 mask 的 `-inf` 使被屏蔽项的 exp 为 0。
这与原融合算子的运算顺序、舍入方式可以不同，所以必须实测动作误差。

### 3.5 缓存 wrapper：命中也要走严格残差

文件：[diffusion_cache_wrapper.py](../BACInfer/core/diffusion_cache_wrapper.py)。

```python
add = getattr(self, '_residual_add', torch.add)
x = add(x, sa_result)
# 缓存命中时：
x = add(x, cache['block_cache'][sa_block_key])
```

这里 `self` 是某一层 decoder。严格层定义：

```python
_residual_add = staticmethod(residual_add)
```

`staticmethod` 表示这是静态方法，经实例访问时不会额外自动传入 `self`。
因此 `add(x, contribution)` 恰好接收两个 Tensor。
普通层没有 `_residual_add` 属性，仍使用原来的 `torch.add`。
SA、CA、FFN 三种分支的缓存命中及重新计算路径全部作了同样处理。

缓存存入 `result.detach()`；`detach()` 只切断梯度关系，不改变 dtype。
新动作预测前仍自动清空缓存。
另将强制末步刷新从硬编码 `99` 改为 `num_steps-1`，100 步时仍是 99。

### 3.6 ACS 与 BUA 在何时执行

校准入口仍是 [get_optimal_cache_update_steps.py](../BACInfer/analysis/get_optimal_cache_update_steps.py)：

1. `run_policy(..., precision=...)` 先配置精度。
2. `collect_activations` 用无缓存 Original 实际采集 100 步特征，严格模式的分支特征保存为 FP16。
3. `activation_analysis` 在 `.float()` 后计算相似度，原 DP 选择更新时间。
4. 评估加载调度时，`bu_block_selection` 在 `.float()` 后计算特征变化、选分支。
5. wrapper 按原来的“同层及更深层 FFN 更新时间并集”规则补点。
6. 在线只查询整数更新时间、更新或复用 FP16 分支缓存，不再每步运行 DP 或 BUA 排名。

`.float()` 放在离线统计中，使减法、平方、余弦和排名稳定；它不改变磁盘上原始特征精度。

新增 [precision.py](../BACInfer/core/precision.py) 统一处理精度选择和校准标记。
严格校准路径增加 `decoder_fp16_strict/` 层，防止覆盖旧 FP32 数据。
`precision.json` 记录精度、算术实现版本、checkpoint SHA-256 和 `weights=model`。
重复校准检查这些标记；在线 wrapper 检查精度及严格算术版本。
旧的无标记 FP32 校准仍可用于 FP32；严格模式不允许无标记或精度不匹配的调度。

### 3.7 官方命令如何传递精度

[eval_fast_diffusion_policy.py](../BACInfer/scripts/eval_fast_diffusion_policy.py)、
[run_policy.py](../BACInfer/analysis/run_policy.py)、
[collect_activations.py](../BACInfer/analysis/collect_activations.py) 和调度入口均增加：

```python
@click.option('--precision', default='fp32',
              type=click.Choice(['fp32', 'decoder_fp16_strict']))
```

`@click.option` 让 Click 解析终端参数。`default` 是省略时的值；
`click.Choice` 限制合法字符串，拼错会直接提示。
`precision` 成为 `main()` 的参数，再逐层以 `precision=precision` 这个关键字参数传递。

官方计时比较的是**所选精度无缓存**与**所选精度有缓存**。
严格模式跳过不能覆盖自定义内核的 THOP 统计，JSON 的 `flops` 为 `null`。
环境动作复制到 CPU 后增加有限性检查，防止将 NaN/Inf 动作传入环境。
依赖清单补充已安装的 `triton==3.6.0`，默认 FP32 路径懒加载，不会导入严格内核模块。

## 4. 可以直接执行的命令

完整终端命令见 [README](../README.md#严格-fp16-decoder--acs20--bua)。
当前机器已经生成了调度，日常直接执行：

```bash
cd /home/zenia/Projects/BAC/BAC-original-fp32
conda activate bac
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B \
  -m BACInfer.scripts.eval_fast_diffusion_policy \
  --checkpoint /home/zenia/Projects/BAC/BAC-original-fp32/checkpoints/epoch=7550-test_mean_score=1.000.ckpt \
  -o results/strict_fp16_acs20_bua3_run01 \
  --device cuda:0 --cache_mode optimal \
  --optimal_steps_dir /home/zenia/Projects/BAC/BAC-original-fp32/assets/blockpush_strict/decoder_fp16_strict/original/optimal_steps/cosine \
  --metric cosine --num_caches 20 --num_bu_blocks 3 --skip_video \
  --precision decoder_fp16_strict \
  --n_train 0 --n_test 3 --n_envs 3 \
  --test_start_seed 4300000 --seed 11 --max_steps 350 --rollout_only
```

两个 `...THREADS=1` 是仅针对该进程的 CPU 数值库线程设置，不决定 GPU 或数值精度。
`-B` 禁止 Python 写字节码缓存；`\` 表示 Bash 命令在下一行继续，后面不要加空格。
去掉三个 `n_*` 覆盖参数会使用 checkpoint 的 6 个训练设置及 50 个测试设置。
需要 FP32 时选择 `--precision fp32`，并使用对应的 FP32 校准目录。

## 5. 早期 ACS5 验证记录与精度边界

环境：RTX 5070 Ti Laptop / sm_120，PyTorch `2.11.0+cu128`，Triton `3.6.0`。
Checkpoint SHA-256：`0673d771bc652a61f97068935f51f862a731e82197613d9d02e1b500b7803f29`。

| 验证 | 本轮结果 |
|---|---|
| 早期回归测试 | 当时 11 项通过；当前完整套件另含配对运行验证 |
| 舍入对照 | 构造点积：FP32 累加得到 `2**-20`，严格半精度乘积再求和得到 0 |
| Attention | 与小幅 FP32 参考对照通过，改变被 mask 的 memory 不影响输出 |
| 完整模型 | 参数名与 state_dict keys 保持；外部 FP32，decoder / ln_f 输出 FP16 |
| 缓存 | Threshold1 与相同严格模型输出逐位相同；跳过步骤不调用分支并复用原缓存对象 |
| 实际 checkpoint 固定输入 | 环境初始 seed 0、噪声 seeds 101/102/103、batch 1，三种策略各 100 步输出有限 |
| 调用次数 | 无缓存每次 2400 个分支；严格 ACS5+BUA3 为 152；scheduler 均调用 100 次 |
| BUA | 选 L7 FFN、L6 FFN、L7 CA，新增 8 个分支—时间点 |
| 指令检查 | 官方评估执行的 23 个编译版本通过 PTX/SASS 审计 |
| 官方闭环 | 3 个 seeds `4300000/4300001/4300002`，每任务最多 350 环境步，得分 `[0,0,0]` |
| 默认 FP32 入口回归 | 同一官方入口，已有 FP32 ACS5+BUA3、1 环境步的 smoke check 通过 |

固定输入下，严格无缓存相对 FP32 的 `action` MAE 分别约
`0.00299262 / 0.000953156 / 0.000129734`；严格 ACS5+BUA3 相对 FP32 为
`0.0143567 / 0.00220892 / 0.0209190`。这些是动作标量偏差，不是失败概率。
不能拿之前 AMP 报告的小误差替代本次严格半精度的实测误差。

官方脚本的零观测 batch 1、10 次计时：严格无缓存平均约 538.28 ms，
严格 ACS5+BUA3 约 135.49 ms，缓存内部加速约 3.97 倍。
该计时不能用于证明严格 FP16 比 FP32 更快，也不能与先前 batch 3 的报告直接横比。
完整任务中三个 seeds 得分为 0；同 seeds 的先前 FP32 ACS5+BUA3 也为 0，
这是早期记录；后续对照已确认 ACS20 在已测任务中表现更好，不能沿用 ACS5 的失败结论概括所有缓存配置。

本地证据：

- [固定输入 dtype、计数和成对误差](../results/strict_fp16_check/paired_audit/summary.json)
- [官方计时](../results/strict_fp16_check/official_acs5_bua3_final/benchmark_results.json)
- [官方任务结果](../results/strict_fp16_check/official_acs5_bua3_final/eval_results.json)
- [执行过的内核指令审计](../results/strict_fp16_check/official_acs5_bua3_final/kernel_audit/audit.json)
- [回归测试日志](../results/strict_fp16_check/tests.log)

`export_kernel_audit()` 导出每个实际编译版本的 PTX、cubin、SASS，并检查 FP32 浮点运算、
类型提升以及 Tensor Core 路径；发现禁用指令会报错。
这些证据仅覆盖已执行版本，换 GPU、形状或编译器后应重新审计。
审计不覆盖 encoder/head/scheduler 以及推理前的表生成，它们允许使用更高精度。

可重复的检查命令：

```bash
python -B -m unittest discover -s tests -v
python -B -m BACInfer.analysis.check_strict_fp16 \
  --checkpoint checkpoints/epoch=7550-test_mean_score=1.000.ckpt \
  --optimal_steps_dir assets/blockpush_strict/decoder_fp16_strict/original/optimal_steps/cosine \
  --num_caches 20 --num_bu_blocks 3 \
  --output_dir results/strict_fp16_check/acs20_bua3_audit_repeat
```

`diffusion_policy/` 已转换为主仓库直接管理的源码，保留全部 369 个上游受控文件和本地修改，
详情见 [VENDORED.md](../diffusion_policy/VENDORED.md)。精度实现和兼容修复可在同一个分支提交。
实际创建分支、提交和推送的终端命令见 [STRICT_FP16_RELEASE.md](STRICT_FP16_RELEASE.md)。
大型特征与内核编译产物留在本地。旁边旧的 `BAC` FP16 AMP 报告目录没有修改。
