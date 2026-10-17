- FP16 混合精度推理（Mixed Precision）：模型参数仍为 FP32，矩阵乘法等运算使用 FP16，部分数值敏感的操作保留 FP32。
- 全 FP16 推理（Pure FP16）：使用 `model.half()` 将浮点模型参数转换为 FP16，并处理输入、归一化、Scheduler 等运算的 dtype。实现更复杂，也更容易出现数值问题。

先完成混合精度，再判断是否需要全 FP16。

![alt text](image.png)
torch.autocast 会根据算子的支持情况和精度策略，为某些运算自动选择较低精度的数据类型



##  合适的位置是外层推理入口
### 1. BACInfer/scripts/eval_fast_diffusion_policy.py
在模型和 Checkpoint 完成加载后，推理调用。(包含调度器)
```python
import torch

policy.eval()

with torch.inference_mode():
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16
    ):
        result = policy.predict_action(obs)

```
### 2. 仅给 Transformer 启用 autocast

```python
for timestep in timesteps:

    # Transformer 内部使用混合精度
    with torch.autocast("cuda", dtype=torch.float16):
        noise_pred = model(
            sample=action,
            timestep=timestep,
            cond=condition
        )

    # Scheduler 使用 FP32 输入状态
    noise_pred = noise_pred.float()

    # 在 autocast 区域之外执行
    action = noise_scheduler.step(
        noise_pred,
        timestep,
        action
    ).prev_sample

```
### 3. 直接转换模型为fp16
```python
model = model.half()#模型参数fp16

```


# FP16 转换会怎样影响 ACS 和 BUA？

这是 BAC 算法复现中特别容易忽略的问题。

ACS 的输入来自不同去噪步的模块输出特征：

$$
b_k^{SA},\quad b_k^{CA},\quad b_k^{FFN}
$$

这些特征由网络计算产生。

当 FP32 转为 FP16 后：

$$
b_k^{FP16}\ne b_k^{FP32}
$$

数值量化与运算舍入可能影响 cosine 相似度，进而影响 ACS 生成的更新位置。

因此需要至少设计下面三组实验。

| 实验 | ACS 特征采集精度 | 在线推理精度 | 目的 |
| --- | --- | --- | --- |
| A | FP32 | FP32 | 复现原 BAC 基线 |
| B | FP32 | FP16 | 判断 FP32 调度能否直接复用 |
| C | FP16 | FP16 | 判断 FP16 重新调度是否更合适 |

在计算 FP16 特征的 cosine 或 L1 评分时，可以先转换为 FP32 进行统计：

```python
a = a.float()
b = b.float()

similarity = torch.nn.functional.cosine_similarity(
    a.flatten(),
    b.flatten(),
    dim=0
)
```

这样能够减少相似度统计自身的半精度误差，但不能消除 FP16 推理本身已经引入的差异。

这三组实验可以作为研究报告中非常明确的一项内容：调度表是否具有跨数值精度的可迁移性？

# 验证
第一：模型参数精度

```python
print(next(model.parameters()).dtype)
```

如果使用纯 autocast，输出可能仍然是 `torch.float32`，这是正常的。

第二：模块实际输出精度

```python
def check_dtype(module, inputs, output):
    if isinstance(output, torch.Tensor):
        print(
            module.__class__.__name__,
            output.dtype,
            output.shape
        )

handle = model.decoder.layers[0].linear1.register_forward_hook(
    check_dtype
)

# 运行一次真实前向
# ...

handle.remove()
```

这里的 `register_forward_hook()` 用于观察模块输出，并不直接改变模块计算结果。

第三：BAC 缓存精度

因为你提供的源码保存的是：

```python
cache['block_cache'][ff_block_key] = ff_result.detach()
```

它会保留 `ff_result` 当时的 dtype，而不是自动转换为 FP16。

所以测试时应该检查：

```python
for name, value in policy._cache['block_cache'].items():
    print(name, value.dtype, value.shape)
```