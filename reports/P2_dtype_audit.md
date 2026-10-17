# P2：完整采样、模块与缓存精度审计

Checkpoint：`/home/zenia/Projects/BAC/BAC/checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt`；SHA-256：`0673d771bc652a61f97068935f51f862a731e82197613d9d02e1b500b7803f29`。

GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU；torch：2.11.0+cu128；权重来源：`model`。

观测来源：`environment`，形状 `[3, 3, 16]`；校准环境 seed 0，验证环境 seeds `[10000, 10001, 10002]`；校准噪声 seed 11，验证噪声 seeds `[101, 102, 103]`。

本机原始结果：`results/lowdim_fp16_my_run`。小型完整证据：[summary.json](lowdim_fp16/summary.json)。

两种精度均使用 FP32 参数。采样时模型入口与条件为 FP32，Transformer 内开启或关闭 autocast，原始输出在 AMP 下为 FP16，交给 Scheduler 前转回 FP32。Scheduler 的输入、输出和轨迹均检查为 FP32，且位于 autocast 外。

| 模块输出 | FP32 | AMP |
|---|---|---|
| input_emb | torch.float32 | torch.float16 |
| encoder | torch.float32 | torch.float16 |
| decoder.layers.0.norm1 | torch.float32 | torch.float32 |
| decoder.layers.0.self_attn | torch.float32 | torch.float16 |
| decoder.layers.0.multihead_attn | torch.float32 | torch.float16 |
| decoder.layers.0.linear1 | torch.float32 | torch.float16 |
| decoder.layers.0.linear2 | torch.float32 | torch.float16 |
| decoder | torch.float32 | torch.float32 |
| head | torch.float32 | torch.float16 |

| 模式 | 模型调用 | Scheduler 调用 | 分支执行总数 | 缓存写入 / 复用 |
|---|---:|---:|---:|---|
| fp32_original | 100 | 100 | 2400 | 0 / 0 |
| fp32_threshold1 | 100 | 100 | 2400 | 2400 / 0 |
| fp32_threshold5 | 100 | 100 | 480 | 480 / 1920 |
| fp16_amp_original | 100 | 100 | 2400 | 0 / 0 |
| fp16_amp_threshold1 | 100 | 100 | 2400 | 2400 / 0 |
| fp16_amp_threshold5 | 100 | 100 | 480 | 480 / 1920 |
| fp32_acs5_fp32 | 100 | 100 | 144 | 144 / 2256 |
| fp32_acs5_fp32_bua | 100 | 100 | 152 | 152 / 2248 |
| fp16_amp_acs5_fp32 | 100 | 100 | 144 | 144 / 2256 |
| fp16_amp_acs5_fp32_bua | 100 | 100 | 152 | 152 / 2248 |
| fp16_amp_acs5_fp16_amp | 100 | 100 | 144 | 144 / 2256 |
| fp16_amp_acs5_fp16_amp_bua | 100 | 100 | 152 | 152 / 2248 |
| fp32_acs20_fp32 | 100 | 100 | 502 | 502 / 1898 |
| fp32_acs20_fp32_bua | 100 | 100 | 516 | 516 / 1884 |
| fp16_amp_acs20_fp32 | 100 | 100 | 502 | 502 / 1898 |
| fp16_amp_acs20_fp32_bua | 100 | 100 | 516 | 516 / 1884 |
| fp16_amp_acs20_fp16_amp | 100 | 100 | 502 | 502 / 1898 |
| fp16_amp_acs20_fp16_amp_bua | 100 | 100 | 516 | 516 / 1884 |

每个缓存 Tensor 的 dtype、shape、写入/复用次数均在 JSON 中，且写入次数与对应分支实际执行次数一致。观测到的数据张量均检查有限；注意力 mask 中设计的 -inf 不属于激活错误。所有观察 hook 和 Scheduler 包装均在结束后恢复。模块边界 dtype 不是融合内核内部累加精度的证明。
