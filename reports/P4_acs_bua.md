# P4：ACS、BUA 与精度调度消融

Checkpoint：`/home/zenia/Projects/BAC/BAC/checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt`；SHA-256：`0673d771bc652a61f97068935f51f862a731e82197613d9d02e1b500b7803f29`。

GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU；torch：2.11.0+cu128；权重来源：`model`。

观测来源：`environment`，形状 `[3, 3, 16]`；校准环境 seed 0，验证环境 seeds `[10000, 10001, 10002]`；校准噪声 seed 11，验证噪声 seeds `[101, 102, 103]`。

本机原始结果：`results/lowdim_fp16_my_run`。小型完整证据：[summary.json](lowdim_fp16/summary.json)。

沿用项目 OptimalCacheScheduler 的分段动态规划和 BUA 下游 FFN 调度并集传播。校准特征按真实 FP32/AMP 路径采集，余弦矩阵与分支 L1 排序统计使用 FP32。BUA block 采用仓库原有跨时间特征 L1 定义；未将该定义冒充任务误差排名。

修复：缺少任意 block 调度时报错；BUA 无校准数据时不再静默跳过；调度索引使用去噪循环执行序号，末步由硬编码 99 改为 num_steps-1。当前 100 步实验保留原有末步额外更新行为，所以预算并不总等于实际更新次数。

| 模式 | 分支执行总数 | BUA 增加更新点数（不含末步） | 同精度 Original 的 action MAE 均值 |
|---|---:|---:|---:|
| fp32_acs5_fp32 | 144 | 0 | 0.0055629086 |
| fp32_acs5_fp32_bua | 152 | 8 | 0.0048114034 |
| fp16_amp_acs5_fp32 | 144 | 0 | 0.0055827566 |
| fp16_amp_acs5_fp32_bua | 152 | 8 | 0.0048362229 |
| fp16_amp_acs5_fp16_amp | 144 | 0 | 0.0055822943 |
| fp16_amp_acs5_fp16_amp_bua | 152 | 8 | 0.004839074 |
| fp32_acs20_fp32 | 502 | 0 | 0.00023793464 |
| fp32_acs20_fp32_bua | 516 | 14 | 0.00025548326 |
| fp16_amp_acs20_fp32 | 502 | 0 | 0.00023483489 |
| fp16_amp_acs20_fp32_bua | 516 | 14 | 0.00024923582 |
| fp16_amp_acs20_fp16_amp | 502 | 0 | 0.00026011722 |
| fp16_amp_acs20_fp16_amp_bua | 516 | 14 | 0.00029863203 |

所选 BUA blocks、原始调度和有效调度均保存在 JSON；FP32 与 AMP 的原始特征和调度目录独立。实际调用次数证明跳过了分支计算，但不能当作完整 FLOPs、能耗或带宽测量。
