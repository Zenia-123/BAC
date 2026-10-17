# P3：Original FP32 与 AMP 对比

Checkpoint：`/home/zenia/Projects/BAC/BAC/checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt`；SHA-256：`0673d771bc652a61f97068935f51f862a731e82197613d9d02e1b500b7803f29`。

GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU；torch：2.11.0+cu128；权重来源：`model`。

观测来源：`environment`，形状 `[3, 3, 16]`；校准环境 seed 0，验证环境 seeds `[10000, 10001, 10002]`；校准噪声 seed 11，验证噪声 seeds `[101, 102, 103]`。

本机原始结果：`results/lowdim_fp16_20261017_full`。小型完整证据：[summary.json](lowdim_fp16/summary.json)。

每个验证 seed 使用相同观测和相同初始噪声；两种模式都运行全部去噪步。误差针对反归一化后的 action/action_pred，不是任务成功率。

| 噪声 seed | action MAE | RMSE | max_abs | relative L2 |
|---|---:|---:|---:|---:|
| 101 | 5.1571988e-06 | 5.9419867e-06 | 1.1526048e-05 | 0.00027864033 |
| 102 | 7.5044022e-06 | 9.500036e-06 | 1.6796163e-05 | 0.00044756354 |
| 103 | 7.1183313e-06 | 8.631584e-06 | 1.3086945e-05 | 0.00040519395 |

每步 model_output 与 trajectory 的误差记录在 AMP Original 的 `per_step_error` 中。Threshold 1 对照在同精度下通过采样一致性断言（rtol=1e-4，atol=1e-5；不要求 AMP 逐位相同）。P1 已验证原始 FP32 回归；本轮导入前后 369 个源码文件的内容哈希一致，未改采样模型代码。
