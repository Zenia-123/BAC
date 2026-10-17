"""Reproducible lowdim FP32/AMP, ACS/BUA, dtype and rollout experiment.

Run with python -B -m BACInfer.analysis.lowdim_fp16_experiment --help.
No THOP estimate is used as a latency or full-sampling FLOPs measurement.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import gc
import hashlib
import json
import logging
from pathlib import Path
import pickle
import time
from unittest.mock import patch

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from BACInfer.analysis.analyze_cache_no_env import capture
from BACInfer.analysis.bu_block_selection import compute_block_l1_errors, select_top_error_blocks
from BACInfer.analysis.check_lowdim_fp16 import describe, observe_forward
from BACInfer.analysis.optimal_cache_scheduler import OptimalCacheScheduler
from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / ('checkpoint/low_dim/block_pushing/diffusion_policy_transformer/'
                     'train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt')


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_policy(checkpoint, device, weights):
    with checkpoint.open('rb') as file:
        payload = torch.load(file, pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model if weights == 'model' else workspace.ema_model
    if policy is None:
        raise ValueError('Requested policy weights are absent')
    policy = policy.to(device).eval()
    if not policy.obs_as_cond or policy.model.decoder is None:
        raise ValueError('This BAC experiment requires a conditioned lowdim decoder policy')
    return policy, cfg


def make_env(seed, runner_cfg):
    from gym.wrappers import FlattenObservation
    from diffusion_policy.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal
    return FlattenObservation(BlockPushMultimodal(
        control_frequency=10, shared_memory=False, seed=seed,
        abs_action=runner_cfg.get('abs_action', False)))


def reset_obs(env, runner_cfg):
    obs = np.asarray(env.reset(), dtype=np.float32)
    if not runner_cfg.get('obs_eef_target', True):
        obs[8:10] = 0
    return obs


def observations(policy, runner_cfg, seeds, source):
    if source == 'zeros':
        return {'obs': torch.zeros((len(seeds), policy.n_obs_steps, policy.obs_dim),
                                   device=policy.device)}
    windows = []
    for seed in seeds:
        env = make_env(seed, runner_cfg)
        try:
            obs = reset_obs(env, runner_cfg)
            windows.append(np.repeat(obs[None], policy.n_obs_steps, axis=0))
        finally:
            env.close()
    tensor = torch.as_tensor(np.stack(windows), device=policy.device)
    if tuple(tensor.shape[1:]) != (policy.n_obs_steps, policy.obs_dim):
        raise ValueError(f'Unexpected observation shape {tensor.shape}')
    describe(tensor)
    return {'obs': tensor}


def block_key(name):
    layer, dropout = name.rsplit('.', 1)
    suffix = {'dropout1': 'sa_block', 'dropout2': 'mha_block', 'dropout3': 'ff_block'}[dropout]
    return f'{layer}_{suffix}'


def audit(policy, obs, seed, save_features=False):
    """Observe one whole sampling run; remove all instrumentation afterwards."""
    stats = {'model_calls': 0, 'scheduler_calls': 0, 'branches': {},
             'cache': {}, 'model_output_dtypes': [], 'trajectory_dtype': 'torch.float32'}
    raw_dtypes = set()
    previous_cache = {}
    traces = []
    step = policy.noise_scheduler.step

    def checked_step(model_output, timestep, trajectory, **kwargs):
        if torch.is_autocast_enabled('cuda'):
            raise AssertionError('Scheduler entered autocast')
        if model_output.dtype != torch.float32 or trajectory.dtype != torch.float32:
            raise AssertionError('Scheduler state must be FP32')
        describe(model_output)
        describe(trajectory)
        result = step(model_output, timestep, trajectory, **kwargs)
        if result.prev_sample.dtype != torch.float32:
            raise AssertionError('Scheduler output must be FP32')
        describe(result.prev_sample)
        traces.append((model_output.detach().cpu(), result.prev_sample.detach().cpu()))
        stats['scheduler_calls'] += 1
        return result

    def before_model(module, inputs):
        sample, timestep, cond = inputs
        assert sample.dtype == torch.float32
        assert cond is None or cond.dtype == torch.float32
        assert torch.is_autocast_enabled('cuda') == (policy.inference_precision == 'fp16_amp')
        describe(sample)
        describe(cond)

    def after_model(module, inputs, output):
        describe(output)
        raw_dtypes.add(str(output.dtype))
        stats['model_calls'] += 1
        for name, tensor in getattr(policy, '_cache', {}).get('block_cache', {}).items():
            item = stats['cache'].setdefault(name, {**describe(tensor), 'writes': 0, 'reuses': 0})
            describe(tensor)
            if previous_cache.get(name) is tensor:
                item['reuses'] += 1
            else:
                item['writes'] += 1
            previous_cache[name] = tensor

    def branch_hook(name):
        def hook(module, inputs, output):
            item = stats['branches'].setdefault(name, {**describe(output), 'calls': 0})
            describe(output)
            item['calls'] += 1
        return hook

    with ExitStack() as stack:
        handles = [policy.model.register_forward_pre_hook(before_model),
                   policy.model.register_forward_hook(after_model)]
        for name, module in policy.model.named_modules():
            if name.endswith(('.dropout1', '.dropout2', '.dropout3')):
                handles.append(module.register_forward_hook(branch_hook(name)))
        for handle in handles:
            stack.callback(handle.remove)
        stack.enter_context(patch.object(policy.noise_scheduler, 'step', checked_step))
        with torch.inference_mode():
            stats['obs'] = describe(obs['obs'])
            stats['nobs'] = describe(policy.normalizer['obs'].normalize(obs['obs']))
            output, counts, features, timesteps = capture(policy, obs, seed, save_features)
    n = policy.num_inference_steps
    assert stats['model_calls'] == stats['scheduler_calls'] == n
    assert all(p.dtype == torch.float32 for p in policy.parameters())
    for name, item in stats['cache'].items():
        branch = next(key for key in counts if block_key(key) == name)
        assert item['writes'] == counts[branch]
        assert item['writes'] + item['reuses'] == n
    stats['model_output_dtypes'] = sorted(raw_dtypes)
    stats['timesteps'] = timesteps
    stats['block_compute_counts'] = counts
    stats['outputs'] = {key: describe(value) for key, value in output.items()}
    return output, stats, features, traces


def errors(reference, output):
    values = {}
    for key in ('action', 'action_pred'):
        ref, pred = reference[key].float(), output[key].float()
        delta = pred - ref
        values[key] = {'mae': delta.abs().mean().item(),
                       'rmse': delta.square().mean().sqrt().item(),
                       'max_abs': delta.abs().max().item(),
                       'relative_l2': (delta.norm() / ref.norm().clamp_min(1e-8)).item()}
    return values


def calibrate(policy, obs, output_dir, budgets, seed):
    output, stats, features, _ = audit(policy, obs, seed, save_features=True)
    assert all(len(values) == policy.num_inference_steps for values in features.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'activations.pkl').open('wb') as file:
        pickle.dump(features, file)
    similarities = {}
    schedules = {}
    for name, values in features.items():
        flat = torch.stack(values).float().flatten(start_dim=1)
        normalized = F.normalize(flat, dim=1, eps=1e-8)
        cosine = (normalized @ normalized.T).numpy()
        assert np.isfinite(cosine).all()
        similarities[name] = {'cosine': cosine}
        for budget in budgets:
            scheduler = OptimalCacheScheduler(similarity_matrix=cosine, metric='cosine')
            steps = scheduler.compute_optimal_steps(budget)
            assert len(steps) == budget and steps[0] == 0
            path = output_dir / 'optimal_steps/cosine' / name
            scheduler.save_optimal_steps(str(path / f'optimal_steps_{name}_{budget}_cosine.pkl'))
            schedules.setdefault(str(budget), {})[block_key(name)] = [int(x) for x in steps]
    with (output_dir / 'similarity_matrices.pkl').open('wb') as file:
        pickle.dump(similarities, file)
    ranking = select_top_error_blocks(compute_block_l1_errors(features), num_blocks=len(features))
    return {'dtype_audit': stats, 'schedules': schedules,
            'bu_ranking': [(name, float(error)) for name, error in ranking]}


def candidate(base, spec, calibrations, output_dir, bu_count):
    policy = deepcopy(base)
    policy.inference_precision = spec['precision']
    if spec['mode'] == 'threshold':
        FastDiffusionPolicy.apply_cache(policy, cache_mode='threshold', cache_threshold=spec['budget'])
    elif spec['mode'] == 'optimal':
        calibration = calibrations[spec['schedule_precision']]
        selected = [name for name, _ in calibration['bu_ranking'][:bu_count]] if spec['bu'] else None
        FastDiffusionPolicy.apply_cache(
            policy, cache_mode='optimal', num_caches=spec['budget'],
            optimal_steps_dir=str(output_dir / 'calibration' / spec['schedule_precision'] / 'optimal_steps/cosine'),
            num_bu_blocks=bu_count if spec['bu'] else 0, bu_blocks=selected)
    return policy


def benchmark(policy, obs, seed, trials, warmup):
    def run(i):
        seed_all(seed + i)
        policy.reset()
        with torch.inference_mode():
            return policy.predict_action(obs)
    for i in range(warmup):
        run(i)
    torch.cuda.synchronize(policy.device)
    torch.cuda.reset_peak_memory_stats(policy.device)
    durations = []
    for i in range(trials):
        seed_all(seed + i)
        policy.reset()
        torch.cuda.synchronize(policy.device)
        start = time.perf_counter()
        with torch.inference_mode():
            policy.predict_action(obs)
        torch.cuda.synchronize(policy.device)
        durations.append((time.perf_counter() - start) * 1000)
    return {'mean_ms': float(np.mean(durations)), 'std_ms': float(np.std(durations)),
            'p50_ms': float(np.median(durations)), 'samples_ms': durations,
            'peak_allocated_mb': torch.cuda.max_memory_allocated(policy.device) / 2**20,
            'warmup': warmup, 'trials': trials}


def add_speedups(report):
    fp32_time = report['results']['fp32_original']['benchmark']['mean_ms']
    amp_time = report['results']['fp16_amp_original']['benchmark']['mean_ms']
    for item in report['results'].values():
        timing = item['benchmark']
        timing['speedup_vs_fp32_original'] = fp32_time / timing['mean_ms']
        baseline = fp32_time if item['spec']['precision'] == 'fp32' else amp_time
        timing['speedup_vs_same_precision_original'] = baseline / timing['mean_ms']


def rebenchmark(run_dir, device, trials, warmup):
    """Refresh timings only; preserve the previous report and all evaluations."""
    report_path = run_dir / 'summary.json'
    report = json.loads(report_path.read_text())
    checkpoint = Path(report['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report['checkpoint_sha256']:
        raise ValueError('Checkpoint content differs from the evaluated checkpoint')
    base, cfg = load_policy(checkpoint, device, report['weights'])
    runner_cfg = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
    obs = observations(base, runner_cfg, report['validation_environment_seeds'], report['observation_source'])
    if list(obs['obs'].shape) != report['obs_shape']:
        raise ValueError('Observation shape differs from the original experiment')
    stamp = time.time_ns()
    backup = run_dir / f'summary_before_rebenchmark_{stamp}.json'
    backup.write_text(report_path.read_text())
    for label, item in report['results'].items():
        gc.collect()
        policy = candidate(base, item['spec'], report['calibrations'], run_dir, report['bu_blocks'])
        item['benchmark'] = benchmark(policy, obs, report['calibration_noise_seed'], trials, warmup)
        print('Retimed', label, round(item['benchmark']['mean_ms'], 2), 'ms', flush=True)
        del policy
        gc.collect()
    add_speedups(report)
    report['timing_recheck'] = {
        'previous_summary': backup.name,
        'reason': 'Reclaim candidate method/closure reference cycles before each measurement',
        'gpu': torch.cuda.get_device_name(base.device), 'torch': str(torch.__version__),
        'torch_num_threads': torch.get_num_threads(),
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report


def rollout(policy, runner_cfg, seeds, seed, max_steps):
    envs = []
    try:
        envs = [make_env(s, runner_cfg) for s in seeds]
        histories = [np.repeat(reset_obs(env, runner_cfg)[None], policy.n_obs_steps, axis=0) for env in envs]
        rewards = [set() for _ in envs]
        done = [False] * len(envs)
        policy.reset()
        for t in range(max_steps):
            obs = {'obs': torch.as_tensor(np.stack(histories), device=policy.device)}
            seed_all(seed + t)
            with torch.inference_mode():
                actions = policy.predict_action(obs)['action'].cpu().numpy()
            assert np.isfinite(actions).all()
            # This checkpoint executes one action per plan; reject other protocols.
            if actions.shape[1] != 1:
                raise ValueError('Rollout currently requires n_action_steps=1')
            for i, env in enumerate(envs):
                if done[i]:
                    continue
                ob, reward, terminal, info = env.step(actions[i, 0])
                ob = np.asarray(ob, dtype=np.float32)
                if not runner_cfg.get('obs_eef_target', True):
                    ob[8:10] = 0
                histories[i] = np.concatenate([histories[i][1:], ob[None]])
                rewards[i].add(float(reward))
                done[i] = bool(terminal)
            if all(done):
                break
        scores = [sum(values) for values in rewards]
        return {'environment_seeds': seeds, 'max_steps': max_steps,
                'score_definition': 'sum of unique rewards, matching BlockPushLowdimRunner',
                'scores': scores, 'mean_score': float(np.mean(scores)),
                'p1': float(np.mean(np.array(scores) > 0.4)),
                'p2': float(np.mean(np.array(scores) > 0.9)),
                'episodes': len(seeds), 'steps_executed': t + 1}
    finally:
        for env in envs:
            env.close()


def write_reports(report, run_dir, report_dir):
    """Export compact evidence and human-readable reports, without GPU work."""
    report_dir.mkdir(parents=True, exist_ok=True)
    archive = report_dir / 'lowdim_fp16'
    archive.mkdir(exist_ok=True)
    (archive / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    provenance = (
        f"Checkpoint：`{report['checkpoint']}`；SHA-256：`{report['checkpoint_sha256']}`。\n\n"
        f"GPU：{report['gpu']}；torch：{report['torch']}；权重来源：`{report['weights']}`。\n\n"
        f"观测来源：`{report['observation_source']}`，形状 `{report['obs_shape']}`；"
        f"校准环境 seed 0，验证环境 seeds `{report['validation_environment_seeds']}`；"
        f"校准噪声 seed {report['calibration_noise_seed']}，验证噪声 seeds `{report['validation_noise_seeds']}`。\n\n"
        f"本机原始结果：`{run_dir}`。小型完整证据：[summary.json](lowdim_fp16/summary.json)。\n\n")
    lines = ['# P2：完整采样、模块与缓存精度审计\n\n', provenance,
             '两种精度均使用 FP32 参数。采样时模型入口与条件为 FP32，'
             'Transformer 内开启或关闭 autocast，原始输出在 AMP 下为 FP16，'
             '交给 Scheduler 前转回 FP32。Scheduler 的输入、输出和轨迹均检查为 FP32，且位于 autocast 外。\n\n',
             '| 模块输出 | FP32 | AMP |\n|---|---|---|\n']
    for name in ('input_emb', 'encoder', 'decoder.layers.0.norm1', 'decoder.layers.0.self_attn',
                 'decoder.layers.0.multihead_attn', 'decoder.layers.0.linear1',
                 'decoder.layers.0.linear2', 'decoder', 'head'):
        dtypes = []
        for precision in ('fp32', 'fp16_amp'):
            row = next(row for row in report['single_forward'][precision]['modules'] if row['module'] == name)
            dtypes.append(row['output']['dtype'])
        lines.append(f"| {name} | {dtypes[0]} | {dtypes[1]} |\n")
    lines += ['\n| 模式 | 模型调用 | Scheduler 调用 | 分支执行总数 | 缓存写入 / 复用 |\n|---|---:|---:|---:|---|\n']
    for label, item in report['results'].items():
        stats = item['dtype_audit']
        writes = sum(x['writes'] for x in stats['cache'].values())
        reuses = sum(x['reuses'] for x in stats['cache'].values())
        lines.append(f"| {label} | {stats['model_calls']} | {stats['scheduler_calls']} | "
                     f"{sum(stats['block_compute_counts'].values())} | {writes} / {reuses} |\n")
    lines += ['\n每个缓存 Tensor 的 dtype、shape、写入/复用次数均在 JSON 中，且写入次数与对应分支实际执行次数一致。'
              '观测到的数据张量均检查有限；注意力 mask 中设计的 -inf 不属于激活错误。'
              '所有观察 hook 和 Scheduler 包装均在结束后恢复。模块边界 dtype 不是融合内核内部累加精度的证明。\n']
    (report_dir / 'P2_dtype_audit.md').write_text(''.join(lines))
    amp = report['results']['fp16_amp_original']
    p3 = ['# P3：Original FP32 与 AMP 对比\n\n', provenance,
          '每个验证 seed 使用相同观测和相同初始噪声；两种模式都运行全部去噪步。'
          '误差针对反归一化后的 action/action_pred，不是任务成功率。\n\n',
          '| 噪声 seed | action MAE | RMSE | max_abs | relative L2 |\n|---|---:|---:|---:|---:|\n']
    for row in amp['validation']:
        e = row['vs_fp32']['action']
        p3.append(f"| {row['seed']} | {e['mae']:.8g} | {e['rmse']:.8g} | {e['max_abs']:.8g} | {e['relative_l2']:.8g} |\n")
    p3 += ['\n每步 model_output 与 trajectory 的误差记录在 AMP Original 的 `per_step_error` 中。'
           'Threshold 1 对照在同精度下通过采样一致性断言（rtol=1e-4，atol=1e-5；不要求 AMP 逐位相同）。P1 已验证原始 FP32 回归；'
           '本轮导入前后 369 个源码文件的内容哈希一致，未改采样模型代码。\n']
    (report_dir / 'P3_original_precision_comparison.md').write_text(''.join(p3))
    p4 = ['# P4：ACS、BUA 与精度调度消融\n\n', provenance,
          '沿用项目 OptimalCacheScheduler 的分段动态规划和 BUA 下游 FFN 调度并集传播。'
          '校准特征按真实 FP32/AMP 路径采集，余弦矩阵与分支 L1 排序统计使用 FP32。'
          'BUA block 采用仓库原有跨时间特征 L1 定义；未将该定义冒充任务误差排名。\n\n',
          '修复：缺少任意 block 调度时报错；BUA 无校准数据时不再静默跳过；'
          '调度索引使用去噪循环执行序号，末步由硬编码 99 改为 num_steps-1。'
          '当前 100 步实验保留原有末步额外更新行为，所以预算并不总等于实际更新次数。\n\n',
          '| 模式 | 分支执行总数 | BUA 增加更新点数（不含末步） | 同精度 Original 的 action MAE 均值 |\n|---|---:|---:|---:|\n']
    for label, item in report['results'].items():
        if item['spec']['mode'] != 'optimal':
            continue
        mae = np.mean([x['vs_same_precision_original']['action']['mae'] for x in item['validation']])
        p4.append(f"| {label} | {sum(item['dtype_audit']['block_compute_counts'].values())} | "
                  f"{item['bu_added_updates']} | {mae:.8g} |\n")
    p4 += ['\n所选 BUA blocks、原始调度和有效调度均保存在 JSON；FP32 与 AMP 的原始特征和调度目录独立。'
           '实际调用次数证明跳过了分支计算，但不能当作完整 FLOPs、能耗或带宽测量。\n']
    (report_dir / 'P4_acs_bua.md').write_text(''.join(p4))
    timing_protocol = report['results']['fp32_original']['benchmark']
    episode_count = len(report['validation_environment_seeds'])
    p5 = ['# P5：Lowdim FP16 实验结果\n\n', provenance,
          f"计时在观察 hook 移除后进行，逐次 CUDA 同步，预热 {timing_protocol['warmup']} 次、测量 {timing_protocol['trials']} 次。"
          '种子设置在计时区间外，区间包含完整 predict_action；batch 大小见观测形状。'
          '运行命令固定 OMP_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1。峰值显存为已分配显存，包含常驻原始策略与候选副本。'
          'THOP 不参与该表的速度或 FLOPs 结论。\n\n',
          '| 模式 | mean ± std ms | p50 ms | 相对 FP32 Original | 相对同精度 Original | action MAE vs FP32 | 峰值 MB | rollout score / p2 |\n|---|---:|---:|---:|---:|---:|---:|---|\n']
    for label, item in report['results'].items():
        b = item['benchmark']
        mae = np.mean([x['vs_fp32']['action']['mae'] for x in item['validation']])
        r = item.get('rollout')
        score = f"{r['mean_score']:.4f} / {r['p2']:.4f}" if r else '未测'
        p5.append(f"| {label} | {b['mean_ms']:.2f} ± {b['std_ms']:.2f} | {b['p50_ms']:.2f} | "
                  f"{b['speedup_vs_fp32_original']:.3f}x | {b['speedup_vs_same_precision_original']:.3f}x | "
                  f"{mae:.8g} | {b['peak_allocated_mb']:.2f} | {score} |\n")
    p5 += ['\nrollout score 按 BlockPushLowdimRunner 定义为各 episode 不同奖励的和，p2 为 score > 0.9 的比例，'
           '二者不能混称成功率。闭环 rollout 的观测会因策略动作不同而变化；'
           '动作误差来自另一组固定观测、固定噪声的成对实验。\n\n',
           f'局限：单一校准窗口、{episode_count} 个环境 seeds、短期单机计时；不宣称论文规模复现，'
           '不保证 FP16 或 BUA 在每个设置下都更快、更准。所有失败或退化应按实际数值解读。\n\n',
           '源码已纳入 BAC 普通目录，不依赖子模块补丁。运行和后续开发方法见 '
           '[LOWDIM_FP16_RUNBOOK.md](LOWDIM_FP16_RUNBOOK.md)。\n']
    if report.get('timing_recheck'):
        p5 += ['\n本表使用候选之间显式回收循环引用后的复测计时。初次 summary 保留为 '
               f"`{report['timing_recheck']['previous_summary']}`，dtype、误差和 rollout 沿用同一轮完整评估。\n"]
    (report_dir / 'P5_lowdim_fp16_results.md').write_text(''.join(p5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('-o', '--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--weights', choices=['model', 'ema'], default='model')
    parser.add_argument('--observation-source', choices=['environment', 'zeros'], default='environment')
    parser.add_argument('--budgets', default='5,20')
    parser.add_argument('--bu-blocks', type=int, default=3)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--validation-seeds', default='101,102,103')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--rollout-episodes', type=int, default=3)
    parser.add_argument('--rollout-budget', type=int, default=20)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--report-dir', type=Path, help='Export Markdown and a compact JSON summary here')
    parser.add_argument('--rebenchmark', action='store_true', help='Retain evaluations and retime an existing output directory')
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if not torch.cuda.is_available() or torch.device(args.device).type != 'cuda':
        parser.error('CUDA is required')
    budgets = sorted(set(int(x) for x in args.budgets.split(',')))
    seeds = [int(x) for x in args.validation_seeds.split(',')]
    if args.trials < 1 or args.warmup < 1 or not seeds or args.bu_blocks < 2 or args.rollout_episodes < 0:
        parser.error('Require trials/warmup >= 1, bu-blocks >= 2, episodes >= 0 and validation seeds')
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('max-steps must be positive')
    if args.rebenchmark:
        report = rebenchmark(args.output_dir, args.device, args.trials, args.warmup)
        if args.report_dir:
            write_reports(report, args.output_dir, args.report_dir)
        print('BENCHMARK RECHECK PASSED', args.output_dir, flush=True)
        return
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a new directory to preserve results')
    base, cfg = load_policy(args.checkpoint, args.device, args.weights)
    if not budgets or any(not 1 <= n <= base.num_inference_steps for n in budgets):
        parser.error('Invalid ACS budgets')
    runner_cfg = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
    env_seeds = list(range(10000, 10000 + max(1, args.rollout_episodes)))
    obs = observations(base, runner_cfg, env_seeds, args.observation_source)
    calibration_obs = observations(base, runner_cfg, [0], args.observation_source)
    args.output_dir.mkdir(parents=True)
    report = {'checkpoint': str(args.checkpoint.resolve()),
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'weights': args.weights, 'torch': str(torch.__version__),
              'torch_num_threads': torch.get_num_threads(),
              'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
              'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
              'gpu': torch.cuda.get_device_name(base.device),
              'model_config': OmegaConf.to_container(cfg.policy.model, resolve=True),
              'num_inference_steps': base.num_inference_steps,
              'observation_source': args.observation_source, 'obs_shape': list(obs['obs'].shape),
              'calibration_environment_seed': 0, 'validation_environment_seeds': env_seeds,
              'calibration_noise_seed': args.seed, 'validation_noise_seeds': seeds,
              'bu_blocks': args.bu_blocks, 'calibrations': {}, 'results': {},
              'protocol': 'FP32 parameters, Transformer-only AMP, FP32 Scheduler; no THOP timing',
              'limitations': ['One calibration observation window; not dataset-wide calibration.',
                              'Short local GPU timings; memory includes base and candidate weights.',
                              'Few rollout episodes are not a paper-scale success-rate reproduction.']}
    # Single-forward module audit uses one shared normalized input and sample.
    with torch.inference_mode():
        cond = base.normalizer['obs'].normalize(calibration_obs['obs'])
        length = base.n_action_steps if base.pred_action_steps_only else base.horizon
        generator = torch.Generator(device=base.device).manual_seed(args.seed)
        sample = torch.randn((1, length, base.action_dim), generator=generator, device=base.device)
        base.noise_scheduler.set_timesteps(base.num_inference_steps)
        timestep = base.noise_scheduler.timesteps[0].to(base.device)
        report['single_forward'] = {p: observe_forward(base.model, sample, timestep, cond, p == 'fp16_amp')
                                    for p in ('fp32', 'fp16_amp')}
    for precision in ('fp32', 'fp16_amp'):
        print('Calibrating', precision, flush=True)
        base.inference_precision = precision
        report['calibrations'][precision] = calibrate(
            base, calibration_obs, args.output_dir / 'calibration' / precision, budgets, args.seed)
    specs = []
    for precision in ('fp32', 'fp16_amp'):
        specs.append(dict(label=f'{precision}_original', precision=precision, mode='original', budget=0, bu=False))
        for interval in (1, 5):
            specs.append(dict(label=f'{precision}_threshold{interval}', precision=precision,
                              mode='threshold', budget=interval, bu=False))
    for budget in budgets:
        for precision, schedule in [('fp32', 'fp32'), ('fp16_amp', 'fp32'), ('fp16_amp', 'fp16_amp')]:
            for bu in (False, True):
                specs.append(dict(label=f'{precision}_acs{budget}_{schedule}' + ('_bua' if bu else ''),
                                  precision=precision, mode='optimal', budget=budget,
                                  schedule_precision=schedule, bu=bu))
    references = {}
    reference_traces = {}
    for precision in ('fp32', 'fp16_amp'):
        base.inference_precision = precision
        for seed in seeds:
            output, stats, _, traces = audit(base, obs, seed)
            references[precision, seed] = output
            reference_traces[precision, seed] = traces
    for spec in specs:
        print('Running', spec['label'], flush=True)
        # Cache forwards bind methods/closures into cycles. Reclaim the previous
        # candidate before measuring the next candidate's allocated memory.
        gc.collect()
        policy = candidate(base, spec, report['calibrations'], args.output_dir, args.bu_blocks)
        validations = []
        for seed in seeds:
            output, stats, _, traces = audit(policy, obs, seed)
            if spec['mode'] == 'threshold' and spec['budget'] == 1:
                for key in ('action', 'action_pred'):
                    torch.testing.assert_close(output[key], references[spec['precision'], seed][key], rtol=1e-4, atol=1e-5)
            if spec['mode'] == 'optimal':
                for name, count in stats['block_compute_counts'].items():
                    assert count == len(policy._cache['steps_lookup'][block_key(name)])
            validations.append({'seed': seed, 'vs_fp32': errors(references['fp32', seed], output),
                                'vs_same_precision_original': errors(references[spec['precision'], seed], output)})
        item = {'spec': spec, 'validation': validations, 'dtype_audit': stats,
                'benchmark': benchmark(policy, obs, args.seed, args.trials, args.warmup)}
        if spec['mode'] == 'original' and spec['precision'] == 'fp16_amp':
            item['per_step_error'] = [
                {'index': i, 'model_output_mae': (a[0] - b[0]).abs().mean().item(),
                 'trajectory_mae': (a[1] - b[1]).abs().mean().item()}
                for i, (a, b) in enumerate(zip(reference_traces['fp32', seeds[-1]], traces))]
        if spec['mode'] == 'optimal':
            effective = {name: sorted(int(x) for x in values) for name, values in policy._cache['steps_lookup'].items()}
            raw = report['calibrations'][spec['schedule_precision']]['schedules'][str(spec['budget'])]
            item['effective_schedules'] = effective
            item['bu_selected'] = policy._cache.get('bu_blocks')
            item['bu_added_updates'] = sum(len(set(effective[k]) - (set(v) | {base.num_inference_steps - 1})) for k, v in raw.items())
        do_rollout = spec['mode'] == 'original' or (spec['mode'] == 'optimal' and spec['budget'] == args.rollout_budget)
        if args.rollout_episodes and do_rollout:
            print('Rollout', spec['label'], env_seeds, flush=True)
            item['rollout'] = rollout(policy, runner_cfg, env_seeds, args.seed,
                                      args.max_steps or runner_cfg.get('max_steps', 200))
        report['results'][spec['label']] = item
        (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        print(spec['label'], 'PASS', round(item['benchmark']['mean_ms'], 2), 'ms', flush=True)
        del policy
        gc.collect()
    add_speedups(report)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.report_dir:
        write_reports(report, args.output_dir, args.report_dir)
    print('ALL LOWDIM FP16 EXPERIMENT CHECKS PASSED', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
