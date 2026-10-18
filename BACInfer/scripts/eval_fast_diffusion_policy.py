import sys
import os
import pathlib
import click
import hydra
import torch
import dill
import numpy as np
import time
import json
import logging
from omegaconf import OmegaConf
from copy import deepcopy

from thop import profile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy
from diffusion_policy.common.pytorch_util import dict_apply

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eval_script")

@click.command()
@click.option('-c', '--checkpoint', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--cache_threshold', default=5, type=int)
@click.option('--optimal_steps_dir', default=None)
@click.option('--num_caches', default=30, type=int)
@click.option('--metric', default='cosine')
@click.option('--cache_mode', default=None, type=click.Choice(['original', 'threshold', 'optimal']))
@click.option('--num_bu_blocks', default=0, type=int)
@click.option('--skip_video', is_flag=True)
@click.option('--precision', default='fp32', type=click.Choice(['fp32', 'decoder_fp16_strict']), show_default=True)
@click.option('--n_train', default=None, type=click.IntRange(min=0), help='Override training episodes from checkpoint.')
@click.option('--n_test', default=None, type=click.IntRange(min=0), help='Override test episodes from checkpoint.')
@click.option('--n_envs', default=None, type=click.IntRange(min=1), help='Number of parallel environments.')
@click.option('--max_steps', default=None, type=click.IntRange(min=1), help='Environment steps per episode.')
@click.option('--test_start_seed', default=None, type=click.IntRange(min=0), help='First test environment seed.')
@click.option('--seed', default=None, type=click.IntRange(min=0, max=2**31-1), help='Fix per-prediction sampling noise.')
@click.option('--rollout_only', is_flag=True, help='Skip warmup, THOP and benchmark; evaluate tasks only.')

def main(checkpoint, output_dir, device, cache_threshold, optimal_steps_dir,
         num_caches, metric, cache_mode, num_bu_blocks, skip_video,
         n_train, n_test, n_envs, max_steps, precision, test_start_seed, seed, rollout_only):
    torch.set_num_threads(1)
    from BACInfer.core.reproducibility import configure_reproducibility, PREDICTION_SEED_RULE
    from BACInfer.core.precision import calibration_metadata
    configure_reproducibility(seed)
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise click.ClickException('CUDA is unavailable; activate the bac environment.')
    if cache_mode == 'optimal' and not optimal_steps_dir:
        raise click.ClickException('--cache_mode optimal requires --optimal_steps_dir; generate schedules first.')
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    cache_mode = cache_mode or ('optimal' if optimal_steps_dir else 'threshold')
    protocol = calibration_metadata(checkpoint, precision)
    protocol.update({'seed': seed, 'prediction_seed_rule': PREDICTION_SEED_RULE if seed is not None else None,
                     'rollout_only': rollout_only, 'torch_version': str(torch.__version__),
                     'device': str(torch.device(device)), 'torch_num_threads': torch.get_num_threads(),
                     'gpu': torch.cuda.get_device_name(torch.device(device)) if torch.device(device).type == 'cuda' else None,
                     'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                     'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                     'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32})

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    logger.info(f"Configuration loaded: {cfg._target_}")

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    logger.info("Workspace loaded successfully")

    original_policy = workspace.model
    original_policy.to(device)
    original_policy.eval()
    from BACInfer.core.precision import set_policy_precision
    set_policy_precision(original_policy, precision)
    logger.info('Precision: %s', precision)
    logger.info('Policy source: %s', sys.modules[type(original_policy).__module__].__file__)

    has_model = hasattr(original_policy, 'model')
    logger.info(f"Policy type: {type(original_policy).__name__}")
    if has_model:
        logger.info(f"Model type: {type(original_policy.model).__name__}")

    actions_per_inference = original_policy.n_action_steps
    logger.info(f"Number of action frames per inference: {actions_per_inference}")

    logger.info("Creating policy copy for cache acceleration...")
    fast_policy = deepcopy(original_policy)
    
    if cache_mode == 'optimal':
        logger.info(f"Applying optimal steps-based cache acceleration, dir={optimal_steps_dir}, metric={metric}")
        fast_policy = FastDiffusionPolicy.apply_cache(
            policy=fast_policy,
            cache_mode='optimal',
            optimal_steps_dir=optimal_steps_dir,
            num_caches=num_caches,
            metric=metric,
            num_bu_blocks=num_bu_blocks
        )
    else:  # threshold or original
        logger.info(f"Applying threshold-based cache acceleration, threshold={cache_threshold}")
        fast_policy = FastDiffusionPolicy.apply_cache(
            policy=fast_policy,
            cache_mode=cache_mode,  # Could be None, 'threshold', or 'original'
            cache_threshold=cache_threshold,
            num_bu_blocks=num_bu_blocks
        )
    
    speedup_time = None
    flops_value = 0.0
    if not rollout_only:
        B = 1
        To = cfg.n_obs_steps
    
        if hasattr(cfg, 'shape_meta'):
            logger.info("Detected image model configuration")
            obs_dict = {}
            for key, shape in cfg.shape_meta['obs'].items():
                tensor_shape = [B, To] + list(shape['shape'])
                obs_dict[key] = torch.zeros(tensor_shape, device=device, dtype=torch.float32)
        else:
            logger.info("Detected low-dimensional model configuration")
            obs_dim = cfg.obs_dim if hasattr(cfg, 'obs_dim') else None
        
            if obs_dim is None and hasattr(cfg, 'task'):
                task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
                if 'obs_dim' in task_cfg:
                    obs_dim = task_cfg['obs_dim']
                    logger.info(f"Using obs_dim from configuration: {obs_dim}")
        
            obs_dict = {
                'obs': torch.zeros((B, To, obs_dim), device=device, dtype=torch.float32)
            }
        
            if hasattr(cfg, 'use_past_action') and cfg.use_past_action:
                action_dim = cfg.action_dim
                obs_dict['past_action'] = torch.zeros((B, To, action_dim), device=device, dtype=torch.float32)

        logger.info(f"Created input dictionary with keys: {list(obs_dict.keys())}")
        for key, tensor in obs_dict.items():
            logger.info(f"  {key}: shape={tensor.shape}")
    
        logger.info("Warming up...")
        with torch.no_grad():
            original_action = original_policy.predict_action(obs_dict)
            fast_action = fast_policy.predict_action(obs_dict)
            if not all(torch.isfinite(value).all() for result in (original_action, fast_action) for value in result.values()):
                raise RuntimeError('Nonfinite action during warmup.')
        if precision != 'fp32':
            from diffusion_policy.model.diffusion.strict_fp16 import export_kernel_audit
            export_kernel_audit(pathlib.Path(output_dir) / 'kernel_audit')
        logger.info("Warmup complete")
    
        flops_value = 0.0

        if precision == 'fp32':
            logger.info("\nComputing policy FLOPs using thop...")
            obs_dict_copy = {}
            for key, value in obs_dict.items():
                obs_dict_copy[key] = value.clone()
        
            with torch.no_grad():
                nobs = fast_policy.normalizer.normalize(obs_dict_copy)
                value = next(iter(nobs.values()))
                B, To = value.shape[:2]
        
                if hasattr(fast_policy, 'action_dim'):
                    Da = fast_policy.action_dim
                elif hasattr(cfg, 'action_dim'):
                    Da = cfg.action_dim
                else:
                    Da = cfg.task.action_dim
        
                if hasattr(fast_policy, 'obs_feature_dim'):
                    Do = fast_policy.obs_feature_dim
                else:
                    if hasattr(cfg, 'obs_dim'):
                        Do = cfg.obs_dim
                    else:
                        Do = cfg.task.obs_dim
        
                logger.info(f"Action dimension: {Da}, Observation feature dimension: {Do}")
        
                device = fast_policy.device
                dtype = torch.float32
                timestep = torch.zeros(B, dtype=torch.long, device=device)
        
                cond = None
                sample = None
        
                action_dim = fast_policy.action_dim
                obs_feature_dim = fast_policy.obs_dim if hasattr(fast_policy, 'obs_dim') else Do

                logger.info(f"Action dimension: {action_dim}, Observation feature dimension: {obs_feature_dim}")
        
                if hasattr(fast_policy, 'obs_encoder'):
                    if fast_policy.obs_as_cond:
                        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                        nobs_features = fast_policy.obs_encoder(this_nobs)
                        cond = nobs_features.reshape(B, To, -1)
                        sample = torch.zeros(size=(B, fast_policy.horizon, action_dim), device=device, dtype=dtype)
                    else:
                        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                        nobs_features = fast_policy.obs_encoder(this_nobs)
                        nobs_features = nobs_features.reshape(B, To, -1)
                        sample = torch.zeros(size=(B, fast_policy.horizon, action_dim+obs_feature_dim), device=device, dtype=dtype)
                        sample[:,:To,action_dim:] = nobs_features
                else:
                    if hasattr(fast_policy, 'obs_as_cond') and fast_policy.obs_as_cond:
                        cond = obs_dict['obs'][:,:To]
                        sample = torch.zeros(size=(B, fast_policy.horizon, action_dim), device=device, dtype=dtype)
                        if hasattr(fast_policy, 'pred_action_steps_only') and fast_policy.pred_action_steps_only:
                            sample = torch.zeros(size=(B, fast_policy.n_action_steps, action_dim), device=device, dtype=dtype)
                    else:
                        sample = torch.zeros(size=(B, fast_policy.horizon, action_dim+obs_feature_dim), device=device, dtype=dtype)
                        sample[:,:To,action_dim:] = obs_dict['obs'][:,:To]
        
                fast_policy.eval()
                macs, params = profile(fast_policy.model, inputs=(sample, timestep, cond), verbose=False)
                flops_value = macs * 2
        
                logger.info(f"Model parameter count: {params/1e6:.2f} M")
                logger.info(f"Model MACs: {macs/1e9:.4f} G")
                logger.info(f"Model FLOPs: {flops_value/1e9:.4f} G")

        else:
            logger.info('THOP does not count strict custom kernels; FLOPs left unavailable.')

        logger.info(f"\n=== Performance Testing (action frames/inference: {actions_per_inference}) ===")
    
        num_trials = 10
    
        logger.info("\n----- Original Policy -----")
        original_durations = []
    
        for i in range(num_trials):
            torch.cuda.synchronize()
            start_time = time.time()
        
            with torch.no_grad():
                original_policy.predict_action(obs_dict)
            
            torch.cuda.synchronize()
            duration = time.time() - start_time
            original_durations.append(duration)
            logger.info(f"Run {i+1}/{num_trials}: {duration:.4f} seconds")
    
        avg_original = np.mean(original_durations)
        frequency_original = actions_per_inference / avg_original
        logger.info(f"Average time: {avg_original:.4f} seconds")
        logger.info(f"Action frequency: {frequency_original:.2f} actions/second")
    
        logger.info("\n----- Cached Policy -----")
        fast_durations = []
    
        for i in range(num_trials):
            torch.cuda.synchronize()
            start_time = time.time()
        
            with torch.no_grad():
                fast_policy.predict_action(obs_dict)
            
            torch.cuda.synchronize()
            duration = time.time() - start_time
            fast_durations.append(duration)
            logger.info(f"Run {i+1}/{num_trials}: {duration:.4f} seconds")
    
        avg_fast = np.mean(fast_durations)
        frequency_fast = actions_per_inference / avg_fast
        speedup_time = avg_original / avg_fast
    
        logger.info(f"Average time: {avg_fast:.4f} seconds")
        logger.info(f"Action frequency: {frequency_fast:.2f} actions/second")
        logger.info(f"Speedup: {speedup_time:.2f}x")
    
        cache_config = {
            "device": str(device),
            "precision": precision,
            "timing_reference": f'{precision} uncached',
            "mode": cache_mode,
            "actions_per_inference": int(actions_per_inference),
            "num_bu_blocks": int(num_bu_blocks),
        }
    
    
        if cache_mode == 'optimal':
            cache_config.update({
                "optimal_steps_dir": optimal_steps_dir,
                "metric": metric,
                "num_caches": num_caches
            })
        else:  # threshold
            cache_config.update({
                "cache_threshold": cache_threshold,
            })
    
        benchmark_results = {
            **cache_config,
            "original": {
                "avg_time": float(avg_original),
                "frequency": float(frequency_original),
            },
            "fast": {
                "avg_time": float(avg_fast),
                "frequency": float(frequency_fast),
            },
            "speedup": float(speedup_time),
            "flops": float(flops_value) if precision == 'fp32' else None,
            "config": OmegaConf.to_container(cfg, resolve=True)
        }
    
        with open(os.path.join(output_dir, 'benchmark_results.json'), 'w') as f:
            json.dump(benchmark_results, f, indent=2)
    
        logger.info(f"Benchmark results saved to {output_dir}/benchmark_results.json")
    
    env_runner_cfg = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
    for key, value in {'n_train': n_train, 'n_test': n_test,
                       'n_envs': n_envs, 'max_steps': max_steps,
                       'test_start_seed': test_start_seed}.items():
        if value is not None:
            env_runner_cfg[key] = value
    if seed is not None:
        if not env_runner_cfg['_target_'].endswith('.BlockPushLowdimRunner'):
            raise click.ClickException('--seed per-prediction control currently supports BlockPushLowdimRunner only.')
        env_runner_cfg['policy_seed'] = seed
    if env_runner_cfg['n_train'] + env_runner_cfg['n_test'] == 0:
        raise click.ClickException('At least one train or test episode is required.')
    if skip_video:
        env_runner_cfg['n_train_vis'] = 0
        env_runner_cfg['n_test_vis'] = 0
        logger.info("Skip video rendering")
    
    env_runner = hydra.utils.instantiate(env_runner_cfg, output_dir=output_dir)
    
    logger.info("\nRunning environment evaluation with cached policy...")
    try:
        fast_runner_log = env_runner.run(fast_policy)
    finally:
        env_runner.env.close()
    
    protocol['num_inference_steps'] = fast_policy.num_inference_steps
    protocol['env_runner_config'] = env_runner_cfg
    protocol['rollout_trace'] = getattr(env_runner, 'rollout_trace', [])
    with open(os.path.join(output_dir, 'rollout_protocol.json'), 'w') as f:
        json.dump(protocol, f, indent=2)

    test_mean_score = fast_runner_log.get('test/mean_score', 0.0)
    
    eval_results = {
        "precision": precision,
        "mean_score": float(test_mean_score),
        "speedup": float(speedup_time) if speedup_time is not None else None,
        "flops": float(flops_value) if precision == 'fp32' and not rollout_only else None,
        "cache_mode": cache_mode,
        "num_caches": num_caches if cache_mode in ['optimal'] else None,
        "num_bu_blocks": int(num_bu_blocks)
    }
    eval_results['seed'] = seed
    eval_results['checkpoint_sha256'] = protocol['checkpoint_sha256']
    eval_results['p1'] = fast_runner_log.get('test/p1')
    eval_results['p2'] = fast_runner_log.get('test/p2')
    eval_results['scores'] = {key.split('test/sim_max_reward_', 1)[1]: value
                              for key, value in fast_runner_log.items()
                              if key.startswith('test/sim_max_reward_')}
    eval_results['env_runner_config'] = env_runner_cfg
    eval_results['bu_blocks'] = fast_policy._cache.get('bu_blocks', []) if hasattr(fast_policy, '_cache') else []
    
    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_dir}/eval_results.json")
    logger.info("Test mean score: %.4f; p2=%s", test_mean_score, eval_results['p2'])
    if speedup_time is not None:
        logger.info("Speedup: %.2fx", speedup_time)
    if flops_value > 0:
        logger.info(f"FLOPs: {flops_value/1e9:.4f} GFLOPs")
    
    json_log = {k: v._path if hasattr(v, '_path') else v for k, v in fast_runner_log.items()}
    out_path = os.path.join(output_dir, 'fast_eval_log.json')
    with open(out_path, 'w') as f:
        json.dump(json_log, f, indent=2, sort_keys=True)
    logger.info(f"Detailed evaluation logs saved to {out_path}")
    if precision != 'fp32':
        from diffusion_policy.model.diffusion.strict_fp16 import export_kernel_audit
        export_kernel_audit(pathlib.Path(output_dir) / 'kernel_audit')

if __name__ == "__main__":
    main()
