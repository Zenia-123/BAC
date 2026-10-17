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
    rows = [] #每条 hook 记录一行
    handles = [] # 收集 RemovableHandle，供 finally 清理

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
    policy = workspace.model.to(device).eval()    # 与 eval 采用相同的权重来源，不切换到 EMA
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
        generator = torch.Generator(device=device).manual_seed(args.seed) # torch.Generator(...).manual_seed(11)	生成一份固定的初始轨迹；两次 forward 复用同一份 Tensor
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
