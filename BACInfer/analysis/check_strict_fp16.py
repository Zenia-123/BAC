"""Real-checkpoint dtype/finite/count audit and paired fixed-input comparison."""
from copy import deepcopy
import json
from pathlib import Path

import click
import torch

from BACInfer.analysis.run_policy import run_policy
from BACInfer.core.precision import set_policy_precision
from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy


def audit(policy, obs, noise_seed):
    records = {}
    hooks = []
    scheduler_inputs = []
    def probe(name):
        def hook(module, args, output):
            values = [output] if isinstance(output, torch.Tensor) else list(output) if isinstance(output, tuple) else []
            record = records.setdefault(name, {'calls':0,'dtypes':set(),'finite':True})
            record['calls'] += 1
            for value in values:
                if isinstance(value, torch.Tensor):
                    record['dtypes'].add(str(value.dtype))
                    record['finite'] &= bool(torch.isfinite(value).all())
        return hook
    for name, module in policy.model.named_modules():
        if name.startswith('decoder.layers') or name in ('decoder','ln_f','head','input_emb','encoder'):
            hooks.append(module.register_forward_hook(probe(name)))
    scheduler = policy.noise_scheduler
    original_step = scheduler.step
    def step(model_output, timestep, sample, **kwargs):
        scheduler_inputs.append([str(model_output.dtype),str(sample.dtype)])
        return original_step(model_output,timestep,sample,**kwargs)
    scheduler.step = step
    try:
        torch.manual_seed(noise_seed)
        with torch.inference_mode():
            action = policy.predict_action(obs)
    finally:
        scheduler.step = original_step
        for hook in hooks:
            hook.remove()
    for name, record in records.items():
        record['dtypes'] = sorted(record['dtypes'])
        if not record['finite']:
            raise AssertionError(f'Nonfinite at {name}')
        strict = getattr(policy,'inference_precision','fp32') != 'fp32'
        if strict and (name.startswith('decoder') or name == 'ln_f'):
            assert record['dtypes'] == ['torch.float16'], (name,record)
    assert len(scheduler_inputs) == policy.num_inference_steps
    assert all(pair == ['torch.float32','torch.float32'] for pair in scheduler_inputs)
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in action.values())
    cache = getattr(policy,'_cache',{})
    branch_calls = {name:record['calls'] for name,record in records.items()
                    if name.endswith(('dropout1','dropout2','dropout3'))}
    return action, {'noise_seed':noise_seed,'modules':records,'scheduler_calls':len(scheduler_inputs),
            'scheduler_input_dtypes':sorted({tuple(pair) for pair in scheduler_inputs}),
            'branch_executions':sum(branch_calls.values()),'branch_calls':branch_calls,
            'cache_dtypes':sorted({str(t.dtype) for t in cache.get('block_cache',{}).values()}),
            'bu_blocks':cache.get('bu_blocks',[]),'bu_added_updates':cache.get('bu_added_updates',0)}


@click.command()
@click.option('--checkpoint', required=True, type=click.Path(exists=True))
@click.option('--optimal_steps_dir', required=True, type=click.Path(exists=True))
@click.option('--output_dir', default='results/strict_fp16_check/paired_audit')
@click.option('--device', default='cuda:0')
@click.option('--num_caches', default=20, type=click.IntRange(min=1), show_default=True)
@click.option('--num_bu_blocks', default=3, type=click.IntRange(min=0), show_default=True)
def main(checkpoint, optimal_steps_dir, output_dir, device, num_caches, num_bu_blocks):
    torch.set_num_threads(1)
    root = Path(output_dir)
    root.mkdir(parents=True,exist_ok=True)
    ref, obs = run_policy(checkpoint,root,device,0,num_bu_blocks=0)
    strict = set_policy_precision(deepcopy(ref),'decoder_fp16_strict')
    cached = deepcopy(strict)
    FastDiffusionPolicy.apply_cache(cached,cache_mode='optimal',optimal_steps_dir=optimal_steps_dir,
                                   num_caches=num_caches,metric='cosine',num_bu_blocks=num_bu_blocks)
    cache_name = f'strict_acs{num_caches}_bua{num_bu_blocks}'
    summary = {'environment_seed':0,'batch':1,'denoising_steps':ref.num_inference_steps,
               'num_caches':num_caches,'num_bu_blocks':num_bu_blocks,'comparisons':[]}
    for seed in (101,102,103):
        a, ra = audit(ref,obs,seed)
        b, rb = audit(strict,obs,seed)
        c, rc = audit(cached,obs,seed)
        comparisons = {}
        for name, current, reference in [('strict_vs_fp32',b,a),(cache_name+'_vs_fp32',c,a),(cache_name+'_vs_strict',c,b)]:
            comparisons[name] = {}
            for key in a:
                difference = current[key] - reference[key]
                comparisons[name][key] = {'mae':difference.abs().mean().item(),'max_abs':difference.abs().max().item()}
        summary['comparisons'].append({'noise_seed':seed,'errors':comparisons,
                                       'fp32':ra,'strict':rb,cache_name:rc})
    from diffusion_policy.model.diffusion.strict_fp16 import export_kernel_audit
    summary['kernel_audit_passed'] = export_kernel_audit(root/'kernel_audit')['passed']
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    click.echo(f'Passed: {root}/summary.json')


if __name__ == '__main__':
    main()
