import torch
import types
import logging
import pickle
from typing import List, Dict, Any, Tuple
from pathlib import Path
import torch.nn as nn
import re

from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion

logger = logging.getLogger(__name__)

class FastDiffusionPolicy:

    @staticmethod
    def apply_cache(policy,
                   cache_threshold: int = 5,
                   optimal_steps_dir: str = None,
                   num_caches: int = 30,
                   metric: str = 'cosine',
                   cache_mode: str = None,
                   num_bu_blocks: int = 0):
        if cache_mode is None:
            if optimal_steps_dir is not None:
                cache_mode = 'optimal'
            else:
                cache_mode = 'threshold'
        if cache_mode == 'original':
            logger.info("Using original mode, no caching applied")
            return policy
        if cache_mode == 'optimal' and not optimal_steps_dir:
            raise ValueError('Optimal caching requires an optimal_steps_dir.')

        assert hasattr(policy, 'model')
        model = policy.model
        assert isinstance(model, TransformerForDiffusion)
        num_inference_steps = policy.num_inference_steps

        cache = {
            'threshold': cache_threshold,
            'mode': cache_mode,
            'metric': metric,
            'optimal_steps_dir': optimal_steps_dir,
            'num_caches': num_caches,
            'num_steps': num_inference_steps,
            'current_step': -1,
            'block_cache': {},
            'block_steps': {},
            'num_bu_blocks': num_bu_blocks,
        }
        policy._cache = cache

        # Find all cacheable transformer layers
        cacheable_layers = FastDiffusionPolicy._find_cacheable_layers(model)
        policy._cacheable_layers = cacheable_layers
        logger.info(f"Found {len(cacheable_layers)} cacheable Transformer layers")

        # Load optimal schedules for optimal mode
        if cache_mode == 'optimal' and optimal_steps_dir:
            # Load per-block optimal cache update schedules
            FastDiffusionPolicy._load_block_optimal_steps(cache, cacheable_layers, optimal_steps_dir, num_caches, metric)
            cache_threshold = None

        # Add caching functionality to each layer
        for layer_name, layer in cacheable_layers:
            FastDiffusionPolicy._add_cache_to_block(layer, layer_name, cache)

        # Save original forward method
        original_forward = model.forward

        # Create forward method with caching
        def forward_with_cache(self, sample, timestep, cond=None, **kwargs):
            """Forward pass with caching enabled"""
            cache = getattr(policy, '_cache', None)

            # Increment step counter
            cache['current_step'] += 1
            current_step = cache['current_step']

            # Update cache flags for all blocks at current step
            FastDiffusionPolicy._update_cache_flags(cache, current_step)

            # Execute forward pass with caching
            output = original_forward(sample, timestep, cond, **kwargs)

            return output

        # Replace model's forward method
        model.forward = types.MethodType(forward_with_cache, model)

        # Add cache reset method
        def reset_cache(self):

            if hasattr(self, '_cache'):
                self._cache['current_step'] = -1
                self._cache['block_cache'] = {}
                logger.debug("Cache reset")
            return self

        # Add method to policy object
        policy.reset_cache = types.MethodType(reset_cache, policy)

        # Save original predict_action method
        original_predict_action = policy.predict_action

        # Create predict_action with auto cache reset
        def predict_action_with_auto_reset(self, *args, **kwargs):
            """Automatically reset cache before each predict_action call"""
            # Reset cache
            self.reset_cache()
            # Call original predict_action
            return original_predict_action(*args, **kwargs)

        # Replace policy's predict_action method
        policy.predict_action = types.MethodType(predict_action_with_auto_reset, policy)

        # Log application info
        if cache_mode == 'threshold':
            logger.info(f"Threshold-based caching applied successfully, threshold: {cache_threshold}")
        elif cache_mode == 'optimal':
            logger.info(f"Optimal ACS caching applied successfully, steps_dir: {optimal_steps_dir}, num_caches: {num_caches}")

        return policy

    @staticmethod
    def _find_cacheable_layers(model) -> List[Tuple[str, Any]]:

        # Only search for TransformerDecoderLayer in decoder
        cacheable_layers = []

        # Process decoder layers
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
            for i, layer in enumerate(model.decoder.layers):
                if isinstance(layer, nn.TransformerDecoderLayer):
                    name = f'decoder.layers.{i}'
                    cacheable_layers.append((name, layer))
                    logger.info(f"Selected decoder layer for caching: {name}")

        # Fallback: search all modules for TransformerDecoderLayer
        if not cacheable_layers:
            for name, module in model.named_modules():
                if isinstance(module, nn.TransformerDecoderLayer):
                    cacheable_layers.append((name, module))
                    logger.info(f"Selected Transformer layer for caching: {name}")

        logger.info(f"Total {len(cacheable_layers)} cacheable Transformer layers found")
        return cacheable_layers

    @staticmethod
    def _load_block_optimal_steps(cache: Dict, layers: List[Tuple[str, Any]],
                                 optimal_steps_dir: str, num_caches: int, metric: str):

        steps_dir = Path(optimal_steps_dir)
        assert steps_dir.exists(), f"Optimal steps directory {optimal_steps_dir} does not exist"

        from BACInfer.core.precision import check_calibration
        calibration_dir = next((parent for parent in (steps_dir, *steps_dir.parents)
                                if (parent / 'activations.pkl').is_file()), None)
        precision = getattr(layers[0][1], '_inference_precision', 'fp32')
        if calibration_dir is not None:
            check_calibration(calibration_dir, precision)
        elif precision != 'fp32':
            raise ValueError('Strict ACS requires matching calibration with precision.json.')

        block_steps = {}

        for layer_name, layer in layers:
            dropout1_name = f"{layer_name}.dropout1"
            dropout2_name = f"{layer_name}.dropout2"
            dropout3_name = f"{layer_name}.dropout3"

            sa_block_key = f"{layer_name}_sa_block"
            steps_file = steps_dir/dropout1_name/f'optimal_steps_{dropout1_name}_{num_caches}_{metric}.pkl'
            if steps_file.exists():
                with open(steps_file, 'rb') as f:
                    steps = pickle.load(f)
                block_steps[sa_block_key] = steps
                logger.info(f"Loaded steps for self-attention block {sa_block_key}: {steps}")

            mha_block_key = f"{layer_name}_mha_block"
            steps_file = steps_dir/dropout2_name/f'optimal_steps_{dropout2_name}_{num_caches}_{metric}.pkl'
            if steps_file.exists():
                with open(steps_file, 'rb') as f:
                    steps = pickle.load(f)
                block_steps[mha_block_key] = steps
                logger.info(f"Loaded steps for multi-head attention block {mha_block_key}: {steps}")

            ff_block_key = f"{layer_name}_ff_block"
            steps_file = steps_dir/dropout3_name/f'optimal_steps_{dropout3_name}_{num_caches}_{metric}.pkl'
            if steps_file.exists():
                with open(steps_file, 'rb') as f:
                    steps = pickle.load(f)
                block_steps[ff_block_key] = steps
                logger.info(f"Loaded steps for feedforward block {ff_block_key}: {steps}")

        expected = {f'{name}_{kind}' for name, _ in layers
                    for kind in ('sa_block', 'mha_block', 'ff_block')}
        missing = expected - block_steps.keys()
        if missing:
            raise ValueError(f'Missing ACS schedules in {steps_dir}: {sorted(missing)}')
        for name, steps in block_steps.items():
            if (len(steps) != num_caches or not steps or steps[0] != 0
                    or list(steps) != sorted(set(steps))
                    or any(int(step) != step or not 0 <= step < cache['num_steps'] for step in steps)):
                raise ValueError(f'Invalid ACS schedule for {name}: {steps}')

        cache['bu_blocks'] = []
        cache['bu_added_updates'] = 0
        if cache.get('num_bu_blocks', 0) > 0:
            from BACInfer.analysis.bu_block_selection import analyze_block_errors
            # Locate the calibration that actually owns these schedule files.
            calibration_dir = next((parent for parent in (steps_dir, *steps_dir.parents)
                                    if (parent / 'activations.pkl').is_file()), None)
            if calibration_dir is None:
                raise FileNotFoundError(
                    f'BUA requires activations.pkl above {steps_dir}; run calibration first.')
            selected = analyze_block_errors(
                str(calibration_dir / 'activations.pkl'),
                str(calibration_dir / 'bu_block_selection'), cache['num_bu_blocks'])
            bu_blocks = [name for name, _ in selected]
            if len(bu_blocks) != cache['num_bu_blocks'] or not set(bu_blocks) <= expected:
                raise ValueError(f'Invalid BUA selection: {bu_blocks}')
            cache['bu_blocks'] = bu_blocks

            def layer_index(name):
                return int(re.match(r'decoder\.layers\.(\d+)_', name).group(1))

            def priority(name):
                return {'sa_block': 0, 'mha_block': 1, 'ff_block': 2}[name.split('_', 1)[1]]

            # Preserve the original same-layer/deeper-layer FFN union rule.
            ffn_blocks = sorted((name for name in block_steps if name.endswith('_ff_block')),
                                key=layer_index)
            for name in sorted(bu_blocks, key=lambda key: (layer_index(key), priority(key))):
                union = set()
                for ffn in ffn_blocks:
                    if layer_index(ffn) >= layer_index(name):
                        union.update(block_steps[ffn])
                extra = union - set(block_steps[name])
                block_steps[name] = sorted(set(block_steps[name]) | union)
                cache['bu_added_updates'] += len(extra - {cache['num_steps'] - 1})
                logger.info('BU: Added updates to %s: %s', name, sorted(extra))
            logger.info('BUA applied to %s; additional updates=%s',
                        bu_blocks, cache['bu_added_updates'])

        cache['block_steps'] = block_steps
        logger.info(f"Loaded optimal steps for {len(block_steps)} computational blocks")

    @staticmethod
    def _update_cache_flags(cache: Dict, current_step: int):

        cache_mode = cache['mode']

        if cache_mode == 'threshold':
            threshold = cache['threshold']
            should_cache = (current_step % threshold == 0)

            cache['should_cache'] = should_cache

        else:
            if 'steps_lookup' not in cache:
                steps_lookup = {}
                for block_key, steps in cache['block_steps'].items():
                    if not steps:
                        steps_lookup[block_key] = set()
                    else:
                        steps_lookup[block_key] = set(steps)
                        steps_lookup[block_key].add(cache['num_steps'] - 1)
                       
                cache['steps_lookup'] = steps_lookup

            should_cache = {}
            for block_key, step_set in cache['steps_lookup'].items():
                if not step_set:
                    should_cache[block_key] = False
                else:
                    should_cache[block_key] = current_step in step_set

            cache['should_cache'] = should_cache

    @staticmethod
    def _add_cache_to_block(layer, layer_name: str, cache: Dict):

        original_forward = layer.forward

        if isinstance(layer, nn.TransformerDecoderLayer):
            def forward_with_cache(self, tgt, memory, tgt_mask=None, memory_mask=None,
                                  tgt_key_padding_mask=None, memory_key_padding_mask=None,
                                  tgt_is_causal=None, memory_is_causal=False):
                cache_mode = cache['mode']
                cache_ff_block = cache.get('cache_ff_block', True)

                sa_block_key = f"{layer_name}_sa_block"
                mha_block_key = f"{layer_name}_mha_block"
                ff_block_key = f"{layer_name}_ff_block"

                # Determine whether to update cache
                if cache_mode == 'threshold':
                    should_cache = cache.get('should_cache', False)
                    should_cache_sa = should_cache
                    should_cache_mha = should_cache
                    should_cache_ff = should_cache and cache_ff_block
                else:
                    block_should_cache = cache.get('should_cache', {})
                    should_cache_sa = block_should_cache.get(sa_block_key, False)
                    should_cache_mha = block_should_cache.get(mha_block_key, False)
                    should_cache_ff = block_should_cache.get(ff_block_key, False) and cache_ff_block

                has_sa_cache = sa_block_key in cache['block_cache'] and not should_cache_sa
                has_mha_cache = mha_block_key in cache['block_cache'] and not should_cache_mha
                has_ff_cache = ff_block_key in cache['block_cache'] and not should_cache_ff

                add = getattr(self, '_residual_add', torch.add)
                x = tgt

                if has_sa_cache:
                    x = add(x, cache['block_cache'][sa_block_key])
                else:
                    sa_result = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                    if should_cache_sa:
                        cache['block_cache'][sa_block_key] = sa_result.detach()
                    x = add(x, sa_result)

                if has_mha_cache:
                    x = add(x, cache['block_cache'][mha_block_key])
                else:
                    mha_result = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                    if should_cache_mha:
                        cache['block_cache'][mha_block_key] = mha_result.detach()
                    x = add(x, mha_result)

                if has_ff_cache:
                    x = add(x, cache['block_cache'][ff_block_key])
                else:
                    ff_result = self._ff_block(self.norm3(x))
                    if should_cache_ff:
                        cache['block_cache'][ff_block_key] = ff_result.detach()
                    x = add(x, ff_result)

                return x
        else:
            def forward_with_cache(self, *args, **kwargs):
                cache_mode = cache['mode']
                block_key = layer_name

                if cache_mode == 'threshold':
                    should_cache = cache.get('should_cache', False)
                else:
                    block_should_cache = cache.get('should_cache', {})
                    should_cache = block_should_cache.get(block_key, False)

                if block_key in cache['block_cache'] and not should_cache:
                    return cache['block_cache'][block_key]

                output = original_forward(*args, **kwargs)

                if should_cache:
                    if isinstance(output, torch.Tensor):
                        cache['block_cache'][block_key] = output.detach()

                return output

        # Replace layer's forward method
        layer.forward = types.MethodType(forward_with_cache, layer)
