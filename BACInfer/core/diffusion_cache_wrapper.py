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
    ##  修改传入的Policy对象，给Policy安装缓存执行逻辑
    @staticmethod##静态方法，可以不创建对象直接 FastDiffusionPolicy.apply_cache（policy，）
    def apply_cache(policy,
                   cache_threshold: int = 5,
                   optimal_steps_dir: str = None,
                   num_caches: int = 30,
                   metric: str = 'cosine',
                   cache_mode: str = None,
                   num_bu_blocks: int = 0,
                   bu_blocks: List[str] = None):
        if cache_mode is None:
            if optimal_steps_dir is not None:
                cache_mode = 'optimal'
            else:
                cache_mode = 'threshold'
        if cache_mode == 'original':
            logger.info("Using original mode, no caching applied")
            return policy

        if cache_mode not in ('threshold', 'optimal'):
            raise ValueError(f'Unsupported cache mode: {cache_mode}')
        if cache_mode == 'threshold' and cache_threshold < 1:
            raise ValueError('cache_threshold must be positive')
        if cache_mode == 'optimal' and optimal_steps_dir is None:
            raise ValueError('optimal mode requires optimal_steps_dir')

        assert hasattr(policy, 'model')     #判断对象是否存在model属性
        model = policy.model
        assert isinstance(model, TransformerForDiffusion)  
        num_inference_steps = policy.num_inference_steps

        cache = {
            'threshold': cache_threshold, #缓存复用的判据
            'mode': cache_mode,
            'metric': metric,
            'optimal_steps_dir': optimal_steps_dir,#最优调度文件目录
            'num_caches': num_caches,
            'num_steps': num_inference_steps,#总去噪数目
            'current_step': -1,#当前去噪信号
            'block_cache': {},#保存个模块的缓存输出，{层名：特征}
            'block_steps': {},#各模块应该在哪些步骤更新
            'num_bu_blocks': num_bu_blocks,    #可复用块的数量
            'bu_blocks': bu_blocks,
        }
        policy._cache = cache # 动态给policy对象添加了一个_cache属性

        # Find all cacheable transformer layers
        cacheable_layers = FastDiffusionPolicy._find_cacheable_layers(model)  #遍历模型，找到适合做缓存的Transformer层
        policy._cacheable_layers = cacheable_layers
        logger.info(f"Found {len(cacheable_layers)} cacheable Transformer layers")

        # Load optimal schedules for optimal mode
        if cache_mode == 'optimal' and optimal_steps_dir:
            # Load per-block optimal cache update schedules
            FastDiffusionPolicy._load_block_optimal_steps(cache, cacheable_layers, optimal_steps_dir, num_caches, metric)# 加载每个block的最优缓存调度
            cache_threshold = None

        # Add caching functionality to each layer
        for layer_name, layer in cacheable_layers:
            FastDiffusionPolicy._add_cache_to_block(layer, layer_name, cache) #需要更新，计算结果存进cache['block_cache'][layer_name];不更新结果直接返回cache[‘block_cache'][layer_name]

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
            FastDiffusionPolicy._update_cache_flags(cache, current_step)# 更新当前时间步所有模块的执行状态

            # Execute forward pass with caching
            output = original_forward(sample, timestep, cond, **kwargs)

            return output

        # Replace model's forward method
        model.forward = types.MethodType(forward_with_cache, model)#将新函数绑定到当前模型对象

        # Add cache reset method，避免两个动作之间复用错误旧特征
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

        expected = {
            f'{name}_{suffix}' for name, _ in layers
            for suffix in ('sa_block', 'mha_block', 'ff_block')
        }
        missing = expected - set(block_steps)
        if missing:
            raise ValueError(f'Missing ACS schedules for blocks: {sorted(missing)}')
        for name, steps in block_steps.items():
            if not steps or 0 not in steps or any(
                    int(step) != step or not 0 <= int(step) < cache['num_steps'] for step in steps):
                raise ValueError(f'Invalid execution-index schedule for {name}: {steps}')

        if cache.get('num_bu_blocks', 0) > 0:
            logger.info("Applying BU (Bubbling Union) algorithm for step propagation")

            bu_blocks = cache.get('bu_blocks', None)

            if bu_blocks is None:
                from BACInfer.analysis.bu_block_selection import analyze_block_errors
                activation_dir = next(
                    (p for p in (steps_dir, *steps_dir.parents) if (p / 'activations.pkl').is_file()),
                    None,
                )
                if activation_dir is None:
                    raise ValueError('BUA requires explicit bu_blocks or calibration activations.pkl')
                selected = analyze_block_errors(
                    str(activation_dir / 'activations.pkl'),
                    str(activation_dir / 'bu_block_selection'), cache['num_bu_blocks'],
                )
                bu_blocks = [name for name, _ in selected]
            if not bu_blocks or not set(bu_blocks) <= set(block_steps):
                raise ValueError(f'Invalid BUA block selection: {bu_blocks}')
            cache['bu_blocks'] = list(bu_blocks)

            if len(bu_blocks) > 1:
                def get_layer_idx(block_key):
                    match = re.match(r'decoder\.layers\.(\d+)_([a-z_]+)', block_key)
                    if match:
                        return int(match.group(1))
                    return -1

                def get_block_type_priority(block_key):
                    match = re.match(r'decoder\.layers\.\d+_([a-z_]+)', block_key)
                    if match:
                        block_type = match.group(1)
                        priorities = {'sa_block': 0, 'mha_block': 1, 'ff_block': 2}
                        return priorities.get(block_type, 10)
                    return 10

                def get_block_type(block_key):
                    match = re.match(r'decoder\.layers\.\d+_([a-z_]+)', block_key)
                    if match:
                        return match.group(1)
                    return ""

                sorted_bu_blocks = sorted(bu_blocks, key=lambda x: (get_layer_idx(x), get_block_type_priority(x)))
                logger.info(f"BU: Sorted blocks: {sorted_bu_blocks}")

                all_ffn_blocks = []
                for block_key in block_steps.keys():
                    if get_block_type(block_key) == 'ff_block':
                        all_ffn_blocks.append(block_key)

                sorted_all_ffn_blocks = sorted(all_ffn_blocks, key=lambda x: get_layer_idx(x))
                logger.info(f"Phase 1, BU: All FFN blocks (sorted by layer): {sorted_all_ffn_blocks}")

                for i, block_key in enumerate(sorted_bu_blocks):
                    block_layer_idx = get_layer_idx(block_key)

                    deeper_ffn_blocks = []
                    for ffn_block in sorted_all_ffn_blocks:
                        ffn_layer_idx = get_layer_idx(ffn_block)
                        if ffn_layer_idx >= block_layer_idx:
                            deeper_ffn_blocks.append(ffn_block)

                    if deeper_ffn_blocks:
                        logger.info(f"BU: For block {block_key} (layer {block_layer_idx}), found deeper FFN blocks: {deeper_ffn_blocks}")

                        all_deeper_steps = set()
                        for deeper_ffn_block in deeper_ffn_blocks:
                            if deeper_ffn_block in block_steps:
                                all_deeper_steps.update(block_steps[deeper_ffn_block])

                        if block_key in block_steps and all_deeper_steps:
                            current_steps = set(block_steps[block_key])
                            missing_steps = all_deeper_steps - current_steps

                            if missing_steps:
                                updated_steps = sorted(list(current_steps.union(missing_steps)))
                                block_steps[block_key] = updated_steps
                                logger.info(f"BU: Added missing steps to {block_key}: {sorted(list(missing_steps))}. New steps: {updated_steps}")

            else:
                logger.info("BU: Insufficient blocks for propagation (need at least 2)")

            logger.info("\n===== BU Algorithm Summary =====" )
            logger.info(f"Total BU blocks processed: {len(bu_blocks)}")

            for block_key in sorted(bu_blocks):
                if block_key in block_steps:
                    steps_count = len(block_steps[block_key])
                    logger.info(f"  Block {block_key}: {steps_count} steps")

        cache['block_steps'] = block_steps
        logger.info(f"Loaded optimal steps for {len(block_steps)} computational blocks")
        # Fail loudly instead of silently running without any caching:
        # an empty schedule means every block recomputes every step and the
        # benchmark measures ~1.0x speedup that looks like a "successful" run.
        if len(block_steps) == 0:
            raise ValueError(
                f"No optimal steps files found under '{steps_dir}'. "
                f"Expected per-module subdirectories like "
                f"'{steps_dir}/<module_name>/optimal_steps_<module_name>_{num_caches}_{metric}.pkl'. "
                f"Hint: optimal_steps_dir must point at the metric subdirectory, "
                f"e.g. '.../original/optimal_steps/{metric}'."
            )

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

                x = tgt

                if has_sa_cache:
                    x = x + cache['block_cache'][sa_block_key]
                else:
                    sa_result = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                    if should_cache_sa:
                        cache['block_cache'][sa_block_key] = sa_result.detach()
                    x = x + sa_result

                if has_mha_cache:
                    x = x + cache['block_cache'][mha_block_key]
                else:
                    mha_result = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                    if should_cache_mha:
                        cache['block_cache'][mha_block_key] = mha_result.detach()
                    x = x + mha_result

                if has_ff_cache:
                    x = x + cache['block_cache'][ff_block_key]
                else:
                    ff_result = self._ff_block(self.norm3(x))
                    if should_cache_ff:
                        cache['block_cache'][ff_block_key] = ff_result.detach()
                    x = x + ff_result

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
