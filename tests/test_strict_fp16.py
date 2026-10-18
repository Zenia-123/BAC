"""Numerical, instruction and cache regressions for the strict CUDA path."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import BACInfer
import torch

from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy
from BACInfer.core.precision import check_calibration, set_policy_precision
from test_fp32_entrypoints import policy, schedules


class ProvenanceTests(unittest.TestCase):
    def test_strict_schedules_cannot_be_mixed_with_fp32(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'Missing calibration provenance'):
                check_calibration(root, 'decoder_fp16_strict')
            (root/'precision.json').write_text(json.dumps({'precision': 'fp32'}))
            with self.assertRaisesRegex(ValueError, 'precision mismatch'):
                check_calibration(root, 'decoder_fp16_strict')


@unittest.skipUnless(torch.cuda.is_available(), 'Strict decoder needs CUDA')
class StrictTests(unittest.TestCase):
    def test_matmul_rounds_product_before_half_reduction(self):
        from diffusion_policy.model.diffusion.strict_fp16 import matmul
        # Exact FP32 products leave 2**-20; rounding each half product cancels to zero.
        a = torch.tensor([[[1+2**-10, 1]]], device='cuda', dtype=torch.float16)
        b = torch.tensor([[[1+2**-10], [-(1+2**-9)]]], device='cuda', dtype=torch.float16)
        with torch.no_grad():
            actual = matmul(a,b)
        self.assertEqual(actual.item(), 0)
        self.assertEqual(torch.bmm(a.float(), b.float()).item(), 2**-20)

    def test_layernorm_constant_and_nearby_values(self):
        from diffusion_policy.model.diffusion.strict_fp16 import StrictLayerNorm
        norm = StrictLayerNorm(256).cuda().half().eval()
        with torch.no_grad():
            x = torch.full((2,5,256), 3., device='cuda', dtype=torch.float16)
            self.assertEqual(norm(x).abs().max().item(), 0)
            torch.manual_seed(42)
            x = torch.randn_like(x) * 0.25
            y = norm(x)
            self.assertTrue(torch.isfinite(y).all())
            torch.testing.assert_close(y.float(), torch.nn.functional.layer_norm(x.float(), (256,)), rtol=0.005, atol=0.005)

    def test_masked_attention_matches_reference_at_small_scale(self):
        from diffusion_policy.model.diffusion.strict_fp16 import StrictAttention
        torch.manual_seed(2)
        ref = torch.nn.MultiheadAttention(8,2,batch_first=True,dropout=0).cuda().eval()
        strict = deepcopy(ref).half()
        strict.__class__ = StrictAttention
        from diffusion_policy.model.diffusion.strict_fp16 import StrictLinear
        strict.out_proj.__class__ = StrictLinear
        x = torch.randn(2,3,8,device='cuda') * 0.1
        memory = torch.randn(2,5,8,device='cuda') * 0.1
        mask = torch.zeros(3,5,device='cuda')
        mask[:,2:] = -float('inf')
        with torch.no_grad():
            y, _ = strict(x.half(),memory.half(),memory.half(),attn_mask=mask.half())
            expected, _ = ref(x,memory,memory,attn_mask=mask,need_weights=False)
            torch.testing.assert_close(y.float(),expected,rtol=.03,atol=.0005)
            changed = memory.clone()
            changed[:,2:] = 200
            z, _ = strict(x.half(),changed.half(),changed.half(),attn_mask=mask.half())
            torch.testing.assert_close(y,z,rtol=0,atol=0)

    def test_full_model_dtype_boundary_and_cache_hits(self):
        p = policy().cuda()
        before_keys = list(p.state_dict())
        set_policy_precision(p,'decoder_fp16_strict')
        self.assertEqual(before_keys, list(p.state_dict()))
        self.assertTrue(all(t.dtype == torch.float16 for t in p.model.decoder.parameters()))
        self.assertEqual(p.model.input_emb.weight.dtype,torch.float32)
        self.assertEqual(p.model.head.weight.dtype,torch.float32)
        x = torch.randn(2,3,2,device='cuda') * .1
        cond = torch.randn(2,2,16,device='cuda') * .1
        with torch.no_grad():
            y = p.model(x,3,cond)
            self.assertEqual(y.dtype,torch.float32)
            cached = deepcopy(p)
            FastDiffusionPolicy.apply_cache(cached,cache_mode='threshold',cache_threshold=1)
            z = cached.model(x,3,cond)
            torch.testing.assert_close(y,z,rtol=0,atol=0)
            self.assertEqual(len(cached._cache['block_cache']),6)
            self.assertTrue(all(t.dtype == torch.float16 for t in cached._cache['block_cache'].values()))
            # Actual reuse: no dropout branch should execute on a skipped step.
            reused = deepcopy(p)
            FastDiffusionPolicy.apply_cache(reused,cache_mode='threshold',cache_threshold=2)
            reused.model(x,3,cond)
            writes = dict(reused._cache['block_cache'])
            hits = []
            hooks = [m.register_forward_hook(lambda *args: hits.append(1)) for n,m in reused.model.named_modules()
                     if n.endswith(('dropout1','dropout2','dropout3'))]
            try:
                reused.model(x,2,cond)
                self.assertFalse(hits)
                self.assertTrue(all(writes[k] is v for k,v in reused._cache['block_cache'].items()))
            finally:
                for h in hooks:
                    h.remove()
        with self.assertRaisesRegex(RuntimeError,'inference-only'):
            p.model(x,3,cond)

    def test_acs_bua_and_last_step_with_strict_layers(self):
        from BACInfer.core.precision import ARITHMETIC_VERSION
        p = set_policy_precision(policy().cuda(),'decoder_fp16_strict')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = schedules(root)
            (root/'precision.json').write_text(json.dumps({'precision':'decoder_fp16_strict',
                'arithmetic_version': ARITHMETIC_VERSION}))
            FastDiffusionPolicy.apply_cache(p,cache_mode='optimal',optimal_steps_dir=path,num_caches=2,num_bu_blocks=2)
            with torch.no_grad():
                for step in range(4):
                    p.model(torch.zeros(1,3,2,device='cuda'),3-step,torch.zeros(1,2,16,device='cuda'))
            self.assertTrue(p._cache['bu_blocks'])
            self.assertTrue(all(3 in points for points in p._cache['steps_lookup'].values()))
            self.assertTrue(all(t.dtype == torch.float16 for t in p._cache['block_cache'].values()))

    def test_z_instruction_audit(self):
        from diffusion_policy.model.diffusion.strict_fp16 import export_kernel_audit
        with tempfile.TemporaryDirectory() as directory:
            audit = export_kernel_audit(directory)
            self.assertTrue(audit['passed'])
            self.assertTrue(any(r['kernel'] == '_softmax' for r in audit['kernels']))


if __name__ == '__main__':
    unittest.main()
