# Local Diffusion Policy source

This directory is tracked directly by the BAC repository. It is no longer a Git
submodule, so a normal clone of the strict FP16 branch includes the modified
Transformer, strict decoder kernels, and environment compatibility changes.

Upstream: https://github.com/real-stanford/diffusion_policy

Upstream revision: `5ba07ac6661db573af695b419a7947ecb704690f`

All 369 files tracked at that revision were retained, including the original
[MIT license](LICENSE). Local changes add `strict_fp16.py`, configure decoder
precision in `transformer_for_diffusion.py`, support paired evaluation in
`blockpush_lowdim_runner.py`, and adapt the vector environments to the local Gym
interface. Existing source bytes were preserved during conversion.

Edit these files normally and commit them together with the BAC changes. No
separate submodule repository, patch application, or submodule push is required.
Checkpoint weights, activation dumps, videos, and generated kernel audit files
are local runtime artifacts and are not part of this source snapshot.

The root `.gitattributes` preserves upstream whitespace without flagging the whole
snapshot as newly introduced whitespace errors. The new strict kernel and this
provenance file still use normal Git whitespace checks.
