# Source provenance and Git workflow

This directory is ordinary source tracked by the BAC repository, not a Git
submodule. Clone BAC normally; no patch application or submodule update is needed.
Edit Python files here and commit them together with BAC changes.

Upstream: https://github.com/real-stanford/diffusion_policy

Original baseline: `5ba07ac6661db573af695b419a7947ecb704690f`.
The import preserves the seven local compatibility, annotation and FP16 changes
previously delivered in `patches/diffusion_policy/lowdim_fp16.patch`.
All 369 files tracked by the original checkout were imported, including assets,
LICENSE and third-party notices. No model checkpoint or dataset is included.

The previous local Git history and pre-import changes are backed up under
`.git/lowdim-full-cy8w51br/` on the original development machine. Historical
commits before this migration still use the old submodule-plus-patch workflow;
the current branch uses direct source management.
