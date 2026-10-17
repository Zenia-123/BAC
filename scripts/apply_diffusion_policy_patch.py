#!/usr/bin/env python3
"""Restore the pinned diffusion_policy changes without overwriting local edits."""

import argparse
import configparser
import hashlib
from pathlib import Path
import subprocess
import sys


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='Verify the patch is applicable or already present without changing files.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    repo = root / 'diffusion_policy'
    if (repo / 'VENDORED.md').is_file():
        print('diffusion_policy is tracked directly in BAC; no patch is needed. No files changed.')
        return
    patch_dir = root / 'patches' / 'diffusion_policy'
    patch = patch_dir / 'lowdim_fp16.patch'
    manifest = configparser.ConfigParser()
    with (patch_dir / 'manifest.ini').open() as file:
        manifest.read_file(file)
    baseline = manifest['baseline']['commit']
    if sha256(patch) != manifest['baseline']['patch_sha256']:
        raise RuntimeError('Patch checksum mismatch; restore the versioned patch before applying it.')

    top = git(repo, 'rev-parse', '--show-toplevel')
    if top.returncode or Path(top.stdout.strip()).resolve() != repo.resolve():
        raise RuntimeError('Initialize diffusion_policy first: git submodule update --init --recursive')
    current = git(repo, 'rev-parse', 'HEAD').stdout.strip()
    expected = dict(manifest['files'])

    def target_files_match():
        return all((repo / path).is_file() and sha256(repo / path) == digest
                   for path, digest in expected.items())

    # Includes a local safety commit containing exactly these patched files.
    if target_files_match():
        print(f'Patch already present; {len(expected)} file checksums match. No files changed.')
        print(f'Submodule HEAD: {current}; pinned clean baseline: {baseline}')
        return
    if current != baseline:
        raise RuntimeError(f'Expected clean baseline {baseline}, found {current}. '
                           'Existing checkout was left untouched; use a fresh clone to restore.')
    if git(repo, 'status', '--porcelain=v1', '--untracked-files=all').stdout.strip():
        raise RuntimeError('Submodule has local changes and does not match the saved patch. '
                           'No files changed; preserve those changes before restoring in a fresh clone.')
    result = git(repo, 'apply', '--check', '--whitespace=nowarn', str(patch))
    if result.returncode:
        raise RuntimeError('Patch applicability check failed; no files changed:\n' + result.stderr)
    if args.check:
        print(f'Patch applies cleanly to {baseline}; check only, no files changed.')
        return
    result = git(repo, 'apply', '--whitespace=nowarn', str(patch))
    if result.returncode:
        raise RuntimeError('git apply failed:\n' + result.stderr)
    if not target_files_match():
        raise RuntimeError('Post-apply checksum verification failed. Preserve this checkout for inspection.')
    print(f'Applied patch to {baseline}; all {len(expected)} file checksums match.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, configparser.Error, KeyError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
