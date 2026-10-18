"""Shared precision selection and calibration provenance for public entrypoints."""
import hashlib
import json
from pathlib import Path

PRECISIONS = ('fp32', 'decoder_fp16_strict')
ARITHMETIC_VERSION = 'half-tree-lut-v1'


def set_policy_precision(policy, precision='fp32'):
    if precision not in PRECISIONS:
        raise ValueError(f'Unsupported precision {precision}')
    if not hasattr(policy.model, 'set_inference_precision'):
        if precision != 'fp32':
            raise ValueError('Strict precision supports the lowdim Transformer only.')
    else:
        policy.model.set_inference_precision(precision)
    policy.inference_precision = precision
    return policy


def calibration_metadata(checkpoint, precision):
    digest = hashlib.sha256()
    with open(checkpoint, 'rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return {'precision': precision, 'arithmetic_version': ARITHMETIC_VERSION,
            'checkpoint_sha256': digest.hexdigest(), 'weights': 'model'}


def check_calibration(directory, precision, expected=None):
    path = Path(directory) / 'precision.json'
    if not path.exists():
        if precision == 'fp32':
            return  # Legacy original FP32 calibration remains usable.
        raise ValueError(f'Missing calibration provenance: {path}. Recompute calibration.')
    metadata = json.loads(path.read_text())
    if metadata['precision'] != precision or (precision != 'fp32' and metadata.get('arithmetic_version') != ARITHMETIC_VERSION):
        raise ValueError(f'Calibration precision mismatch: requested {precision}, found {metadata}')
    if expected is not None and metadata != expected:
        raise ValueError(f'Calibration metadata mismatch at {path}; use a new directory or force recompute.')
