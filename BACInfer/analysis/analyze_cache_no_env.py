"""Collect lowdim block features and validate ACS without an environment runner.

Zero observations are a smoke test. Supply a saved observation dictionary with
--obs_file for task data; this script never measures task success rates.
"""

import json
import logging
import pickle
import re
import time
from copy import deepcopy
from pathlib import Path

import click
import dill
import hydra
import numpy as np
import torch
import torch.nn.functional as F

from BACInfer.analysis.optimal_cache_scheduler import OptimalCacheScheduler
from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy


logger = logging.getLogger(__name__)
BLOCK_PATTERN = re.compile(r"decoder\.layers\.\d+\.dropout[123]")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict(policy, obs_dict, seed):
    torch.manual_seed(seed)
    if policy.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    policy.reset()
    with torch.no_grad():
        return policy.predict_action(obs_dict)


def capture(policy, obs_dict, seed, save_tensors=False):
    """Dropout hooks run only when the corresponding block is recomputed."""
    counts = {}
    activations = {}
    timesteps = []
    hooks = []

    def record_timestep(module, inputs):
        timesteps.append(int(torch.as_tensor(inputs[1]).reshape(-1)[0].item()))

    def record_block(name):
        def hook(module, inputs, output):
            counts[name] += 1
            if save_tensors:
                activations[name].append(output.detach().cpu().clone())
        return hook

    for name, module in policy.model.named_modules():
        if BLOCK_PATTERN.fullmatch(name):
            counts[name] = 0
            activations[name] = []
            hooks.append(module.register_forward_hook(record_block(name)))
    if not counts:
        raise ValueError("No decoder SA/CA/FFN output modules found")
    hooks.append(policy.model.register_forward_pre_hook(record_timestep))
    try:
        output = predict(policy, obs_dict, seed)
    finally:
        for hook in hooks:
            hook.remove()
    return output, counts, activations, timesteps


def output_error(reference, output):
    errors = {}
    for key in ("action", "action_pred"):
        delta = (output[key] - reference[key]).detach().float()
        errors[key] = {
            "mae": float(delta.abs().mean().item()),
            "rmse": float(delta.square().mean().sqrt().item()),
            "max_abs": float(delta.abs().max().item()),
        }
    return errors


@click.command()
@click.option("-c", "--checkpoint", required=True, type=click.Path(exists=True))
@click.option("-o", "--output_dir", required=True, type=click.Path())
@click.option("-d", "--device", default="cuda:0", show_default=True)
@click.option("--obs_file", type=click.Path(exists=True),
              help="Torch file containing raw {'obs': tensor[B, To, Do]}.")
@click.option("--policy_weights", type=click.Choice(["model", "ema"]),
              default="model", show_default=True,
              help="model matches the existing benchmark; ema matches run_policy.")
@click.option("--num_caches", default="5,20", show_default=True)
@click.option("--cache_threshold", default=5, type=click.IntRange(min=1))
@click.option("--seed", default=11, type=int)
@click.option("--trials", default=3, type=click.IntRange(min=1), show_default=True)
def main(checkpoint, output_dir, device, obs_file, policy_weights, num_caches,
         cache_threshold, seed, trials):
    """Run Original -> Threshold -> ACS on identical lowdim observations/noise."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(device)
    output_dir = Path(output_dir)
    original_dir = output_dir / "original"
    original_dir.mkdir(parents=True, exist_ok=True)
    try:
        cache_counts = sorted(set(int(x.strip()) for x in num_caches.split(",")))
    except ValueError as exc:
        raise click.BadParameter("Expected comma-separated integer counts",
                                 param_hint="--num_caches") from exc

    with open(checkpoint, "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model if policy_weights == "model" else workspace.ema_model
    if policy is None:
        raise click.BadParameter("Checkpoint has no EMA policy",
                                 param_hint="--policy_weights")
    policy.to(device).eval()
    if not hasattr(policy, "obs_dim") or hasattr(policy, "obs_encoder"):
        raise click.ClickException("This entry point supports lowdim policies only")
    if not all(layer.norm_first for layer in policy.model.decoder.layers):
        raise click.ClickException("Cache wrapper requires pre-norm decoder layers")
    n_steps = policy.num_inference_steps
    if not cache_counts or any(n < 1 or n > n_steps for n in cache_counts):
        raise click.BadParameter(f"Counts must be in [1, {n_steps}]",
                                 param_hint="--num_caches")
    # The current cache wrapper adds index 99 to every nonempty ACS schedule.
    if n_steps != 100:
        raise click.ClickException("Current optimal wrapper assumes 100 steps (index 99)")

    if obs_file:
        raw_obs = torch.load(obs_file, map_location="cpu", weights_only=True)
        if not isinstance(raw_obs, dict) or set(raw_obs) != {"obs"}:
            raise click.BadParameter("Expected a dictionary with only 'obs'",
                                     param_hint="--obs_file")
        obs = torch.as_tensor(raw_obs["obs"], dtype=torch.float32, device=device)
        source = "saved_observations"
    else:
        obs = torch.zeros((1, policy.n_obs_steps, policy.obs_dim), device=device)
        source = "synthetic_zeros_smoke_test"
        logger.warning("Zero observations: results validate the pipeline, not task quality")
    if obs.ndim != 3 or obs.shape[0] < 1 or tuple(obs.shape[1:]) != (policy.n_obs_steps, policy.obs_dim):
        raise click.ClickException(f"Expected obs[B, {policy.n_obs_steps}, {policy.obs_dim}]")
    if not torch.isfinite(obs).all():
        raise click.ClickException("Observations must be finite")
    obs_dict = {"obs": obs}

    logger.info("Collecting Original features: %d denoising steps", n_steps)
    _, counts, activations, timesteps = capture(policy, obs_dict, seed, save_tensors=True)
    if len(timesteps) != n_steps or any(n != n_steps for n in counts.values()):
        raise RuntimeError(f"Incomplete activation capture: {counts}")
    with open(original_dir / "activations.pkl", "wb") as f:
        pickle.dump(activations, f)

    similarities = {}
    module_summary = {}
    for name, values in activations.items():
        flat = torch.stack(values).float().flatten(start_dim=1)
        normalized = F.normalize(flat, dim=1, eps=1e-8)
        cosine = (normalized @ normalized.T).numpy()
        if not np.isfinite(cosine).all():
            raise RuntimeError(f"Nonfinite similarities for {name}")
        similarities[name] = {"cosine": cosine}
        module_summary[name] = {
            "shape": list(values[0].shape), "dtype": str(values[0].dtype),
            "adjacent_cosine_mean": float(np.diag(cosine, k=1).mean()),
        }
        for n in cache_counts:
            scheduler = OptimalCacheScheduler(similarity_matrix=cosine, metric="cosine")
            steps = scheduler.compute_optimal_steps(n)
            if len(steps) != n or steps[0] != 0:
                raise RuntimeError(f"Invalid ACS schedule for {name}: {steps}")
            steps_path = output_dir / "optimal_steps" / "cosine" / name
            scheduler.save_optimal_steps(str(steps_path / f"optimal_steps_{name}_{n}_cosine.pkl"))
    with open(original_dir / "similarity_matrices.pkl", "wb") as f:
        pickle.dump(similarities, f)

    references = [predict(policy, obs_dict, seed + i) for i in range(trials)]
    modes = [("original", None, None), ("threshold1_control", "threshold", 1),
             (f"threshold{cache_threshold}", "threshold", cache_threshold)]
    modes += [(f"optimal{n}", "optimal", n) for n in cache_counts]
    results = {}
    for label, mode, budget in modes:
        candidate = deepcopy(policy)
        if mode == "threshold":
            FastDiffusionPolicy.apply_cache(candidate, cache_mode=mode, cache_threshold=budget)
        elif mode == "optimal":
            FastDiffusionPolicy.apply_cache(
                candidate, cache_mode=mode, num_caches=budget, metric="cosine",
                optimal_steps_dir=str(output_dir / "optimal_steps" / "cosine"),
                num_bu_blocks=0)
            if len(candidate._cache["block_steps"]) != len(counts):
                raise RuntimeError("ACS schedules did not load for all blocks")
        validation = []
        for i, reference in enumerate(references):
            output, executed, _, observed_timesteps = capture(candidate, obs_dict, seed + i)
            if observed_timesteps != timesteps:
                raise RuntimeError("Denoising timesteps changed between modes")
            if label in ("original", "threshold1_control"):
                for key in ("action", "action_pred"):
                    torch.testing.assert_close(output[key], reference[key], rtol=1e-4, atol=1e-5)
            if mode == "threshold":
                expected = len(range(0, n_steps, budget))
                if any(n != expected for n in executed.values()):
                    raise RuntimeError(f"Unexpected threshold block counts: {executed}")
            elif mode == "optimal":
                for name, actual in executed.items():
                    layer, dropout = name.rsplit(".", 1)
                    suffix = {"dropout1": "sa_block", "dropout2": "mha_block", "dropout3": "ff_block"}[dropout]
                    expected = len(candidate._cache["steps_lookup"][f"{layer}_{suffix}"])
                    if actual != expected:
                        raise RuntimeError(f"ACS update mismatch for {name}: {actual} != {expected}")
            validation.append({"seed": seed + i, "errors": output_error(reference, output),
                               "block_compute_counts": executed})
        # Time without hooks or tensor copies; warm up each mode first.
        predict(candidate, obs_dict, seed)
        durations = []
        for i in range(trials):
            synchronize(device)
            start = time.perf_counter()
            predict(candidate, obs_dict, seed + i)
            synchronize(device)
            durations.append(time.perf_counter() - start)
        results[label] = {"avg_time_s": float(np.mean(durations)),
                          "durations_s": durations, "validation": validation}
        results[label]["speedup"] = results["original"]["avg_time_s"] / results[label]["avg_time_s"]
        logger.info("%s: %.4f s, %.2fx, action MAE %.6g", label,
                    results[label]["avg_time_s"], results[label]["speedup"],
                    validation[0]["errors"]["action"]["mae"])
        del candidate

    summary = {
        "checkpoint": str(Path(checkpoint).resolve()), "policy_weights": policy_weights,
        "device": str(device), "observation_source": source,
        "obs_shape": list(obs.shape), "calibration_seed": seed,
        "denoising_timesteps": timesteps, "num_bu_blocks": 0,
        "mean_score": None, "modules": module_summary, "results": results,
        "notes": ["ACS calibration uses one inference, validation uses the listed seeds.",
                  "Current wrapper additionally updates index 99 in optimal mode.",
                  "Block counts are measured; they are not complete FLOPs or energy estimates.",
                  "Timings are short local measurements without collection hooks."]}
    summary_path = output_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
