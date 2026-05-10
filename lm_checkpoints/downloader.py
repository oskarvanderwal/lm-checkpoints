"""Download checkpoints from the HuggingFace hub."""

import argparse
from huggingface_hub import snapshot_download

from lm_checkpoints.pythia import PythiaCheckpoints
from lm_checkpoints.multiberts import MultiBERTCheckpoints

CHECKPOINT_REGISTRY = {
    "pythia": PythiaCheckpoints,
    "multiberts": MultiBERTCheckpoints,
}
MODELS_REQUIRING_SIZE = {"pythia"}


def _get_download_info(checkpoints, model_type: str, cfg: dict) -> tuple:
    """Get repo_id and revision for a checkpoint config."""
    if model_type == "pythia":
        return checkpoints.get_model_name(cfg["seed"]), f"step{cfg['step']}"
    elif model_type == "multiberts":
        return checkpoints.get_model_name(cfg["step"], cfg["seed"]), None
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Download checkpoints from the HuggingFace hub.")
    parser.add_argument("checkpoints", type=str, choices=list(CHECKPOINT_REGISTRY.keys()))
    parser.add_argument("--seed", type=int, nargs="+")
    parser.add_argument("--step", type=int, nargs="+")
    parser.add_argument("--size", type=str)
    parser.add_argument("--cache_dir", type=str)
    args = parser.parse_args()

    if args.checkpoints in MODELS_REQUIRING_SIZE and not args.size:
        raise ValueError(f"--size is required for {args.checkpoints}")

    cls = CHECKPOINT_REGISTRY[args.checkpoints]
    kwargs = {}
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    if args.step:
        kwargs["step"] = args.step
    if args.seed:
        kwargs["seed"] = args.seed
    if args.size:
        kwargs["size"] = args.size

    checkpoints = cls(**kwargs)

    for cfg in checkpoints.checkpoints:
        repo_id, revision = _get_download_info(checkpoints, args.checkpoints, cfg)
        print(f"Downloading {repo_id}" + (f" @ {revision}" if revision else ""))
        snapshot_download(repo_id=repo_id, revision=revision, cache_dir=args.cache_dir)
