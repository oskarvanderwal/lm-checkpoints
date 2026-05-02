import argparse
from lm_checkpoints import (
    PythiaCheckpoints,
    MultiBERTCheckpoints,
    OLMoCheckpoints,
    TriCheckpoints,
    OpenMoECheckpoints,
)
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download checkpoints from the HuggingFace hub.")
    parser.add_argument(
        "checkpoints",
        type=str,
        choices=["pythia", "multiberts", "olmo", "tri", "openmoe"],
        help="Checkpoints to download",
    )
    parser.add_argument("--seed", type=int, nargs="+", help="Selection of seeds for the checkpoints. Defaults to all.")
    parser.add_argument("--step", type=int, nargs="+", help="Selection of steps for the checkpoints. Defaults to all.")
    parser.add_argument("--size", type=str, help="Size of the checkpoints model. Required for some models.")
    parser.add_argument("--cache_dir", type=str, help="Custom cache directory.")
    args = parser.parse_args()

    cache_kwargs = {}
    if args.cache_dir:
        cache_kwargs["cache_dir"] = args.cache_dir

    if args.checkpoints == "multiberts":
        checkpoints = MultiBERTCheckpoints(seed=args.seed, step=args.step, **cache_kwargs)
    elif args.checkpoints == "pythia":
        if not args.size:
            raise ValueError("Please provide the model size of the Pythia models, e.g., `--size 70m`.")
        checkpoints = PythiaCheckpoints(size=args.size, seed=args.seed, step=args.step, **cache_kwargs)
    elif args.checkpoints == "olmo":
        if not args.size:
            raise ValueError("Please provide the model size of OLMo models, e.g., `--size 7b`.")
        checkpoints = OLMoCheckpoints(size=args.size, step=args.step, **cache_kwargs)
    elif args.checkpoints == "tri":
        if not args.size:
            raise ValueError("Please provide the model size of Tri models, e.g., `--size 7b`.")
        checkpoints = TriCheckpoints(size=args.size, step=args.step, **cache_kwargs)
    elif args.checkpoints == "openmoe":
        if not args.size:
            raise ValueError("Please provide the model size of OpenMoE models, e.g., `--size 8b`.")
        checkpoints = OpenMoECheckpoints(size=args.size, step=args.step, **cache_kwargs)

    for cfg in checkpoints.checkpoints:
        if args.checkpoints == "pythia":
            repo_id = checkpoints.get_model_name(cfg["seed"])
            revision = f"step{cfg['step']}"
        elif args.checkpoints == "multiberts":
            repo_id = checkpoints.get_model_name(cfg["step"], cfg["seed"])
            revision = None
        elif args.checkpoints == "olmo":
            repo_id = checkpoints.get_model_name()
            revision = checkpoints._get_revision(cfg["step"])
        elif args.checkpoints == "tri":
            repo_id = checkpoints.get_model_name()
            revision = checkpoints._get_revision(cfg["step"])
        elif args.checkpoints == "openmoe":
            repo_id = checkpoints.get_model_name(cfg["step"])
            revision = None

        print(f"Downloading {repo_id}" + (f" @ {revision}" if revision else ""))
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=args.cache_dir,
        )
