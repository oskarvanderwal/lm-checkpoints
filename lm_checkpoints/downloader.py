import argparse
from lm_checkpoints import PythiaCheckpoints, MultiBERTCheckpoints
from huggingface_hub import snapshot_download

def main():
    # All the logic of argparse goes in this function
    parser = argparse.ArgumentParser(description="Download checkpoints from the HuggingFace hub.")
    parser.add_argument("checkpoints", type=str, choices=["pythia", "multiberts"], help="Checkpoints to download")
    parser.add_argument("--seed", type=int, nargs="+", help="Selection of seeds for the checkpoints. Defaults to all.")
    parser.add_argument("--step", type=int, nargs="+", help="Selection of steps for the checkpoints. Defaults to all.")
    parser.add_argument("--size", type=int, help="Size of the checkpoints model. Required for some models.")
    args = parser.parse_args()

    if args.checkpoints == "multiberts":
        checkpoints = MultiBERTCheckpoints(
            seed=args.seed, step=args.step, device=args.device, clean_cache=args.clean_cache
        )
    elif args.checkpoints == "pythia":
        if not args.size:
            raise ValueError("Please provide the model size of the Pythia models to evaluate, e.g., `--size 70`.")
        checkpoints = PythiaCheckpoints(
            size=args.size, seed=args.seed, step=args.step, device=args.device, clean_cache=args.clean_cache
        )

    for ckpt in checkpoints:
        repo_id = ckpt.config["model_name"]
        revision = ckpt.config.get("revision", None)
        if revision:
            snapshot_download(repo_id=repo_id, revision=revision)
        else:
            snapshot_download(repo_id=repo_id)