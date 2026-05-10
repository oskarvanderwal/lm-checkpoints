#!/usr/bin/env python3
"""Evaluate checkpoints using lm-evaluation-harness.

Usage:
    python evaluate_lm_eval.py pythia --size 14m --step 0 1000 --tasks hellaswag --output results/
    python evaluate_lm_eval.py olmo --size 7b --step 1000 2000 --tasks triviaqa --device cuda

Requires: pip install lm-eval
"""

import argparse
import json
from pathlib import Path

import lm_eval
from lm_eval.models.huggingface import HFLM

from lm_checkpoints import (
    PythiaCheckpoints,
    MultiBERTCheckpoints,
    OLMoCheckpoints,
    TriCheckpoints,
    OpenMoECheckpoints,
)

CHECKPOINT_CLASSES = {
    "pythia": PythiaCheckpoints,
    "multiberts": MultiBERTCheckpoints,
    "olmo": OLMoCheckpoints,
    "tri": TriCheckpoints,
    "openmoe": OpenMoECheckpoints,
}


def evaluate_checkpoint(ckpt, tasks, batch_size=16, device="cpu"):
    """Run lm-eval on a single checkpoint."""
    return lm_eval.simple_evaluate(
        model=HFLM(pretrained=ckpt.model, tokenizer=ckpt.tokenizer),
        tasks=tasks,
        batch_size=batch_size,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints with lm-evaluation-harness")
    parser.add_argument("model", choices=list(CHECKPOINT_CLASSES.keys()))
    parser.add_argument("--size", type=str)
    parser.add_argument("--step", type=int, nargs="+")
    parser.add_argument("--seed", type=int, nargs="+")
    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    kwargs = {"device": args.device}
    if args.size:
        kwargs["size"] = args.size
    if args.step:
        kwargs["step"] = args.step
    if args.seed:
        kwargs["seed"] = args.seed

    checkpoints = CHECKPOINT_CLASSES[args.model](**kwargs)
    output_dir = Path(args.output)

    for ckpt in checkpoints:
        result_path = output_dir / ckpt.config["model_name"] / f"step_{ckpt.config['step']}" / "results.json"

        if args.skip_existing and result_path.exists():
            print(f"Skipping {result_path} (exists)")
            continue

        print(f"Evaluating {ckpt.config['model_name']} step {ckpt.config['step']}...")
        results = evaluate_checkpoint(ckpt, args.tasks, args.batch_size, args.device)

        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"Saved to {result_path}")


if __name__ == "__main__":
    main()
