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


def make_evaluator(tasks, batch_size, device):
    """Create an evaluation function for use with map()."""
    def evaluate(ckpt):
        return lm_eval.simple_evaluate(
            model=HFLM(pretrained=ckpt.model, tokenizer=ckpt.tokenizer),
            tasks=tasks,
            batch_size=batch_size,
            device=device,
        )
    return evaluate


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

    evaluator = make_evaluator(args.tasks, args.batch_size, args.device)
    results = checkpoints.map_collect(evaluator)

    for entry in results:
        result_path = output_dir / entry["model_name"] / f"step_{entry['step']}" / "results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(entry, indent=2, default=str))
        print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
