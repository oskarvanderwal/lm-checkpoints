#!/usr/bin/env python3
"""Evaluate checkpoints using Inspect AI.

Usage:
    python evaluate_inspect.py pythia --size 14m --step 0 1000 --tasks mmlu --output results/
    python evaluate_inspect.py olmo --size 7b --step 1000 --tasks hellaswag --device cuda

Requires: pip install inspect-ai
"""

import argparse
from pathlib import Path

from inspect_ai import eval
from inspect_ai.model import HuggingFaceModel

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


def make_evaluator(tasks, output_dir):
    """Create an evaluation function for use with map()."""
    def evaluate(ckpt):
        ckpt_output = output_dir / ckpt.config["model_name"] / f"step_{ckpt.config['step']}"
        model = HuggingFaceModel(model=ckpt.model, tokenizer=ckpt.tokenizer)
        return eval(tasks, model=model, log_dir=str(ckpt_output))
    return evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints with Inspect AI")
    parser.add_argument("model", choices=list(CHECKPOINT_CLASSES.keys()))
    parser.add_argument("--size", type=str)
    parser.add_argument("--step", type=int, nargs="+")
    parser.add_argument("--seed", type=int, nargs="+")
    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
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

    evaluator = make_evaluator(args.tasks, output_dir)
    results = list(checkpoints.map(evaluator))
    print(f"Completed {len(results)} evaluations")


if __name__ == "__main__":
    main()
