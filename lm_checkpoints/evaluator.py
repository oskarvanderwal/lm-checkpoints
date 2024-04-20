"""Functionality for evaluating the checkpoints on multiple tasks using lm-evaluation-harness.
Borrowed most of the implementation from https://github.com/EleutherAI/lm-evaluation-harness/blob/3196e907fa195b684470a913c7235ed7f08a4383/lm_eval/__main__.py
"""

from importlib.util import find_spec
from pathlib import Path
import json
from lm_checkpoints import AbstractCheckpoints, PythiaCheckpoints, MultiBERTCheckpoints
import numpy as np
from typing import List
import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def evaluate(
    checkpoints: AbstractCheckpoints,
    tasks: List,
    output_dir: str,
    batch_size: int = 16,
    log_samples: bool = False,
    skip_if_exists: bool = True,
    overwrite: bool = False,
    **kwargs,
) -> None:
    """Uses lm-evaluation-harness for evaluating all of the checkpoints on the tasks, and writes the results to disk.
    `kwargs` are passed to `lm_eval.simple_evaluate`.

    Args:
        checkpoints (AbstractCheckpoints): The checkpoints to evaluate.
        tasks (List): List of tasks implemented in lm-evaluation-harness.
        output_dir (str): Directory where the results will be written to.
        batch_size (int, optional): batch size lm-evaluation-harness should use. Defaults to 16.
        log_samples (bool, optional): If True, will also write the model's answers to the individual test items. Defaults to False.
        skip_if_exists (bool, optional): If True, skips evaluating the checkpoints for which the results that already exist on disk. Defaults to True.
        overwrite (bool, optional): If True, overwrites the results on disk. Defaults to False.

    Raises:
        Exception: Raised if lm-evaluation-harness is not installed.
        FileExistsError: Raised if the results files already exists, and both the flags `skip_if_exists` and `overwrite` are False.
    """
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
    if not find_spec("lm_eval"):
        raise Exception("Please install lm_eval through `pip install lm-checkpoints[eval]` or `pip install -e .[eval]`")
    else:
        import lm_eval
        from lm_eval.models.huggingface import HFLM

    device = checkpoints.device
    checkpoints.low_cpu_mem_usage = False

    output_dir = Path(output_dir)

    for ckpt in checkpoints:
        path = (
            output_dir
            / ckpt.config["model_name"]
            / f"step_{ckpt.config['step']}"
            / f"results_{','.join(sorted(tasks))}.json"
        )

        if path.is_file():
            if skip_if_exists:
                continue
            elif not overwrite:
                raise FileExistsError(f"File already exists at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        results = lm_eval.simple_evaluate(
            model=HFLM(pretrained=ckpt.model, tokenizer=ckpt.tokenizer),
            tasks=tasks,
            batch_size=batch_size,
            device=device,
            **kwargs,
            # task_manager=lm_eval.tasks.TaskManager(),
        )

        if results is not None:
            if log_samples:
                samples = results.pop("samples")
            dumped = json.dumps(results, indent=2, default=_handle_non_serializable, ensure_ascii=False)
            path.open("w", encoding="utf-8").write(dumped)

            if log_samples:
                for task_name, config in results["configs"].items():
                    output_samples_file = path.parent / f"samples_{task_name}"
                    samples_dumped = json.dumps(
                        samples[task_name],
                        indent=2,
                        default=_handle_non_serializable,
                        ensure_ascii=False,
                    )
                    output_samples_file.with_suffix(".jsonl").write_text(samples_dumped, encoding="utf-8")


def main():
    # All the logic of argparse goes in this function
    parser = argparse.ArgumentParser(description="Evaluate checkpoints using lm-evaluation-harness.")
    parser.add_argument("checkpoints", type=str, choices=["pythia", "multiberts"], help="Checkpoints to evaluate")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--output", type=str, required=True, help="Path to directory where to store results.")
    parser.add_argument("--seed", type=int, nargs="+", help="Selection of seeds for the checkpoints. Defaults to all.")
    parser.add_argument("--step", type=int, nargs="+", help="Selection of steps for the checkpoints. Defaults to all.")
    parser.add_argument("--size", type=int, help="Size of the checkpoints model. Required for some models.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--tasks", type=str, nargs="+", help="List of tasks to evaluate.")
    parser.add_argument("--log_samples", type=bool, default=False)
    parser.add_argument("--skip_if_exists", type=bool, default=True)
    parser.add_argument("--overwrite", type=bool, default=False)

    args = parser.parse_args()

    if args.checkpoints == "multiberts":
        checkpoints = MultiBERTCheckpoints(seed=args.seed, step=args.step, device=args.device)
    elif args.checkpoints == "pythia":
        if not args.size:
            raise ValueError("Please provide the model size of the Pythia models to evaluate, e.g., `--size 70`.")
        checkpoints = PythiaCheckpoints(size=args.size, seed=args.seed, step=args.step, device=args.device)

    evaluate(
        checkpoints,
        tasks=args.tasks,
        output_dir=args.output,
        log_samples=args.log_samples,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        skip_if_exists=args.skip_if_exists,
    )
