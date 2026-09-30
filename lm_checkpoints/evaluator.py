"""Functionality for evaluating the checkpoints on multiple tasks using lm-evaluation-harness.
Borrowed most of the implementation from https://github.com/EleutherAI/lm-evaluation-harness/blob/3196e907fa195b684470a913c7235ed7f08a4383/lm_eval/__main__.py
"""

import argparse
import json
import os
from importlib.util import find_spec
from pathlib import Path
from typing import List

from .checkpoints import Checkpoint, Checkpoints
from .multiberts import MultiBERTCheckpoints
from .pythia import PythiaCheckpoints


def _handle_non_serializable(o):
    if isinstance(o, set):
        return list(o)
    try:
        # numpy/torch scalars
        return o.item()
    except (AttributeError, ValueError):
        return str(o)


def results_dir(output_dir, ckpt: Checkpoint) -> Path:
    """Directory where the results of a checkpoint are written to: <output_dir>/<model name>/step_<step>."""
    step = f"step_{ckpt.step}" if ckpt.step is not None else (ckpt.revision or "main")
    return Path(output_dir) / ckpt.name / step


def evaluate(
    checkpoints: Checkpoints,
    tasks: List[str],
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
        checkpoints (Checkpoints): The checkpoints to evaluate.
        tasks (List): List of tasks implemented in lm-evaluation-harness.
        output_dir (str): Directory where the results will be written to.
        batch_size (int, optional): batch size lm-evaluation-harness should use. Defaults to 16.
        log_samples (bool, optional): If True, will also write the model's answers to the individual test items.
            Defaults to False.
        skip_if_exists (bool, optional): If True, skips evaluating the checkpoints for which the results already exist
            on disk (without downloading or loading them). Defaults to True.
        overwrite (bool, optional): If True, overwrites the results on disk. Defaults to False.

    Raises:
        ImportError: Raised if lm-evaluation-harness is not installed.
        FileExistsError: Raised if the results files already exists, and both the flags `skip_if_exists` and
            `overwrite` are False.
    """
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
    if not find_spec("lm_eval"):
        raise ImportError(
            'Please install lm_eval through `pip install "lm-checkpoints[eval]"` or `pip install -e .[eval]`'
        )
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not tasks:
        raise ValueError("Please provide at least one task.")

    for ckpt in checkpoints:
        path = results_dir(output_dir, ckpt) / f"results_{','.join(sorted(tasks))}.json"

        # Checked before accessing ckpt.model, so nothing is downloaded or loaded for skipped checkpoints
        if path.is_file():
            if skip_if_exists:
                continue
            elif not overwrite:
                raise FileExistsError(f"File already exists at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        # batch_size must be given to HFLM: simple_evaluate ignores it (and device) for a pre-initialized model
        lm = HFLM(pretrained=ckpt.model, tokenizer=ckpt.tokenizer, batch_size=batch_size)
        results = lm_eval.simple_evaluate(model=lm, tasks=tasks, log_samples=log_samples, **kwargs)
        ckpt.unload()
        if results is None:
            continue

        samples = results.pop("samples", None)
        results["lm_checkpoints"] = ckpt.config
        path.write_text(
            json.dumps(results, indent=2, default=_handle_non_serializable, ensure_ascii=False), encoding="utf-8"
        )

        if log_samples and samples:
            for task_name, task_samples in samples.items():
                (path.parent / f"samples_{task_name}.json").write_text(
                    json.dumps(task_samples, indent=2, default=_handle_non_serializable, ensure_ascii=False),
                    encoding="utf-8",
                )


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints using lm-evaluation-harness.")
    parser.add_argument(
        "checkpoints",
        type=str,
        choices=["pythia", "multiberts", "hub", "local"],
        help="Checkpoints to evaluate. Use `hub` with --repo_id or `local` with --path for other models.",
    )
    parser.add_argument("--repo_id", type=str, help="HF hub repository with checkpoints as branches (for `hub`).")
    parser.add_argument("--path", type=str, help="Directory with checkpoint subdirectories (for `local`).")
    parser.add_argument("--pattern", type=str, help="Regex matching the checkpoint branches/directories.")
    parser.add_argument("--device", type=str, default=None, help="E.g., cpu, cuda, cuda:1 or mps.")
    parser.add_argument("--torch_dtype", type=str, default=None, help="E.g., float16, bfloat16 or float32.")
    parser.add_argument("--output", type=str, required=True, help="Path to directory where to store results.")
    parser.add_argument("--seed", type=int, nargs="+", help="Selection of seeds for the checkpoints. Defaults to all.")
    parser.add_argument("--step", type=int, nargs="+", help="Selection of steps for the checkpoints. Defaults to all.")
    parser.add_argument("--size", type=str, help="Size of the model. Required for Pythia, e.g., `70m`.")
    parser.add_argument("--deduped", action="store_true", help="Use the deduped Pythia models.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--tasks", type=str, nargs="+", required=True, help="List of tasks to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples per task (for testing).")
    parser.add_argument("--log_samples", action="store_true")
    parser.add_argument("--skip_if_exists", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean_cache", action="store_true")
    args = parser.parse_args()

    kwargs = {"device": args.device, "clean_cache": args.clean_cache}
    if args.torch_dtype:
        import torch

        kwargs["torch_dtype"] = args.torch_dtype if args.torch_dtype == "auto" else getattr(torch, args.torch_dtype)
    pattern = {"pattern": args.pattern} if args.pattern else {}

    if args.checkpoints == "multiberts":
        checkpoints = MultiBERTCheckpoints(seed=args.seed, step=args.step, **kwargs)
    elif args.checkpoints == "pythia":
        if not args.size:
            parser.error("Please provide the size of the Pythia models to evaluate, e.g., `--size 70m`.")
        checkpoints = PythiaCheckpoints(size=args.size, seed=args.seed, step=args.step, deduped=args.deduped, **kwargs)
    else:
        if args.checkpoints == "hub":
            if not args.repo_id:
                parser.error("Please provide --repo_id.")
            checkpoints = Checkpoints.from_hub(args.repo_id, **pattern, **kwargs)
        else:
            if not args.path:
                parser.error("Please provide --path.")
            checkpoints = Checkpoints.from_local(args.path, **pattern, **kwargs)
        checkpoints = checkpoints.filter(step=args.step, seed=args.seed)

    evaluate(
        checkpoints,
        tasks=args.tasks,
        output_dir=args.output,
        log_samples=args.log_samples,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        skip_if_exists=args.skip_if_exists,
        limit=args.limit,
    )
