"""Evaluate checkpoints on multiple tasks using lm-evaluation-harness."""

from importlib.util import find_spec
from pathlib import Path
import json
import os
import argparse
from typing import List

import numpy as np

from lm_checkpoints.checkpoints import AbstractCheckpoints
from lm_checkpoints.pythia import PythiaCheckpoints
from lm_checkpoints.multiberts import MultiBERTCheckpoints
from lm_checkpoints.olmo import OLMoCheckpoints
from lm_checkpoints.tri import TriCheckpoints
from lm_checkpoints.openmoe import OpenMoECheckpoints

CHECKPOINT_REGISTRY = {
    "pythia": PythiaCheckpoints,
    "multiberts": MultiBERTCheckpoints,
    "olmo": OLMoCheckpoints,
    "tri": TriCheckpoints,
    "openmoe": OpenMoECheckpoints,
}
MODELS_REQUIRING_SIZE = {"pythia", "olmo", "tri", "openmoe"}

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _handle_non_serializable(o):
    if isinstance(o, (np.int64, np.int32)):
        return int(o)
    elif isinstance(o, set):
        return list(o)
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
    """Evaluate checkpoints using lm-evaluation-harness."""
    if not find_spec("lm_eval"):
        raise ImportError(
            'Please install lm_eval: pip install "lm-checkpoints[eval]"'
        )

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
        )

        if results is not None:
            samples = results.pop("samples") if log_samples else None
            dumped = json.dumps(results, indent=2, default=_handle_non_serializable, ensure_ascii=False)
            path.write_text(dumped, encoding="utf-8")

            if log_samples and samples:
                for task_name in results["configs"]:
                    output_file = path.parent / f"samples_{task_name}.json"
                    samples_dumped = json.dumps(
                        samples[task_name],
                        indent=2,
                        default=_handle_non_serializable,
                        ensure_ascii=False,
                    )
                    output_file.write_text(samples_dumped, encoding="utf-8")


def _create_checkpoints(args):
    """Create checkpoint instance from CLI args using registry."""
    cls = CHECKPOINT_REGISTRY[args.checkpoints]

    if args.checkpoints in MODELS_REQUIRING_SIZE and not args.size:
        raise ValueError(f"--size is required for {args.checkpoints}")

    kwargs = {
        "device": args.device,
        "cache_policy": args.cache_policy,
        "cache_dir": args.cache_dir,
    }
    if args.max_cache_size_gb:
        kwargs["max_cache_size_gb"] = args.max_cache_size_gb
    if args.step:
        kwargs["step"] = args.step
    if args.seed:
        kwargs["seed"] = args.seed
    if args.size:
        kwargs["size"] = args.size

    return cls(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints using lm-evaluation-harness.")
    parser.add_argument("checkpoints", type=str, choices=list(CHECKPOINT_REGISTRY.keys()))
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, nargs="+")
    parser.add_argument("--step", type=int, nargs="+")
    parser.add_argument("--size", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--log_samples", action="store_true")
    parser.add_argument("--skip_if_exists", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache_policy", type=str, choices=["keep", "previous", "bounded"], default="keep")
    parser.add_argument("--max_cache_size_gb", type=float)
    parser.add_argument("--cache_dir", type=str)

    args = parser.parse_args()
    checkpoints = _create_checkpoints(args)

    evaluate(
        checkpoints,
        tasks=args.tasks,
        output_dir=args.output,
        log_samples=args.log_samples,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        skip_if_exists=args.skip_if_exists,
    )
