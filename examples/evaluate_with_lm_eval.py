"""Evaluate checkpoints with lm-evaluation-harness (`pip install lm-eval`), resuming where you left off.

Writes results.json (with the checkpoint's metadata under "lm_checkpoints") and, with log_samples=True,
samples_<task>.json to a directory per checkpoint; see `output_path`. Adapt freely: this is a recipe, not part of
the library.
"""

import json
from pathlib import Path

from lm_checkpoints import Checkpoint, Checkpoints, PythiaCheckpoints


def output_path(output_dir, ckpt: Checkpoint) -> Path:
    """A unique directory per checkpoint:
    - hub: <output_dir>/<org>/<model>[/seed_<seed>]/<revision>
    - local: <output_dir>/local/<absolute path of the checkpoint directory>[/seed_<seed>]
    """
    if ckpt.is_local:
        path = Path(ckpt.repo_id).resolve()
        out = Path(output_dir) / "local" / path.relative_to(path.anchor)
    else:
        out = Path(output_dir) / ckpt.repo_id
    if ckpt.seed is not None:
        out = out / f"seed_{ckpt.seed}"
    if not ckpt.is_local:
        out = out / (ckpt.revision or "main")
    return out


def evaluate(checkpoints: Checkpoints, tasks, output_dir, log_samples=False, model_args="", **kwargs):
    """`model_args` are appended to lm-eval's model_args (e.g. "dtype=float16"), `kwargs` go to simple_evaluate."""
    import lm_eval

    for ckpt in checkpoints:
        out = output_path(output_dir, ckpt)
        if (out / "results.json").exists():
            continue  # already done: skipped without downloading the checkpoint

        args = f"pretrained={ckpt.repo_id}" + (f",revision={ckpt.revision}" if ckpt.revision else "")
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=",".join(filter(None, [args, model_args])),
            tasks=tasks,
            log_samples=log_samples,
            **kwargs,
        )
        if results is None:  # non-zero ranks when running distributed
            continue

        out.mkdir(parents=True, exist_ok=True)
        samples = results.pop("samples", {})
        results["lm_checkpoints"] = ckpt.config
        (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
        for task, task_samples in samples.items():
            (out / f"samples_{task}.json").write_text(json.dumps(task_samples, indent=2, default=str))


if __name__ == "__main__":
    evaluate(
        PythiaCheckpoints(size="14m", seed=[0], step=[0, 1000, 143000], clean_cache=True),
        tasks=["lambada_openai"],
        output_dir="results",
        batch_size=16,
        device="cuda",
        limit=10,  # for testing; remove for real runs
    )
