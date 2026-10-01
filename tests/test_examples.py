import importlib.util
from pathlib import Path

from lm_checkpoints import Checkpoint, Checkpoints

_spec = importlib.util.spec_from_file_location(
    "evaluate_with_lm_eval", Path(__file__).parents[1] / "examples" / "evaluate_with_lm_eval.py"
)
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


def test__output_paths_are_unique(tmp_path):
    for run in ["seed0/run", "seed1/run"]:
        (tmp_path / run / "checkpoint-500").mkdir(parents=True)
    ckpts = Checkpoints.from_local(tmp_path / "seed0/run", seed=0) + Checkpoints.from_local(tmp_path / "seed1/run")
    ckpts = ckpts + Checkpoints(
        [
            Checkpoint("org/model", revision="step500", step=500, seed=0),
            Checkpoint("org/model", revision="step500", step=500, seed=1),
            Checkpoint("org/model-seed1", revision="step500", step=500, seed=1),
        ]
    )
    paths = [example.output_path(tmp_path / "out", c) for c in ckpts]
    assert len(set(paths)) == len(paths) == 5
    assert all(p.is_relative_to(tmp_path / "out") for p in paths)
    assert paths[2] == tmp_path / "out" / "org/model/seed_0/step500"
