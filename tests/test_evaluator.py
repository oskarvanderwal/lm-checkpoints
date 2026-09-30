"""Tests the evaluator with a fake lm_eval module, so neither lm_eval nor any models are needed."""

import json
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

from lm_checkpoints import Checkpoint, PythiaCheckpoints, evaluate


@pytest.fixture
def fake_lm_eval(monkeypatch):
    calls = {"hflm": [], "simple_evaluate": []}

    class HFLM:
        def __init__(self, **kwargs):
            calls["hflm"].append(kwargs)

    def simple_evaluate(model, tasks, log_samples=True, **kwargs):
        calls["simple_evaluate"].append({"tasks": tasks, "log_samples": log_samples, **kwargs})
        results = {"results": {t: {"acc": 0.5} for t in tasks}}
        if log_samples:
            results["samples"] = {t: [{"doc_id": 0}] for t in tasks}
        return results

    lm_eval = ModuleType("lm_eval")
    lm_eval.__spec__ = ModuleSpec("lm_eval", None)
    lm_eval.simple_evaluate = simple_evaluate
    models, huggingface = ModuleType("lm_eval.models"), ModuleType("lm_eval.models.huggingface")
    huggingface.HFLM = HFLM
    for name, module in [("lm_eval", lm_eval), ("lm_eval.models", models), ("lm_eval.models.huggingface", huggingface)]:
        monkeypatch.setitem(sys.modules, name, module)

    loaded = []
    monkeypatch.setattr(Checkpoint, "load_model", lambda self: loaded.append(self.revision) or "model")
    monkeypatch.setattr(Checkpoint, "load_tokenizer", lambda self: "tokenizer")
    monkeypatch.setattr(Checkpoint, "commit_hash", property(lambda self: "abc"))
    calls["loaded"] = loaded
    return calls


def test__evaluate(fake_lm_eval, tmp_path):
    ckpts = PythiaCheckpoints(step=[0, 1], seed=[0])
    evaluate(ckpts, tasks=["b", "a"], output_dir=tmp_path, batch_size=8, limit=2)

    assert [kw["batch_size"] for kw in fake_lm_eval["hflm"]] == [8, 8]
    assert all(not kw["log_samples"] and kw["limit"] == 2 for kw in fake_lm_eval["simple_evaluate"])
    path = tmp_path / "EleutherAI/pythia-14m/step_1/results_a,b.json"
    results = json.loads(path.read_text())
    assert "samples" not in results
    assert results["lm_checkpoints"]["revision"] == "step1"
    assert results["lm_checkpoints"]["commit_hash"] == "abc"
    assert not list(path.parent.glob("samples_*"))

    # Existing results are skipped without loading the model
    evaluate(ckpts, tasks=["a", "b"], output_dir=tmp_path)
    assert fake_lm_eval["loaded"] == ["step0", "step1"]
    with pytest.raises(FileExistsError):
        evaluate(ckpts, tasks=["a", "b"], output_dir=tmp_path, skip_if_exists=False)


def test__evaluate_log_samples(fake_lm_eval, tmp_path):
    evaluate(PythiaCheckpoints(step=[0], seed=[0]), tasks=["a"], output_dir=tmp_path, log_samples=True)
    step_dir = tmp_path / "EleutherAI/pythia-14m/step_0"
    assert "samples" not in json.loads((step_dir / "results_a.json").read_text())
    assert json.loads((step_dir / "samples_a.json").read_text()) == [{"doc_id": 0}]
