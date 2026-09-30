"""Offline tests: nothing is downloaded, the HF hub and model loading are mocked where needed."""

from types import SimpleNamespace

import pytest

import lm_checkpoints.checkpoints as checkpoints_module
from lm_checkpoints import Checkpoint, Checkpoints, MultiBERTCheckpoints, PythiaCheckpoints


def test__pythia_sizes():
    assert len(PythiaCheckpoints(step=[143000], seed=[0])) == 1
    assert len(PythiaCheckpoints(step=[0, 143000], seed=[0, 1, 2, 3, 4])) == 5 * 2
    assert len(PythiaCheckpoints(size="1b")) == len(PythiaCheckpoints.all_steps)
    assert len(PythiaCheckpoints()) == 10 * 154


def test__pythia_names():
    ckpts = PythiaCheckpoints(size="160m", step=[1000], seed=[0, 3])
    assert [(c.repo_id, c.revision) for c in ckpts] == [
        ("EleutherAI/pythia-160m", "step1000"),
        ("EleutherAI/pythia-160m-seed3", "step1000"),
    ]
    assert PythiaCheckpoints(size="70m", deduped=True, step=[0])[0].repo_id == "EleutherAI/pythia-70m-deduped"


def test__pythia_invalid():
    with pytest.raises(ValueError):
        PythiaCheckpoints(size="100m")
    with pytest.raises(ValueError):
        PythiaCheckpoints(size="1b", seed=[1])
    with pytest.raises(ValueError):
        PythiaCheckpoints(step=[3])


def test__multiberts():
    assert len(MultiBERTCheckpoints(step=[0, 2000], seed=[0, 1, 2, 3, 4])) == 5 * 2
    ckpt = MultiBERTCheckpoints(step=[20], seed=[4])[0]
    assert ckpt.repo_id == "google/multiberts-seed_4-step_20k"
    assert ckpt.model_class == "AutoModelForMaskedLM"
    assert len(MultiBERTCheckpoints.final_checkpoints()) == 5


def test__split_is_partition():
    ckpts = PythiaCheckpoints(step=[0, 1, 2], seed=[0, 1], device="cuda", clean_cache=True)
    for n in range(1, 9):
        chunks = ckpts.split(n)
        assert len(chunks) == min(n, len(ckpts))
        flat = [c for chunk in chunks for c in chunk.checkpoints]
        assert flat == ckpts.checkpoints
        assert max(map(len, chunks)) - min(map(len, chunks)) <= 1
        # Settings are preserved
        assert all(isinstance(chunk, PythiaCheckpoints) and chunk.clean_cache for chunk in chunks)
        assert all(c.device == "cuda" for c in flat)


def test__slicing_filter_and_final():
    ckpts = PythiaCheckpoints(step=[0, 1, 143000], seed=[0, 1])
    assert len(ckpts[1:3]) == 2
    assert ckpts[-1].step == 143000
    assert len(ckpts.filter(step=0)) == 2
    assert [c.step for c in ckpts.final()] == [143000, 143000]
    assert ckpts.seeds == [0, 1] and ckpts.steps == [0, 1, 143000]
    assert len(ckpts.filter(seed=0) + ckpts.filter(seed=1)) == len(ckpts)


def test__load_kwargs_are_passed(monkeypatch):
    calls = []

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            calls.append(("to", device))
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append((name, kwargs))
            return FakeModel()

    ckpts = PythiaCheckpoints(step=[8], seed=[0], device="cuda:1", model_class=FakeAutoModel, torch_dtype="bf16")
    ckpt = ckpts[0]
    assert ckpt.model is ckpt.model  # loaded once
    assert calls == [
        ("EleutherAI/pythia-14m", {"torch_dtype": "bf16", "revision": "step8"}),
        ("to", "cuda:1"),
    ]
    # The collection does not keep loaded models alive
    assert ckpts.checkpoints[0]._model is None


def test__from_hub(monkeypatch):
    branches = ["main", "stage1-step0-tokens0B", "stage1-step2000-tokens9B", "stage1-step1000-tokens4B"]
    monkeypatch.setattr(
        checkpoints_module,
        "list_repo_refs",
        lambda repo_id, token=None: SimpleNamespace(branches=[SimpleNamespace(name=b) for b in branches]),
    )
    ckpts = Checkpoints.from_hub("org/model", pattern=r"stage1-step(?P<step>\d+)-tokens(?P<tokens>\d+)B", seed=3)
    assert ckpts.steps == [0, 1000, 2000]
    assert ckpts[1].revision == "stage1-step1000-tokens4B"
    assert ckpts[1].meta == {"tokens": "4"}
    assert ckpts.seeds == [3]
    with pytest.raises(ValueError):
        Checkpoints.from_hub("org/model", pattern=r"nope(\d+)")


def test__from_local(tmp_path):
    for step in [1000, 500, 1500]:
        (tmp_path / "my-run" / f"checkpoint-{step}").mkdir(parents=True)
    (tmp_path / "my-run" / "runs").mkdir()
    ckpts = Checkpoints.from_local(tmp_path / "my-run")
    assert ckpts.steps == [500, 1000, 1500]
    assert ckpts[0].is_local and ckpts[0].name == "my-run" and ckpts[0].is_cached()


def test__clean_cache(monkeypatch):
    deleted = []
    cached_before = {"step0"}
    monkeypatch.setattr(Checkpoint, "is_cached", lambda self: self.revision in cached_before)
    monkeypatch.setattr(Checkpoint, "delete_from_cache", lambda self: deleted.append(self.revision))

    seen = []
    for ckpt in PythiaCheckpoints(step=[0, 1, 2], seed=[0], clean_cache=True):
        seen.append(ckpt.revision)
        # The previous checkpoint is deleted before the next one is handed out
        assert deleted == [r for r in seen[:-1] if r not in cached_before]
    # Also the last one is deleted, but never what was cached before
    assert deleted == ["step1", "step2"]


def test__clean_cache_on_break(monkeypatch):
    deleted = []
    monkeypatch.setattr(Checkpoint, "is_cached", lambda self: False)
    monkeypatch.setattr(Checkpoint, "delete_from_cache", lambda self: deleted.append(self.revision))
    it = iter(PythiaCheckpoints(step=[0, 1], seed=[0], clean_cache=True))
    next(it)
    it.close()
    assert deleted == ["step0"]
