"""Tests that download (small) models from the HF hub. Run with `pytest --run-network`."""

import pytest

from lm_checkpoints import Checkpoints, MultiBERTCheckpoints, PythiaCheckpoints

pytestmark = pytest.mark.network


def _device_type(device):
    return device.split(":")[0]


@pytest.mark.parametrize("device", ["cpu"])
def test__pythia_load(device):
    ckpt = PythiaCheckpoints(step=[0], seed=[0], device=device)[0]
    assert ckpt.model.device.type == _device_type(device)
    assert ckpt.tokenizer("hello")["input_ids"]
    assert ckpt.commit_hash is not None


def test__multibert_load():
    ckpt = MultiBERTCheckpoints(step=[0], seed=[0])[0]
    assert ckpt.model.device.type == "cpu"


def test__from_hub_pythia():
    ckpts = Checkpoints.from_hub("EleutherAI/pythia-14m")
    assert ckpts.steps == PythiaCheckpoints.all_steps
