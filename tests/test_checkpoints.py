from lm_checkpoints import PythiaCheckpoints, MultiBERTCheckpoints
from lm_checkpoints.testing import multi_device
import torch


@multi_device
def test__pythia_multi_device(device: str):
    ckpt = PythiaCheckpoints(step=[143000], seed=[0], device=device)[0]
    assert ckpt.model.device.type == device


@multi_device
def test__multibert_multi_device(device: str):
    ckpt = MultiBERTCheckpoints(step=[2000], seed=[0], device=device)[0]
    assert ckpt.model.device.type == device


def test__pythia_sizes():
    pythia_ckpts = PythiaCheckpoints(step=[143000], seed=[0])
    assert len(pythia_ckpts) == 1

    pythia_ckpts = PythiaCheckpoints(step=[0, 143000], seed=[0, 1, 2, 3, 4])
    assert len(pythia_ckpts) == 5 * 2


def test__pythia_splits():
    pythia_ckpts = PythiaCheckpoints(step=[143000], seed=[0, 1])
    ckpts_splits = pythia_ckpts.split(2)
    assert len(ckpts_splits) == 2
    assert len(ckpts_splits[0]) == 1


def test__multiberts_sizes():
    berts_ckpts = MultiBERTCheckpoints(step=[0], seed=[0])
    assert len(berts_ckpts) == 1

    berts_ckpts = MultiBERTCheckpoints(step=[0, 2000], seed=[0, 1, 2, 3, 4])
    assert len(berts_ckpts) == 5 * 2


def test__multiberts_splits():
    berts_ckpts = MultiBERTCheckpoints(step=[0], seed=[0, 1])
    ckpts_splits = berts_ckpts.split(2)
    assert len(berts_ckpts) == 2
    assert len(ckpts_splits[0]) == 1
