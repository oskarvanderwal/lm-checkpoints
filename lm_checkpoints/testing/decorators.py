import torch
import pytest

_available_devices = ["cpu"]
if torch.cuda.is_available():
    _available_devices.append("cuda:0")
if torch.backends.mps.is_available():
    _available_devices.append("mps")

def multi_device(test_method):
    """
    Adapted from AllenNLP.

    Decorator that provides an argument `device` of type `str` to a test function.

    If you have a CUDA capable GPU available, device will be "cuda:0", otherwise the device will
    be "cpu".

    !!! Note
        If you have a CUDA capable GPU available, but you want to run the test using CPU only,
        just set the environment variable "CUDA_CAPABLE_DEVICES=''" before running pytest.
    """
    return pytest.mark.parametrize("device", _available_devices)(test_method)