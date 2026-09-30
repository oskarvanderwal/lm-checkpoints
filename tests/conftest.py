import pytest


def pytest_addoption(parser):
    parser.addoption("--run-network", action="store_true", help="Run tests that download models from the HF hub.")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
