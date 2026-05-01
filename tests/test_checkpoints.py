from lm_checkpoints import PythiaCheckpoints, MultiBERTCheckpoints, Checkpoint, AbstractCheckpoints
from lm_checkpoints.checkpoints import records_to_list, chunk
from lm_checkpoints.testing import multi_device
import pytest
import torch


# =============================================================================
# Device loading tests (require model downloads)
# =============================================================================


@multi_device
def test__pythia_multi_device(device: str):
    ckpt = PythiaCheckpoints(step=[143000], seed=[0], device=device)[0]
    assert ckpt.model.device.type == device


@multi_device
def test__multibert_multi_device(device: str):
    ckpt = MultiBERTCheckpoints(step=[2000], seed=[0], device=device)[0]
    assert ckpt.model.device.type == device


# -----------------------------------------------------------------------------
# Large model download tests (skipped by default)
# Run with: pytest -k "olmo_download or tri_download or openmoe_download" --run-skipped
# -----------------------------------------------------------------------------

from lm_checkpoints import OLMoCheckpoints, TriCheckpoints, OpenMoECheckpoints


@pytest.mark.skip(reason="Large model download (~2GB) - run explicitly if needed")
def test__olmo_download():
    """Test OLMo model download and loading."""
    ckpt = OLMoCheckpoints(size="1b-v2", step=[1000], device="cpu")[0]
    assert ckpt.model is not None
    assert ckpt.tokenizer is not None


@pytest.mark.skip(reason="Large model download (~1GB) - run explicitly if needed")
def test__tri_download():
    """Test Tri model download and loading."""
    ckpt = TriCheckpoints(size="0.5b", step=[20], device="cpu")[0]
    assert ckpt.model is not None
    assert ckpt.tokenizer is not None


@pytest.mark.skip(reason="Large model download (~16GB+) - run explicitly if needed")
def test__openmoe_download():
    """Test OpenMoE model download and loading."""
    ckpt = OpenMoECheckpoints(size="8b", step=[400], device="cpu")[0]
    assert ckpt.model is not None
    assert ckpt.tokenizer is not None


# =============================================================================
# PythiaCheckpoints tests
# =============================================================================


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


def test__pythia_all_valid_sizes():
    valid_sizes = ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
    for size in valid_sizes:
        ckpts = PythiaCheckpoints(size=size, step=[0], seed=[0])
        assert ckpts.size == size


def test__pythia_invalid_size():
    with pytest.raises(AssertionError):
        PythiaCheckpoints(size="invalid", step=[0], seed=[0])


def test__pythia_invalid_step():
    with pytest.raises(AssertionError):
        PythiaCheckpoints(step=[99999], seed=[0])


def test__pythia_invalid_seed():
    with pytest.raises(AssertionError):
        PythiaCheckpoints(step=[0], seed=[99])


def test__pythia_large_model_single_seed():
    for size in ["1b", "1.4b", "2.8b", "6.9b", "12b"]:
        ckpts = PythiaCheckpoints(size=size, step=[0])
        assert ckpts.seeds == [0]


def test__pythia_small_model_multiple_seeds():
    for size in ["14m", "31m", "70m", "160m", "410m"]:
        ckpts = PythiaCheckpoints(size=size, step=[0])
        assert ckpts.seeds == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test__pythia_name_property():
    ckpts = PythiaCheckpoints(size="70m", step=[0], seed=[0])
    assert ckpts.name == "Pythia 70m"


def test__pythia_last_step():
    assert PythiaCheckpoints.last_step() == 143000


def test__pythia_config():
    ckpts = PythiaCheckpoints(size="70m", step=[0], seed=[0])
    config = ckpts.config
    assert config["size"] == "70m"
    assert config["deduped"] == False


def test__pythia_checkpoints_property():
    ckpts = PythiaCheckpoints(size="14m", step=[0, 1], seed=[0, 1])
    checkpoints = ckpts.checkpoints
    assert len(checkpoints) == 4
    assert {"seed": 0, "step": 0} in checkpoints
    assert {"seed": 1, "step": 1} in checkpoints


def test__pythia_model_name():
    ckpts = PythiaCheckpoints(size="70m", step=[0], seed=[0])
    assert ckpts.get_model_name(0) == "EleutherAI/pythia-70m"
    assert ckpts.get_model_name(1) == "EleutherAI/pythia-70m-seed1"


def test__pythia_final_checkpoints():
    ckpts = PythiaCheckpoints.final_checkpoints(size="14m", seed=[0])
    assert ckpts.steps == [143000]


def test__pythia_deduped_not_implemented():
    with pytest.raises(NotImplementedError):
        PythiaCheckpoints(size="14m", deduped=True)


# =============================================================================
# MultiBERTCheckpoints tests
# =============================================================================


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


def test__multiberts_invalid_step():
    with pytest.raises(AssertionError):
        MultiBERTCheckpoints(step=[99999], seed=[0])


def test__multiberts_invalid_seed():
    with pytest.raises(AssertionError):
        MultiBERTCheckpoints(step=[0], seed=[99])


def test__multiberts_name_property():
    ckpts = MultiBERTCheckpoints(step=[0], seed=[0])
    assert ckpts.name == "MultiBERTs"


def test__multiberts_last_step():
    assert MultiBERTCheckpoints.last_step() == 2000


def test__multiberts_config():
    ckpts = MultiBERTCheckpoints(step=[0], seed=[0])
    config = ckpts.config
    assert config == {}


def test__multiberts_checkpoints_property():
    ckpts = MultiBERTCheckpoints(step=[0, 20], seed=[0, 1])
    checkpoints = ckpts.checkpoints
    assert len(checkpoints) == 4
    assert {"seed": 0, "step": 0} in checkpoints
    assert {"seed": 1, "step": 20} in checkpoints


def test__multiberts_model_name():
    ckpts = MultiBERTCheckpoints(step=[0], seed=[0])
    assert ckpts.get_model_name(step=100, seed=2) == "google/multiberts-seed_2-step_100k"


def test__multiberts_final_checkpoints():
    ckpts = MultiBERTCheckpoints.final_checkpoints(seed=[0])
    assert ckpts.steps == [2000]


def test__multiberts_default_seeds():
    ckpts = MultiBERTCheckpoints(step=[0])
    assert ckpts.seeds == [0, 1, 2, 3, 4]


def test__multiberts_default_steps():
    ckpts = MultiBERTCheckpoints(seed=[0])
    expected_steps = [
        0,
        20,
        40,
        60,
        80,
        100,
        120,
        140,
        160,
        180,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        1100,
        1200,
        1300,
        1400,
        1500,
        1600,
        1700,
        1800,
        1900,
        2000,
    ]
    assert ckpts.steps == expected_steps


# =============================================================================
# Device validation tests
# =============================================================================


def test__invalid_device():
    with pytest.raises(ValueError, match="Invalid device"):
        PythiaCheckpoints(size="14m", step=[0], seed=[0], device="invalid")


def test__valid_devices():
    for device in ["cpu", "cuda", "mps"]:
        ckpts = PythiaCheckpoints(size="14m", step=[0], seed=[0], device=device)
        assert ckpts._device == device


# =============================================================================
# Checkpoint class tests
# =============================================================================


def test__checkpoint_basic():
    mock_model = torch.nn.Linear(10, 10)
    ckpt = Checkpoint(model=mock_model, model_name="test-model")
    assert ckpt.model == mock_model
    assert ckpt.config["model_name"] == "test-model"


def test__checkpoint_with_tokenizer():
    mock_model = torch.nn.Linear(10, 10)
    mock_tokenizer = "mock_tokenizer"
    ckpt = Checkpoint(model=mock_model, model_name="test-model", tokenizer=mock_tokenizer)
    assert ckpt.tokenizer == mock_tokenizer


def test__checkpoint_no_tokenizer():
    mock_model = torch.nn.Linear(10, 10)
    ckpt = Checkpoint(model=mock_model, model_name="test-model")
    with pytest.raises(ValueError, match="no corresponding tokenizer"):
        _ = ckpt.tokenizer


def test__checkpoint_extra_kwargs():
    mock_model = torch.nn.Linear(10, 10)
    ckpt = Checkpoint(model=mock_model, model_name="test-model", seed=42, step=1000)
    assert ckpt.config["seed"] == 42
    assert ckpt.config["step"] == 1000


# =============================================================================
# Utility function tests
# =============================================================================


def test__records_to_list_from_list():
    input_data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    result = records_to_list(input_data)
    assert result == {"a": [1, 3], "b": [2, 4]}


def test__records_to_list_from_dict():
    input_data = {"a": 1, "b": 2}
    result = records_to_list(input_data)
    assert result == {"a": [1], "b": [2]}


def test__chunk_even_split():
    result = chunk(['a', 'b', 'c', 'd'], 2)
    assert result == [['a', 'b'], ['c', 'd']]


def test__chunk_uneven_split():
    result = chunk(['a', 'b', 'c', 'd'], 3)
    assert len(result) == 3
    assert sum(len(c) for c in result) == 4


def test__chunk_more_chunks_than_items():
    result = chunk(['a', 'b'], 4)
    assert len(result) == 4
    non_empty = [c for c in result if c]
    assert len(non_empty) == 2


# =============================================================================
# Split edge cases
# =============================================================================


def test__split_more_than_checkpoints():
    ckpts = PythiaCheckpoints(size="14m", step=[0], seed=[0])
    splits = ckpts.split(5)
    assert len(splits) == 1
    assert len(splits[0]) == 1


def test__split_equal_to_checkpoints():
    ckpts = PythiaCheckpoints(size="14m", step=[0, 1], seed=[0, 1])
    splits = ckpts.split(4)
    assert len(splits) == 4
    for s in splits:
        assert len(s) == 1


# =============================================================================
# OLMoCheckpoints tests
# =============================================================================

from lm_checkpoints import OLMoCheckpoints


def test__olmo_valid_sizes():
    valid_sizes = ["7b", "7b-twin-2t", "1b-v2", "7b-v2", "13b-v2"]
    for size in valid_sizes:
        ckpts = OLMoCheckpoints(size=size, step=[1000])
        assert ckpts.size == size


def test__olmo_invalid_size():
    with pytest.raises(ValueError, match="Invalid size"):
        OLMoCheckpoints(size="invalid", step=[1000])


def test__olmo_name_property():
    ckpts = OLMoCheckpoints(size="7b", step=[1000])
    assert ckpts.name == "OLMo 7b"


def test__olmo_last_step():
    assert OLMoCheckpoints.last_step() == 556000


def test__olmo_config():
    ckpts = OLMoCheckpoints(size="7b", step=[1000])
    assert ckpts.config == {"size": "7b"}


def test__olmo_checkpoints_property():
    ckpts = OLMoCheckpoints(size="7b", step=[1000, 2000, 3000])
    checkpoints = ckpts.checkpoints
    assert len(checkpoints) == 3
    assert {"step": 1000} in checkpoints
    assert {"step": 3000} in checkpoints


def test__olmo_len():
    ckpts = OLMoCheckpoints(size="7b", step=[1000, 2000])
    assert len(ckpts) == 2


def test__olmo_model_name():
    ckpts = OLMoCheckpoints(size="7b", step=[1000])
    assert ckpts.get_model_name() == "allenai/OLMo-7B"

    ckpts_v2 = OLMoCheckpoints(size="7b-v2", step=[1000])
    assert ckpts_v2.get_model_name() == "allenai/OLMo-2-1124-7B"


# =============================================================================
# TriCheckpoints tests
# =============================================================================

from lm_checkpoints import TriCheckpoints


def test__tri_valid_sizes():
    # Each size has different token intervals
    size_to_step = {"0.5b": [20], "1.9b": [40], "7b": [160], "70b": [160]}
    for size, step in size_to_step.items():
        ckpts = TriCheckpoints(size=size, step=step)
        assert ckpts.size == size


def test__tri_invalid_size():
    with pytest.raises(ValueError, match="Invalid size"):
        TriCheckpoints(size="invalid")


def test__tri_name_property():
    ckpts = TriCheckpoints(size="70b", step=[160])
    assert ckpts.name == "Tri 70b"


def test__tri_last_step():
    assert TriCheckpoints.last_step() == 2000


def test__tri_config():
    ckpts = TriCheckpoints(size="7b", step=[160])
    assert ckpts.config == {"size": "7b"}


def test__tri_checkpoints_property():
    ckpts = TriCheckpoints(size="7b", step=[160, 320])
    checkpoints = ckpts.checkpoints
    assert len(checkpoints) == 2
    assert {"step": 160} in checkpoints


def test__tri_len():
    ckpts = TriCheckpoints(size="0.5b", step=[20, 40])
    assert len(ckpts) == 2


def test__tri_model_name():
    ckpts = TriCheckpoints(size="70b", step=[160])
    assert ckpts.get_model_name() == "trillionlabs/Tri-70B-Intermediate-Checkpoints"


# =============================================================================
# OpenMoECheckpoints tests
# =============================================================================

from lm_checkpoints import OpenMoECheckpoints


def test__openmoe_valid_sizes():
    valid_sizes = ["base", "8b", "34b"]
    for size in valid_sizes:
        step = [200] if size == "34b" else ([None] if size == "base" else [400])
        ckpts = OpenMoECheckpoints(size=size, step=step)
        assert ckpts.size == size


def test__openmoe_invalid_size():
    with pytest.raises(ValueError, match="Invalid size"):
        OpenMoECheckpoints(size="invalid")


def test__openmoe_name_property():
    ckpts = OpenMoECheckpoints(size="8b", step=[400])
    assert ckpts.name == "OpenMoE 8b"


def test__openmoe_last_step():
    assert OpenMoECheckpoints.last_step() == 1100


def test__openmoe_config():
    ckpts = OpenMoECheckpoints(size="8b", step=[400])
    assert ckpts.config == {"size": "8b"}


def test__openmoe_checkpoints_property():
    ckpts = OpenMoECheckpoints(size="8b", step=[400, 600, 800])
    checkpoints = ckpts.checkpoints
    assert len(checkpoints) == 3
    assert {"step": 400} in checkpoints
    assert {"step": 800} in checkpoints


def test__openmoe_len():
    ckpts = OpenMoECheckpoints(size="8b", step=[400, 600])
    assert len(ckpts) == 2


def test__openmoe_model_name():
    ckpts = OpenMoECheckpoints(size="8b", step=[400])
    assert ckpts.get_model_name(400) == "OrionZheng/openmoe-8b-400B"
    assert ckpts.get_model_name(1100) == "OrionZheng/openmoe-8b"


def test__openmoe_8b_all_steps():
    ckpts = OpenMoECheckpoints(size="8b")
    assert ckpts.steps == [400, 600, 800, 1000, 1100]


def test__openmoe_34b_steps():
    ckpts = OpenMoECheckpoints(size="34b")
    assert ckpts.steps == [200]


# =============================================================================
# Step to tokens conversion tests
# =============================================================================


def test__pythia_step_to_tokens():
    ckpts = PythiaCheckpoints(size="14m", step=[0, 1000], seed=[0])
    # Pythia uses 2M tokens per step
    assert ckpts.step_to_tokens(0) == 0
    assert ckpts.step_to_tokens(1000) == 1000 * 2_097_152
    assert ckpts.step_to_tokens(143000) == 143000 * 2_097_152  # ~300B tokens


def test__pythia_tokens_to_step():
    ckpts = PythiaCheckpoints(size="14m", step=[0, 1000, 2000], seed=[0])
    # Should find nearest available step
    assert ckpts.tokens_to_step(0) == 0
    assert ckpts.tokens_to_step(2_097_152_000) == 1000  # 1000 steps worth
    assert ckpts.tokens_to_step(3_000_000_000) == 1000  # Closer to 1000 than 2000


def test__multiberts_step_to_tokens():
    ckpts = MultiBERTCheckpoints(step=[0, 2000], seed=[0])
    # MultiBERTs: step=2000 means 2M training steps * 131,072 tokens each
    assert ckpts.step_to_tokens(0) == 0
    assert ckpts.step_to_tokens(2000) == 2000 * 1000 * 131_072  # ~262B tokens


def test__multiberts_tokens_to_step():
    ckpts = MultiBERTCheckpoints(step=[0, 100, 200], seed=[0])
    assert ckpts.tokens_to_step(0) == 0
    # 100k steps * 131,072 = ~13B tokens
    assert ckpts.tokens_to_step(13_000_000_000) == 100


def test__olmo_step_to_tokens():
    ckpts = OLMoCheckpoints(size="7b", step=[1000, 2000])
    # OLMo uses ~4M tokens per step
    assert ckpts.step_to_tokens(1000) == 1000 * 4_194_304  # ~4B tokens
    assert ckpts.step_to_tokens(2000) == 2000 * 4_194_304  # ~8B tokens


def test__olmo_tokens_to_step():
    ckpts = OLMoCheckpoints(size="7b", step=[1000, 2000])
    # 4B tokens should map to step 1000
    assert ckpts.tokens_to_step(4_000_000_000) == 1000
    assert ckpts.tokens_to_step(8_000_000_000) == 2000


def test__tri_step_to_tokens():
    ckpts = TriCheckpoints(size="70b", step=[160, 320])
    # Tri steps are in billions of tokens
    assert ckpts.step_to_tokens(160) == 160_000_000_000  # 160B
    assert ckpts.step_to_tokens(320) == 320_000_000_000  # 320B


def test__tri_tokens_to_step():
    ckpts = TriCheckpoints(size="70b", step=[160, 320])
    assert ckpts.tokens_to_step(160_000_000_000) == 160
    assert ckpts.tokens_to_step(200_000_000_000) == 160  # Closer to 160 than 320


def test__openmoe_step_to_tokens():
    ckpts = OpenMoECheckpoints(size="8b", step=[400, 600])
    # OpenMoE steps are in billions of tokens
    assert ckpts.step_to_tokens(400) == 400_000_000_000  # 400B
    assert ckpts.step_to_tokens(600) == 600_000_000_000  # 600B


def test__openmoe_tokens_to_step():
    ckpts = OpenMoECheckpoints(size="8b", step=[400, 600, 800])
    assert ckpts.tokens_to_step(400_000_000_000) == 400
    assert ckpts.tokens_to_step(500_000_000_000) == 400  # Closer to 400 than 600


def test__openmoe_base_step_to_tokens():
    ckpts = OpenMoECheckpoints(size="base", step=[None])
    # Base model returns 0 for None step
    assert ckpts.step_to_tokens(None) == 0


# =============================================================================
# Cache management tests
# =============================================================================


def test__max_cache_size_parameter():
    """Test that max_cache_size_gb parameter is accepted."""
    ckpts = PythiaCheckpoints(size="14m", step=[0], seed=[0], max_cache_size_gb=50.0)
    assert ckpts.max_cache_size_gb == 50.0


def test__max_cache_size_default_none():
    """Test that max_cache_size_gb defaults to None."""
    ckpts = PythiaCheckpoints(size="14m", step=[0], seed=[0])
    assert ckpts.max_cache_size_gb is None


def test__get_cache_size_gb():
    """Test that get_cache_size_gb returns a float (or raises CacheNotFound if no cache)."""
    from huggingface_hub.errors import CacheNotFound
    try:
        cache_size = AbstractCheckpoints.get_cache_size_gb()
        assert isinstance(cache_size, float)
        assert cache_size >= 0
    except CacheNotFound:
        # No HuggingFace cache exists yet, which is fine
        pass


def test__clean_cache_parameter():
    """Test that clean_cache parameter is accepted."""
    ckpts = PythiaCheckpoints(size="14m", step=[0], seed=[0], clean_cache=True)
    assert ckpts.clean_cache is True


def test__clean_cache_default_false():
    """Test that clean_cache defaults to False."""
    ckpts = PythiaCheckpoints(size="14m", step=[0], seed=[0])
    assert ckpts.clean_cache is False


# =============================================================================
# Map/apply functionality tests
# =============================================================================


class MockCheckpointsForMap(AbstractCheckpoints):
    """Mock checkpoint class for testing map functionality without downloads."""

    def __init__(self, num_checkpoints=3, **kwargs):
        super().__init__(**kwargs)
        self._num = num_checkpoints

    @property
    def name(self):
        return "MockCheckpoints"

    @property
    def checkpoints(self):
        return [{"step": i} for i in range(self._num)]

    @staticmethod
    def last_step():
        return 2

    @property
    def config(self):
        return {}

    def get_checkpoint(self, step):
        mock_model = torch.nn.Linear(10, 10)
        return Checkpoint(model=mock_model, model_name=f"mock-{step}", step=step)

    def __len__(self):
        return self._num


def test__map_basic():
    """Test basic map functionality."""
    ckpts = MockCheckpointsForMap(num_checkpoints=3)
    results = list(ckpts.map(lambda ckpt: ckpt.config["step"]))
    assert results == [0, 1, 2]


def test__map_with_config():
    """Test map with include_config=True."""
    ckpts = MockCheckpointsForMap(num_checkpoints=2)
    results = list(ckpts.map(lambda ckpt: ckpt.config["step"] * 2, include_config=True))
    assert len(results) == 2
    assert results[0][1] == 0  # step 0 * 2
    assert results[1][1] == 2  # step 1 * 2
    assert "model_name" in results[0][0]


def test__map_collect():
    """Test map_collect returns list of dicts with results."""
    ckpts = MockCheckpointsForMap(num_checkpoints=3)
    results = ckpts.map_collect(lambda ckpt: ckpt.config["step"] ** 2)
    assert len(results) == 3
    assert results[0]["result"] == 0
    assert results[1]["result"] == 1
    assert results[2]["result"] == 4
    assert "model_name" in results[0]
    assert "step" in results[0]


def test__map_with_model_operation():
    """Test map with actual model operation."""
    ckpts = MockCheckpointsForMap(num_checkpoints=2)
    results = list(ckpts.map(lambda ckpt: sum(p.numel() for p in ckpt.model.parameters())))
    # Linear(10, 10) has 10*10 + 10 = 110 parameters
    assert all(r == 110 for r in results)
