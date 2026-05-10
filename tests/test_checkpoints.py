from lm_checkpoints import PythiaCheckpoints, MultiBERTCheckpoints, Checkpoint, AbstractCheckpoints
from lm_checkpoints.utils import records_to_list, chunk
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
    with pytest.raises(ValueError, match="Invalid size"):
        PythiaCheckpoints(size="invalid", step=[0], seed=[0])


def test__pythia_invalid_step():
    with pytest.raises(ValueError, match="Invalid steps"):
        PythiaCheckpoints(step=[99999], seed=[0])


def test__pythia_invalid_seed():
    with pytest.raises(ValueError, match="Invalid seeds"):
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
    with pytest.raises(ValueError, match="Invalid steps"):
        MultiBERTCheckpoints(step=[99999], seed=[0])


def test__multiberts_invalid_seed():
    with pytest.raises(ValueError, match="Invalid seeds"):
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


# =============================================================================
# Cache management tests
# =============================================================================


def test__cache_policy_default():
    """Test that cache_policy defaults to 'keep'."""
    ckpts = PythiaCheckpoints(size="14m", step=[0], seed=[0])
    assert ckpts.cache_policy == "keep"


def test__cache_policy_previous():
    """Test cache_policy='previous' is accepted."""
    ckpts = PythiaCheckpoints(size="14m", step=[0], seed=[0], cache_policy="previous")
    assert ckpts.cache_policy == "previous"


def test__cache_policy_invalid():
    """Test that invalid cache_policy raises ValueError."""
    with pytest.raises(ValueError, match="Invalid cache"):
        PythiaCheckpoints(size="14m", step=[0], seed=[0], cache_policy="invalid")


def test__cache_dir_parameter():
    """Test that cache_dir parameter is accepted."""
    ckpts = PythiaCheckpoints(size="14m", step=[0], seed=[0], cache_dir="/tmp/test-cache")
    assert ckpts.cache_dir == "/tmp/test-cache"


def test__cache_dir_default_none():
    """Test that cache_dir defaults to None (use HF default)."""
    ckpts = PythiaCheckpoints(size="14m", step=[0], seed=[0])
    assert ckpts.cache_dir is None


def test__local_files_only_parameter():
    """Test that local_files_only parameter is accepted."""
    ckpts = PythiaCheckpoints(size="14m", step=[0], seed=[0], local_files_only=True)
    assert ckpts.local_files_only is True


def test__local_files_only_default_false():
    """Test that local_files_only defaults to False."""
    ckpts = PythiaCheckpoints(size="14m", step=[0], seed=[0])
    assert ckpts.local_files_only is False


def test__get_cache_size_gb():
    """Test that get_cache_size_gb returns a float."""
    cache_size = AbstractCheckpoints.get_cache_size_gb()
    assert isinstance(cache_size, float)
    assert cache_size >= 0


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
