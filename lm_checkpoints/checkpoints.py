from abc import ABC, abstractmethod
import logging
import torch
import numpy as np
from typing import List, Dict, Optional

from lm_checkpoints.cache import CacheManager, CachePolicy
from lm_checkpoints.utils import records_to_list, nearest_available_step


logger = logging.getLogger(__name__)


class AbstractCheckpoints(ABC):
    """Abstract class for iterating over model checkpoints."""

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        cache_policy: CachePolicy = "keep",
        max_cache_size_gb: Optional[float] = None,
        local_files_only: bool = False,
    ):
        """Initialize checkpoints iterator.

        Args:
            device: Device to load models on ('cpu', 'cuda', 'mps').
            cache_dir: Custom HuggingFace cache directory. If None, uses default HF cache.
            cache_policy: How to manage cached checkpoints:
                - "keep": Default HF behavior, keep all downloaded checkpoints.
                - "previous": Delete previous checkpoint after loading next one.
                - "bounded": Prune oldest cached models when cache exceeds max_cache_size_gb.
            max_cache_size_gb: Maximum cache size in GB (only used with cache_policy="bounded").
            local_files_only: If True, only load from local cache (no downloads).
        """
        self.low_cpu_mem_usage = device == "cpu"

        self._device = device
        if device == "cpu":
            self.device = torch.device("cpu")
        elif device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "mps":
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            raise ValueError(f"Invalid device: {device}. Must be one of: 'cpu', 'cuda', 'mps'")

        self.cache_dir = cache_dir
        self.cache_policy = cache_policy
        self.max_cache_size_gb = max_cache_size_gb
        self.local_files_only = local_files_only

        self._cache = CacheManager(
            policy=cache_policy,
            cache_dir=cache_dir,
            max_size_gb=max_cache_size_gb,
        )

    def _get_effective_cache_dir(self) -> Optional[str]:
        """Get the effective cache directory."""
        return self.cache_dir

    def get_revision_hash(self, name: str, revision: str) -> Optional[str]:
        """Returns the commit hash for the model and revision combination."""
        return self._cache.get_revision_hash(name, revision)

    @staticmethod
    def get_cache_size_gb(cache_dir: Optional[str] = None) -> float:
        """Get the current HuggingFace cache size in GB."""
        return CacheManager.get_cache_size_gb(cache_dir)

    def split(self, n):
        """Split checkpoints into n chunks for parallel computing."""
        total_length = len(self)
        chunk_indices = np.array_split(range(total_length), n)

        ckpts = []
        for ci in chunk_indices:
            if len(ci) > 0:
                start = ci[0].item()
                end = ci[-1].item()
                cfg = self.config
                cfg.update({k: set(v) for k, v in records_to_list(self.checkpoints[start : end + 1]).items()})
                ckpts.append(self.__class__(**cfg))
        return ckpts

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def checkpoints(self):
        pass

    @staticmethod
    @abstractmethod
    def last_step():
        pass

    @abstractmethod
    def get_checkpoint(self):
        pass

    def step_to_tokens(self, step: int) -> int:
        """Convert a training step to the number of tokens seen."""
        raise NotImplementedError("Subclass must implement step_to_tokens")

    def tokens_to_step(self, tokens: int) -> int:
        """Convert number of tokens to the nearest training step."""
        raise NotImplementedError("Subclass must implement tokens_to_step")

    @classmethod
    def final_checkpoints(cls, **kwargs):
        return cls(step=[cls.last_step()], **kwargs)

    @abstractmethod
    def __len__(self):
        pass

    def __getitem__(self, index):
        cfg = self.checkpoints[index]
        return self.get_checkpoint(**cfg)

    def __iter__(self):
        for cfg in self.checkpoints:
            self._cache.on_pre_load()
            ckpt = self.get_checkpoint(**cfg)
            commit_hash = ckpt.config.get("commit_hash")
            self._cache.on_post_load(commit_hash)
            yield ckpt

    def map(self, fn, include_config: bool = False):
        """Apply a function to each checkpoint.

        Args:
            fn: Callable that takes a Checkpoint and returns a result.
            include_config: If True, yield (config, result) tuples.

        Yields:
            Results from applying fn to each checkpoint, optionally with config.
        """
        for ckpt in self:
            result = fn(ckpt)
            if include_config:
                yield (ckpt.config, result)
            else:
                yield result

    def map_collect(self, fn) -> List[Dict]:
        """Apply a function to each checkpoint and collect results with metadata.

        Args:
            fn: Callable that takes a Checkpoint and returns a result.

        Returns:
            List of dicts with checkpoint config and result.
        """
        results = []
        for ckpt in self:
            entry = dict(ckpt.config)
            entry["result"] = fn(ckpt)
            results.append(entry)
        return results


class Checkpoint:
    """Represents a single model checkpoint."""

    def __init__(self, model, model_name, tokenizer=None, **kwargs):
        self._model = model
        self._tokenizer = tokenizer
        self._config = {"model_name": model_name}
        self._config.update(kwargs)

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        if not self._tokenizer:
            raise ValueError("This checkpoint has no corresponding tokenizer.")
        return self._tokenizer
