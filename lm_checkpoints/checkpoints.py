from abc import ABC, abstractmethod
import tempfile
import torch
import numpy as np
from typing import List, Dict, Union, Literal, Optional
from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound


CachePolicy = Literal["keep", "previous", "bounded", "temporary"]


def records_to_list(list_of_dicts: Union[List[Dict[str, int]], Dict[str, int]]):
    """Transform a list of dictionaries to a dictionary of lists.
    If list_of_dicts is a dictionary, it will simply make a list of each values.
    From: https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists

    Args:
        list_of_dicts (list[dict[str, int]] | dict[str, int]): List of dictionaries, assuming each dictionary has the same keys.

    Returns:
        dict[list]: Dictionary of lists.
    """
    if not isinstance(list_of_dicts, list):
        list_of_dicts = [list_of_dicts]
    return {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}


def chunk(L, n):
    """
    Partition L into n chunks using every item in L and
    such that the resulting chunks differ in size by at
    most one element.

    >>> L = ['a', 'b', 'c', 'd']
    ['a', 'b', 'c', 'd']
    >>> chunk(L, 2)
    [['a', 'b'], ['c', 'd']]
    >>> chunk(L, 3)
    [['a'], ['b', 'c'], ['d']]
    >>> chunk(L, 4)
    [['a'], ['b'], ['c'], ['d']]
    >>> chunk(L, 5)
    [['a'], ['b'], [], ['c'], ['d']]
    """
    size = len(L) / float(n)

    def I(i):
        return int(round(i))

    return [L[I(size * i) : I(size * (i + 1))] for i in range(n)]


class AbstractCheckpoints(ABC):
    """Abstract class for iterating over model checkpoints"""

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
                Useful for isolating checkpoints in a project-specific location.
            cache_policy: How to manage cached checkpoints:
                - "keep": Default HF behavior, keep all downloaded checkpoints.
                - "previous": Delete previous checkpoint after loading next one.
                - "bounded": Prune oldest cached models when cache exceeds max_cache_size_gb.
                - "temporary": Use a temporary cache directory, deleted when iteration ends.
            max_cache_size_gb: Maximum cache size in GB (only used with cache_policy="bounded").
            local_files_only: If True, only load from local cache (no downloads).
        """
        self.low_cpu_mem_usage = True if device == "cpu" else False

        self._device = device
        if device == "cpu":
            self.device = torch.device("cpu")
        elif device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "mps":
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            raise ValueError(f"Invalid device: {device}. Must be one of: 'cpu', 'cuda', 'mps'")

        if cache_policy not in ("keep", "previous", "bounded", "temporary"):
            raise ValueError(
                f"Invalid cache_policy: {cache_policy}. "
                "Must be one of: 'keep', 'previous', 'bounded', 'temporary'"
            )

        if cache_policy == "bounded" and max_cache_size_gb is None:
            raise ValueError("max_cache_size_gb is required when cache_policy='bounded'")

        self.cache_dir = cache_dir
        self.cache_policy = cache_policy
        self.max_cache_size_gb = max_cache_size_gb
        self.local_files_only = local_files_only
        self._temp_cache_dir = None

    def get_revision_hash(self, name: str, revision: str) -> Optional[str]:
        """Returns the commit hash for the model and revision (e.g., step) combination."""
        try:
            cache_info = scan_cache_dir(cache_dir=self.cache_dir)
            for repo in cache_info.repos:
                if repo.repo_id == name and revision in repo.refs:
                    return repo.refs[revision].commit_hash
        except CacheNotFound:
            pass
        return None

    @staticmethod
    def get_cache_size_gb(cache_dir: Optional[str] = None) -> float:
        """Get the current HuggingFace cache size in GB.

        Args:
            cache_dir: Optional custom cache directory. If None, uses default HF cache.

        Returns:
            Cache size in GB, or 0.0 if cache doesn't exist.
        """
        try:
            cache_info = scan_cache_dir(cache_dir=cache_dir)
            return cache_info.size_on_disk / (1024**3)
        except CacheNotFound:
            return 0.0

    def _get_effective_cache_dir(self) -> Optional[str]:
        """Get the effective cache directory, creating temp dir if needed."""
        if self.cache_policy == "temporary" and self._temp_cache_dir is None:
            self._temp_cache_dir = tempfile.TemporaryDirectory()
        if self._temp_cache_dir is not None:
            return self._temp_cache_dir.name
        return self.cache_dir

    def _enforce_cache_limit(self) -> None:
        """Delete oldest cached revisions if cache exceeds max_cache_size_gb."""
        if self.cache_policy != "bounded" or self.max_cache_size_gb is None:
            return

        effective_cache_dir = self._get_effective_cache_dir()

        try:
            cache_info = scan_cache_dir(cache_dir=effective_cache_dir)
        except CacheNotFound:
            return

        current_size_gb = cache_info.size_on_disk / (1024**3)

        if current_size_gb <= self.max_cache_size_gb:
            return

        # Collect all revisions with their last accessed time
        revisions = []
        for repo in cache_info.repos:
            for revision in repo.revisions:
                revisions.append((revision.last_accessed, revision.commit_hash, revision.size_on_disk))

        # Sort by last accessed time (oldest first)
        revisions.sort(key=lambda x: x[0])

        # Delete oldest revisions until under limit
        deleted_count = 0
        for last_accessed, commit_hash, size in revisions:
            if current_size_gb <= self.max_cache_size_gb:
                break
            try:
                delete_strategy = cache_info.delete_revisions(commit_hash)
                delete_strategy.execute()
                current_size_gb -= size / (1024**3)
                deleted_count += 1
            except Exception:
                pass  # Skip if revision can't be deleted

    def _delete_revision(self, commit_hash: str) -> bool:
        """Delete a specific revision from the cache.

        Args:
            commit_hash: The commit hash to delete.

        Returns:
            True if deletion succeeded, False otherwise.
        """
        effective_cache_dir = self._get_effective_cache_dir()
        try:
            cache_info = scan_cache_dir(cache_dir=effective_cache_dir)
            delete_strategy = cache_info.delete_revisions(commit_hash)
            delete_strategy.execute()
            return True
        except (CacheNotFound, Exception):
            return False

    def _cleanup_temp_cache(self) -> None:
        """Clean up temporary cache directory if used."""
        if self._temp_cache_dir is not None:
            self._temp_cache_dir.cleanup()
            self._temp_cache_dir = None

    def split(self, n):
        """Convenience function for splitting checkpoints for e.g. parallel computing.
        If n > m (number of total checkpoints), it will return a list of m checkpoints objects instead.
        """
        total_length = len(self)
        chunk_indices = np.array_split(range(total_length), n)

        ckpts = []
        for ci in chunk_indices:
            if len(ci) > 0:
                start = ci[0].item()
                end = ci[-1].item()
                cfg = self.config
                cfg.update({k: set(v) for k, v in records_to_list(self.checkpoints[start : end + 1]).items()})
                # ckpts.append(self.__class__(seeds=set(cfg["seed"]), steps=set(cfg["step"])))
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
        """Convert a training step to the number of tokens seen.

        Args:
            step: Training step number.

        Returns:
            Number of tokens seen at this step.
        """
        raise NotImplementedError("Subclass must implement step_to_tokens")

    def tokens_to_step(self, tokens: int) -> int:
        """Convert number of tokens to the nearest training step.

        Args:
            tokens: Number of tokens.

        Returns:
            Training step number (rounded to nearest available step).
        """
        raise NotImplementedError("Subclass must implement tokens_to_step")

    @classmethod
    def final_checkpoints(cls, **kwargs):
        return cls(step=[cls.last_step()], **kwargs)

    @abstractmethod
    def __len__(self):
        pass

    def __getitem__(self, index):
        cfg = self.checkpoints[index]
        ckpt = self.get_checkpoint(**cfg)
        return ckpt

    def __iter__(self):
        delete_hashes = []
        try:
            for cfg in self.checkpoints:
                # Enforce cache size limit before loading (for "bounded" policy)
                if self.cache_policy == "bounded":
                    self._enforce_cache_limit()

                # Delete previous checkpoint (for "previous" policy)
                if self.cache_policy == "previous" and delete_hashes:
                    for commit_hash in delete_hashes:
                        self._delete_revision(commit_hash)
                    delete_hashes.clear()

                ckpt = self.get_checkpoint(**cfg)

                # Track commit_hash for deletion in "previous" policy
                if self.cache_policy == "previous":
                    commit_hash = ckpt.config.get("commit_hash")
                    if commit_hash:
                        delete_hashes.append(commit_hash)

                yield ckpt
        finally:
            # Clean up temporary cache directory if used
            if self.cache_policy == "temporary":
                self._cleanup_temp_cache()

    def map(self, fn, include_config: bool = False):
        """Apply a function to each checkpoint.

        Args:
            fn: Callable that takes a Checkpoint and returns a result.
            include_config: If True, yield (config, result) tuples.

        Yields:
            Results from applying fn to each checkpoint, optionally with config.

        Example:
            >>> def evaluate(ckpt):
            ...     return run_eval(ckpt.model, ckpt.tokenizer)
            >>> results = list(checkpoints.map(evaluate))

            >>> # With Inspect AI:
            >>> results = list(checkpoints.map(
            ...     lambda ckpt: inspect_eval(ckpt.model, tasks=[my_task])
            ... ))
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

        Example:
            >>> results = checkpoints.map_collect(lambda ckpt: ckpt.model.num_parameters())
            >>> # [{"model_name": "...", "step": 1000, "result": 125000000}, ...]
        """
        results = []
        for ckpt in self:
            entry = dict(ckpt.config)
            entry["result"] = fn(ckpt)
            results.append(entry)
        return results


class Checkpoint:
    """Convenience class for representing a checkpoint.
    Each checkpoint should at least have a model and model_name.
    """

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
