"""Core data structures: a lazily loaded `Checkpoint` record and a `Checkpoints` collection.

A checkpoint is nothing more than a pointer to model weights (a HF hub repo + revision, or a local directory)
plus some metadata (step, seed, ...). Nothing is downloaded or loaded until you access `.model` or `.tokenizer`.
"""

import copy
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from huggingface_hub import constants, list_repo_refs, scan_cache_dir

# kwargs of `from_pretrained` that also make sense for loading the tokenizer
_TOKENIZER_KWARGS = ("cache_dir", "token", "trust_remote_code", "local_files_only")


def _resolve_class(model_class):
    if model_class is None or isinstance(model_class, str):
        import transformers

        return getattr(transformers, model_class or "AutoModelForCausalLM")
    return model_class


@dataclass
class Checkpoint:
    """A single training checkpoint.

    Args:
        repo_id (str): Name of the repository on the HF hub, or a path to a local directory.
        revision (str, optional): Branch/tag/commit on the hub (e.g., "step1000"). Ignored for local paths.
        step (int, optional): Training step of the checkpoint.
        seed (int, optional): Seed of the training run (not a random seed for your experiments).
        meta (dict, optional): Any additional metadata (e.g., number of tokens seen).
        model_class: `transformers` class (or its name) used for loading. Defaults to "AutoModelForCausalLM".
        device (str, optional): Device the model is moved to after loading (e.g., "cuda", "cuda:1", "mps").
        load_kwargs (dict, optional): Passed on to `from_pretrained` (e.g., torch_dtype, device_map, token, cache_dir).
    """

    repo_id: str
    revision: Optional[str] = None
    step: Optional[int] = None
    seed: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    model_class: Any = field(default=None, repr=False)
    device: Optional[str] = field(default=None, repr=False)
    load_kwargs: Dict[str, Any] = field(default_factory=dict, repr=False)

    _model: Any = field(default=None, init=False, repr=False, compare=False)
    _tokenizer: Any = field(default=None, init=False, repr=False, compare=False)

    @property
    def is_local(self) -> bool:
        return Path(self.repo_id).is_dir()

    @property
    def name(self) -> str:
        """Short, filesystem-friendly name of the model (the repo id, or the run directory for local checkpoints)."""
        if self.is_local:
            path = Path(self.repo_id).resolve()
            return path.parent.name if self.step is not None else path.name
        return self.repo_id

    @property
    def config(self) -> Dict[str, Any]:
        """Metadata describing this checkpoint (not to be confused with `model.config`)."""
        return {
            "model_name": self.repo_id,
            "revision": self.revision,
            "step": self.step,
            "seed": self.seed,
            "commit_hash": self.commit_hash,
            **self.meta,
        }

    @property
    def commit_hash(self) -> Optional[str]:
        """Commit hash of the cached revision, or None if not (yet) in the local HF cache."""
        if self.is_local:
            return None
        # Resolved from the cache's refs rather than a specific file, as e.g. loading only the tokenizer caches the
        # revision without config.json. Layout: <cache>/models--org--name/{refs/<revision>,snapshots/<commit_hash>}
        repo_dir = Path(self.load_kwargs.get("cache_dir") or constants.HF_HUB_CACHE) / (
            "models--" + self.repo_id.replace("/", "--")
        )
        revision = self.revision or "main"
        ref = repo_dir / "refs" / revision
        commit_hash = ref.read_text().strip() if ref.is_file() else revision
        if re.fullmatch(r"[0-9a-f]{40}", commit_hash) and (repo_dir / "snapshots" / commit_hash).is_dir():
            return commit_hash
        return None

    def is_cached(self) -> bool:
        return self.is_local or self.commit_hash is not None

    def download(self) -> str:
        """Downloads the checkpoint without loading it, and returns the local path."""
        if self.is_local:
            return self.repo_id
        from huggingface_hub import snapshot_download

        return snapshot_download(
            self.repo_id,
            revision=self.revision,
            cache_dir=self.load_kwargs.get("cache_dir"),
            token=self.load_kwargs.get("token"),
        )

    def load_model(self, **kwargs):
        """Loads and returns a new model instance. `kwargs` override the `load_kwargs` of this checkpoint."""
        kwargs = {**self.load_kwargs, **kwargs}
        if not self.is_local and self.revision is not None:
            kwargs.setdefault("revision", self.revision)
        model = _resolve_class(self.model_class).from_pretrained(self.repo_id, **kwargs)
        model.eval()
        if self.device is not None:
            model = model.to(self.device)
        return model

    def load_tokenizer(self, **kwargs):
        """Loads and returns the tokenizer of this checkpoint."""
        from transformers import AutoTokenizer

        kwargs = {**{k: v for k, v in self.load_kwargs.items() if k in _TOKENIZER_KWARGS}, **kwargs}
        if not self.is_local and self.revision is not None:
            kwargs.setdefault("revision", self.revision)
        return AutoTokenizer.from_pretrained(self.repo_id, **kwargs)

    @property
    def model(self):
        """The model, loaded (and cached on this object) on first access."""
        if self._model is None:
            self._model = self.load_model()
        return self._model

    @property
    def tokenizer(self):
        """The tokenizer, loaded (and cached on this object) on first access."""
        if self._tokenizer is None:
            self._tokenizer = self.load_tokenizer()
        return self._tokenizer

    def unload(self) -> None:
        """Drops the references to the loaded model and tokenizer."""
        self._model = None
        self._tokenizer = None

    def delete_from_cache(self) -> None:
        """Deletes this revision from the local HF cache. Does nothing for local checkpoints."""
        commit_hash = self.commit_hash
        if commit_hash is None:
            return
        cache_info = scan_cache_dir(self.load_kwargs.get("cache_dir"))
        cache_info.delete_revisions(commit_hash).execute()


def _as_set(values) -> Optional[set]:
    if values is None:
        return None
    if isinstance(values, (str, int)):
        return {values}
    return set(values)


class Checkpoints:
    """An ordered collection of checkpoints that can be filtered, sliced, split and iterated over.

    Args:
        checkpoints (Iterable[Checkpoint]): The checkpoints.
        clean_cache (bool, optional): If True, deletes each checkpoint from the HF cache after you are done with it
            while iterating, unless it was already cached before. Defaults to False.
        device (str, optional): Device to move the models to (e.g., "cpu", "cuda", "cuda:1", "mps").
        model_class: `transformers` class (or its name) used for loading. Defaults to "AutoModelForCausalLM".
        **load_kwargs: Passed on to `from_pretrained` (e.g., torch_dtype, device_map, token, cache_dir).
    """

    def __init__(
        self,
        checkpoints: Iterable[Checkpoint] = (),
        clean_cache: bool = False,
        device: Optional[str] = None,
        model_class=None,
        **load_kwargs,
    ):
        self.clean_cache = clean_cache
        self._checkpoints = [
            replace(
                ckpt,
                model_class=model_class or ckpt.model_class,
                device=device or ckpt.device,
                load_kwargs={**ckpt.load_kwargs, **load_kwargs},
            )
            for ckpt in checkpoints
        ]

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        pattern: str = r"step(?P<step>\d+)",
        seed: Optional[int] = None,
        **kwargs,
    ) -> "Checkpoints":
        """Discovers the checkpoints stored as branches of a (possibly private) repository on the HF hub.

        Args:
            repo_id (str): Repository on the HF hub, e.g., "EleutherAI/pythia-14m" or "allenai/OLMo-2-0425-1B".
            pattern (str, optional): Regex that should fully match the branch names of checkpoints. The group named
                `step` (or else the first group) is parsed as the training step; other named groups are stored in
                `meta`. Defaults to r"step(?P<step>\\d+)".
            seed (int, optional): Seed to attach to all of these checkpoints.
            **kwargs: Passed on to `Checkpoints` (e.g., device, torch_dtype, token).
        """
        refs = list_repo_refs(repo_id, token=kwargs.get("token"))
        ckpts = []
        for branch in refs.branches:
            ckpt = _match(pattern, branch.name, repo_id=repo_id, revision=branch.name, seed=seed)
            if ckpt is not None:
                ckpts.append(ckpt)
        if not ckpts:
            raise ValueError(f"No branches of {repo_id} match the pattern {pattern!r}.")
        return cls(sorted(ckpts, key=lambda c: c.step), **kwargs)

    @classmethod
    def from_local(
        cls,
        path: Union[str, Path],
        pattern: str = r"checkpoint-(?P<step>\d+)",
        seed: Optional[int] = None,
        **kwargs,
    ) -> "Checkpoints":
        """Discovers the checkpoints saved as subdirectories of a local directory, e.g. the output directory of the
        HF `Trainer` (checkpoint-500/, checkpoint-1000/, ...).

        Args:
            path (str | Path): Directory containing the checkpoint subdirectories.
            pattern (str, optional): Regex that should fully match the subdirectory names; see `from_hub`.
                Defaults to r"checkpoint-(?P<step>\\d+)".
            seed (int, optional): Seed to attach to all of these checkpoints.
            **kwargs: Passed on to `Checkpoints` (e.g., device, torch_dtype, model_class).
        """
        ckpts = []
        for subdir in Path(path).iterdir():
            if subdir.is_dir():
                ckpt = _match(pattern, subdir.name, repo_id=str(subdir), seed=seed)
                if ckpt is not None:
                    ckpts.append(ckpt)
        if not ckpts:
            raise ValueError(f"No subdirectories of {path} match the pattern {pattern!r}.")
        return cls(sorted(ckpts, key=lambda c: c.step), **kwargs)

    def _new(self, checkpoints: List[Checkpoint]) -> "Checkpoints":
        new = copy.copy(self)
        new._checkpoints = list(checkpoints)
        return new

    @property
    def checkpoints(self) -> List[Checkpoint]:
        return list(self._checkpoints)

    @property
    def steps(self) -> List[int]:
        return sorted({c.step for c in self._checkpoints if c.step is not None})

    @property
    def seeds(self) -> List[int]:
        return sorted({c.seed for c in self._checkpoints if c.seed is not None})

    def filter(self, step=None, seed=None, **meta) -> "Checkpoints":
        """Returns the subset of checkpoints matching the given step(s), seed(s) and/or metadata value(s).
        Raises a ValueError if any of the requested steps or seeds are not available."""
        steps, seeds = _as_set(step), _as_set(seed)
        for requested, available, what in ((steps, self.steps, "step"), (seeds, self.seeds, "seed")):
            if requested is not None and not requested.issubset(available):
                raise ValueError(f"Unavailable {what}(s): {sorted(requested - set(available))}")
        meta = {k: _as_set(v) for k, v in meta.items()}
        return self._new(
            c
            for c in self._checkpoints
            if (steps is None or c.step in steps)
            and (seeds is None or c.seed in seeds)
            and all(c.meta.get(k) in v for k, v in meta.items())
        )

    def final(self) -> "Checkpoints":
        """Returns the last checkpoint of each seed."""
        last = {}
        for c in self._checkpoints:
            if c.seed not in last or (c.step or 0) >= (last[c.seed].step or 0):
                last[c.seed] = c
        keep = {id(c) for c in last.values()}
        return self._new(c for c in self._checkpoints if id(c) in keep)

    def split(self, n: int) -> List["Checkpoints"]:
        """Splits the checkpoints into (at most) n contiguous, non-overlapping chunks that differ in size by at most
        one, e.g. for parallel computations. Empty chunks are dropped."""
        size = len(self) / n
        chunks = [self._checkpoints[round(size * i) : round(size * (i + 1))] for i in range(n)]
        return [self._new(c) for c in chunks if c]

    def __add__(self, other: "Checkpoints") -> "Checkpoints":
        return self._new(self._checkpoints + other._checkpoints)

    def __len__(self) -> int:
        return len(self._checkpoints)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._new(self._checkpoints[index])
        # Return a fresh copy, so that loaded models are not kept alive by this collection
        return replace(self._checkpoints[index])

    def __iter__(self) -> Iterator[Checkpoint]:
        for ckpt in self._checkpoints:
            ckpt = replace(ckpt)
            was_cached = self.clean_cache and ckpt.is_cached()
            try:
                yield ckpt
            finally:
                if self.clean_cache:
                    ckpt.unload()
                    if not was_cached:
                        ckpt.delete_from_cache()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={len(self)}, seeds={self.seeds}, steps={_summarize(self.steps)})"


def _match(pattern: str, name: str, **kwargs) -> Optional[Checkpoint]:
    m = re.fullmatch(pattern, name)
    if m is None:
        return None
    groups = m.groupdict()
    step = groups.pop("step", None) if "step" in groups else (m.group(1) if m.re.groups else None)
    meta = {k: v for k, v in groups.items() if v is not None}
    return Checkpoint(step=int(step) if step is not None else None, meta=meta, **kwargs)


def _summarize(values: list) -> str:
    return str(values) if len(values) <= 6 else f"[{values[0]}, {values[1]}, ..., {values[-1]}]"


# Backwards compatibility
AbstractCheckpoints = Checkpoints
