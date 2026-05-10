from lm_checkpoints import AbstractCheckpoints, Checkpoint
from lm_checkpoints.utils import nearest_available_step
from itertools import product
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from typing import List, Dict


class PythiaCheckpoints(AbstractCheckpoints):
    """Class for iterating over Pythia checkpoints."""

    _SIZES = ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
    _LARGE_SIZES = ["1b", "1.4b", "2.8b", "6.9b", "12b"]
    _ALL_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000, 144000, 1000))
    TOKENS_PER_STEP = 2_097_152  # batch_size=1024 * seq_len=2048

    def __init__(
        self,
        size: str = "14m",
        step: List[int] = None,
        seed: List[int] = None,
        deduped: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the PythiaCheckpoints.

        Args:
            size: Model size. Defaults to "14m".
            step: List of steps to consider, uses all available steps if not specified.
            seed: List of seeds to consider, uses all available seeds if not specified.
            deduped: Use deduped version of Pythia. Defaults to False.
        """
        super().__init__(**kwargs)

        if size not in self._SIZES:
            raise ValueError(f"Invalid size: {size}. Must be one of: {self._SIZES}")
        self.size = size
        self.deduped = deduped

        if deduped:
            raise NotImplementedError("Deduped Pythia checkpoints not yet supported")

        # Different seeds only available for the smaller models
        self._seeds = [0] if self.size in self._LARGE_SIZES else list(range(10))

        if seed:
            invalid = set(seed) - set(self._seeds)
            if invalid:
                raise ValueError(f"Invalid seeds: {invalid}. Available for {size}: {self._seeds}")
            self.seeds = seed
        else:
            self.seeds = self._seeds

        self._steps = self._ALL_STEPS
        if step:
            invalid = set(step) - set(self._steps)
            if invalid:
                raise ValueError(f"Invalid steps: {invalid}")
            self.steps = step
        else:
            self.steps = self._steps

    @property
    def name(self) -> str:
        return f"Pythia {self.size}"

    @staticmethod
    def last_step() -> int:
        """Last step of training."""
        return 143000

    @property
    def config(self) -> dict:
        """Returns a dictionary for re-initializing this checkpoints class.

        Returns:
            dict: Configuration of this checkpoints object.
        """
        return {"size": self.size, "deduped": self.deduped}

    def get_model_name(self, seed: int) -> str:
        """Get the name for loading from the HF hub.

        Args:
            seed (int): Model seed.

        Returns:
            str: Name of the checkpoint on HF.
        """
        if seed == 0:
            return f"EleutherAI/pythia-{self.size}"
        else:
            return f"EleutherAI/pythia-{self.size}-seed{seed}"

    @property
    def checkpoints(self) -> List[Dict[str, int]]:
        """Returns all step and seed combinations that make up the checkpoints.

        Returns:
            list[dict[str, int]]: List of dicts (seed, step) describing each checkpoint.
        """
        return list({"seed": p[0], "step": p[1]} for p in product(self.seeds, self.steps))

    def __len__(self):
        return len(self.seeds) * len(self.steps)

    def step_to_tokens(self, step: int) -> int:
        """Convert a training step to tokens seen."""
        return step * self.TOKENS_PER_STEP

    def tokens_to_step(self, tokens: int) -> int:
        """Convert tokens to nearest available training step."""
        return nearest_available_step(tokens, self.TOKENS_PER_STEP, self._steps)

    def get_checkpoint(self, seed, step) -> Checkpoint:
        model_name = self.get_model_name(seed)
        revision = f"step{step}"
        cache_dir = self._get_effective_cache_dir()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=self.local_files_only,
        )

        model = GPTNeoXForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            cache_dir=cache_dir,
            local_files_only=self.local_files_only,
        )
        model.eval()
        model = model.to(self.device)

        commit_hash = self.get_revision_hash(model_name, revision)

        return Checkpoint(
            model,
            tokenizer=tokenizer,
            model_name=model_name,
            seed=seed,
            step=step,
            commit_hash=commit_hash,
            revision=revision,
        )
