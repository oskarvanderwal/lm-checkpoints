from lm_checkpoints import AbstractCheckpoints, Checkpoint
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from itertools import product
from typing import List, Dict


class MultiBERTCheckpoints(AbstractCheckpoints):
    """Class for iterating over MultiBERT checkpoints"""

    def __init__(self, step=None, seed=None, **kwargs):
        """Initialize the MultiBERTCheckpoints.

        Args:
            step (List[int], optional): List of steps to consider, uses all available steps if not specified.
            seed (List[int], optional): List of seeds to consider, uses all available seeds if not specified.
        """
        super().__init__(**kwargs)

        self._seeds = [0, 1, 2, 3, 4]
        if seed:
            assert set(seed).issubset(set(self._seeds))
            self.seeds = seed
        else:
            self.seeds = self._seeds

        self._steps = [
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
        if step:
            assert set(step).issubset(set(self._steps))
            self.steps = step
        else:
            self.steps = self._steps

    @property
    def name(self) -> str:
        return "MultiBERTs"

    @staticmethod
    def last_step() -> int:
        """Last step of training."""
        return 2000

    @property
    def config(self) -> dict:
        """Returns a dictionary for re-initializing this checkpoints class.

        Returns:
            dict: Configuration of this checkpoints object.
        """
        return {}

    def get_model_name(self, step: int, seed: int) -> str:
        """Get the name for loading from the HF hub.

        Args:
            step (int): Checkpoint step.
            seed (int): Model seed.

        Returns:
            str: Name of the checkpoint on HF.
        """
        return f"google/multiberts-seed_{seed}-step_{step}k"

    @property
    def checkpoints(self) -> List[Dict[str, int]]:
        """Returns all step and seed combinations that make up the checkpoints.

        Returns:
            list[dict[str, int]]: List of dicts (seed, step) describing each checkpoint.
        """
        return list({"seed": p[0], "step": p[1]} for p in product(self.seeds, self.steps))

    # MultiBERTs: batch_size=256, seq_len=512, steps are in thousands
    # Step values (0, 20, ..., 2000) represent thousands of steps
    TOKENS_PER_TRAINING_STEP = 131_072  # 256 * 512
    STEPS_MULTIPLIER = 1000  # step=2000 means 2,000,000 training steps

    def __len__(self):
        return len(self.seeds) * len(self.steps)

    def step_to_tokens(self, step: int) -> int:
        """Convert a checkpoint step to tokens seen.

        MultiBERTs checkpoints are saved at intervals of 1000s of steps.
        E.g., step=2000 means 2,000,000 training steps.
        Each training step processes 131,072 tokens (batch=256 * seq_len=512).

        Args:
            step: Checkpoint step number (in thousands, e.g., 2000 = 2M steps).

        Returns:
            Number of tokens seen at this checkpoint.
        """
        training_steps = step * self.STEPS_MULTIPLIER
        return training_steps * self.TOKENS_PER_TRAINING_STEP

    def tokens_to_step(self, tokens: int) -> int:
        """Convert tokens to nearest checkpoint step.

        Args:
            tokens: Number of tokens.

        Returns:
            Nearest available checkpoint step (in thousands).
        """
        training_steps = tokens // self.TOKENS_PER_TRAINING_STEP
        target_step = training_steps // self.STEPS_MULTIPLIER
        return min(self._steps, key=lambda s: abs(s - target_step))

    def get_checkpoint(self, seed, step) -> Checkpoint:
        model_name = self.get_model_name(step, seed)
        cache_dir = self._get_effective_cache_dir()

        config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=self.local_files_only,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=self.local_files_only,
        )

        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            config=config,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            cache_dir=cache_dir,
            local_files_only=self.local_files_only,
        )
        model.eval()
        model = model.to(self.device)

        commit_hash = self.get_revision_hash(model_name, "main")

        return Checkpoint(
            model, tokenizer=tokenizer, model_name=model_name, seed=seed, step=step, commit_hash=commit_hash
        )
