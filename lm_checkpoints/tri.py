from lm_checkpoints import AbstractCheckpoints, Checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict


class TriCheckpoints(AbstractCheckpoints):
    """Class for iterating over Tri model checkpoints from Trillion Labs.

    Trillion Labs released intermediate checkpoints for Tri models (0.5B, 1.9B, 7B, 70B).
    Checkpoints are stored as branches at regular token intervals.

    See: https://huggingface.co/trillionlabs/Tri-70B-Intermediate-Checkpoints
    """

    # Available sizes and their approximate token intervals (in billions)
    _SIZE_TO_REPO = {
        "0.5b": "trillionlabs/Tri-70B-Intermediate-Checkpoints",
        "1.9b": "trillionlabs/Tri-70B-Intermediate-Checkpoints",
        "7b": "trillionlabs/Tri-70B-Intermediate-Checkpoints",
        "70b": "trillionlabs/Tri-70B-Intermediate-Checkpoints",
    }

    # Token intervals per size (in billions of tokens)
    _SIZE_TO_TOKEN_INTERVALS = {
        "0.5b": 20,  # ~20B tokens per checkpoint
        "1.9b": 40,  # ~40B tokens per checkpoint
        "7b": 160,  # ~160B tokens per checkpoint
        "70b": 160,  # ~160B tokens per checkpoint
    }

    def __init__(
        self,
        size: str = "7b",
        step: List[int] = None,
        **kwargs,
    ) -> None:
        """Initialize the TriCheckpoints.

        Args:
            size (str): Model size. Options: "0.5b", "1.9b", "7b", "70b". Defaults to "7b".
            step (List[int], optional): List of steps (token checkpoints) to consider.
        """
        super().__init__(**kwargs)

        self._sizes = list(self._SIZE_TO_REPO.keys())
        if size not in self._sizes:
            raise ValueError(f"Invalid size: {size}. Must be one of: {self._sizes}")
        self.size = size

        # Generate available steps based on token intervals
        interval = self._SIZE_TO_TOKEN_INTERVALS[size]
        # Approximate number of checkpoints (varies by model)
        max_tokens = {"0.5b": 500, "1.9b": 500, "7b": 2000, "70b": 2000}
        self._steps = list(range(interval, max_tokens[size] + 1, interval))

        if step:
            if not set(step).issubset(set(self._steps)):
                raise ValueError(f"Invalid steps for size {size}. Available: {self._steps}")
            self.steps = step
        else:
            self.steps = self._steps

    @property
    def name(self) -> str:
        return f"Tri {self.size}"

    @staticmethod
    def last_step() -> int:
        """Last token checkpoint for Tri-70B (in billions)."""
        return 2000

    @property
    def config(self) -> dict:
        return {"size": self.size}

    def get_model_name(self) -> str:
        return self._SIZE_TO_REPO[self.size]

    def _get_revision(self, step: int) -> str:
        """Get the revision/branch name for a given token checkpoint."""
        # Branch naming: size-tokensXXXB (e.g., "7b-tokens160B")
        return f"{self.size}-tokens{step}B"

    @property
    def checkpoints(self) -> List[Dict[str, int]]:
        return [{"step": s} for s in self.steps]

    def __len__(self):
        return len(self.steps)

    def get_checkpoint(self, step) -> Checkpoint:
        model_name = self.get_model_name()
        revision = self._get_revision(step)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            trust_remote_code=True,
        )
        model.eval()
        model = model.to(self.device)

        commit_hash = self.get_revision_hash(model_name, revision)

        return Checkpoint(
            model,
            tokenizer=tokenizer,
            model_name=model_name,
            step=step,
            commit_hash=commit_hash,
            revision=revision,
        )
