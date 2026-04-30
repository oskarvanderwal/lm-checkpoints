from lm_checkpoints import AbstractCheckpoints, Checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict


class OLMoCheckpoints(AbstractCheckpoints):
    """Class for iterating over OLMo checkpoints from AI2.

    OLMo models have intermediate checkpoints released as branches with naming
    convention like 'step1000-tokens4B'.
    """

    # Available sizes and their HuggingFace repo names
    _SIZE_TO_REPO = {
        "1b": "allenai/OLMo-7B",  # Note: 1B uses OLMo-7B repo structure
        "7b": "allenai/OLMo-7B",
        "7b-twin-2t": "allenai/OLMo-7B-Twin-2T",
        "1b-v2": "allenai/OLMo-2-0425-1B",
        "7b-v2": "allenai/OLMo-2-1124-7B",
        "13b-v2": "allenai/OLMo-2-1124-13B",
    }

    # Steps available for each size (step number, tokens in billions)
    _SIZE_TO_STEPS = {
        "7b": list(range(1000, 557000, 1000)),  # step1000 to step556000
        "7b-twin-2t": list(range(1000, 557000, 1000)),
        "1b-v2": list(range(1000, 100000, 1000)),  # Approximate, may vary
        "7b-v2": list(range(1000, 100000, 1000)),
        "13b-v2": list(range(1000, 100000, 1000)),
    }

    def __init__(
        self,
        size: str = "7b",
        step: List[int] = None,
        **kwargs,
    ) -> None:
        """Initialize the OLMoCheckpoints.

        Args:
            size (str): Model size. Defaults to "7b".
                Options: "7b", "7b-twin-2t", "1b-v2", "7b-v2", "13b-v2"
            step (List[int], optional): List of steps to consider.
        """
        super().__init__(**kwargs)

        self._sizes = list(self._SIZE_TO_REPO.keys())
        if size not in self._sizes:
            raise ValueError(f"Invalid size: {size}. Must be one of: {self._sizes}")
        self.size = size

        self._steps = self._SIZE_TO_STEPS.get(size, list(range(1000, 100000, 1000)))
        if step:
            if not set(step).issubset(set(self._steps)):
                raise ValueError(f"Invalid steps. Available steps for {size}: {self._steps[:10]}...")
            self.steps = step
        else:
            self.steps = self._steps

    @property
    def name(self) -> str:
        return f"OLMo {self.size}"

    @staticmethod
    def last_step() -> int:
        """Last step of training for OLMo-7B."""
        return 556000

    @property
    def config(self) -> dict:
        return {"size": self.size}

    def get_model_name(self) -> str:
        return self._SIZE_TO_REPO[self.size]

    def _get_revision(self, step: int) -> str:
        """Get the revision string for a given step."""
        # OLMo uses format like 'step1000-tokens4B'
        # Tokens = step * 4096 (batch size) * 2048 (seq len) / 1e9
        # Simplified: approximately step * 0.004B tokens
        tokens_b = int(step * 4096 * 2048 / 1e9)
        return f"step{step}-tokens{tokens_b}B"

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
