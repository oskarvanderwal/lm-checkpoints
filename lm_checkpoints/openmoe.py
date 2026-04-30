from lm_checkpoints import AbstractCheckpoints, Checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict


class OpenMoECheckpoints(AbstractCheckpoints):
    """Class for iterating over OpenMoE checkpoints.

    OpenMoE is a family of open-sourced Mixture-of-Experts (MoE) LLMs.
    Unlike other models, OpenMoE checkpoints are stored as separate repositories
    rather than branches.

    See: https://github.com/XueFuzhao/OpenMoE
    """

    # Available checkpoints as (size, tokens_billions) -> repo
    _CHECKPOINTS = {
        ("base", None): "OrionZheng/openmoe-base",
        ("8b", 400): "OrionZheng/openmoe-8b-400B",
        ("8b", 600): "OrionZheng/openmoe-8b-600B",
        ("8b", 800): "OrionZheng/openmoe-8b-800B",
        ("8b", 1000): "OrionZheng/openmoe-8b-1T",
        ("8b", 1100): "OrionZheng/openmoe-8b",  # Final 8B checkpoint
        ("34b", 200): "OrionZheng/openmoe-34b-200B",
    }

    # Available sizes
    _SIZES = ["base", "8b", "34b"]

    # Steps (tokens in billions) available per size
    _SIZE_TO_STEPS = {
        "base": [None],  # Base model doesn't have token-based checkpoints
        "8b": [400, 600, 800, 1000, 1100],
        "34b": [200],
    }

    def __init__(
        self,
        size: str = "8b",
        step: List[int] = None,
        **kwargs,
    ) -> None:
        """Initialize the OpenMoECheckpoints.

        Args:
            size (str): Model size. Options: "base", "8b", "34b". Defaults to "8b".
            step (List[int], optional): List of token checkpoints (in billions) to consider.
        """
        super().__init__(**kwargs)

        if size not in self._SIZES:
            raise ValueError(f"Invalid size: {size}. Must be one of: {self._SIZES}")
        self.size = size

        self._steps = self._SIZE_TO_STEPS[size]
        if step:
            if not set(step).issubset(set(self._steps)):
                raise ValueError(f"Invalid steps for size {size}. Available: {self._steps}")
            self.steps = step
        else:
            self.steps = self._steps

    @property
    def name(self) -> str:
        return f"OpenMoE {self.size}"

    @staticmethod
    def last_step() -> int:
        """Last token checkpoint for OpenMoE-8B (in billions)."""
        return 1100

    @property
    def config(self) -> dict:
        return {"size": self.size}

    def get_model_name(self, step: int) -> str:
        """Get the HuggingFace repo for a specific checkpoint."""
        return self._CHECKPOINTS.get((self.size, step), f"OrionZheng/openmoe-{self.size}")

    @property
    def checkpoints(self) -> List[Dict[str, int]]:
        return [{"step": s} for s in self.steps]

    def __len__(self):
        return len(self.steps)

    def get_checkpoint(self, step) -> Checkpoint:
        model_name = self.get_model_name(step)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            trust_remote_code=True,
        )
        model.eval()
        model = model.to(self.device)

        commit_hash = self.get_revision_hash(model_name, "main")

        return Checkpoint(
            model,
            tokenizer=tokenizer,
            model_name=model_name,
            step=step,
            commit_hash=commit_hash,
        )
