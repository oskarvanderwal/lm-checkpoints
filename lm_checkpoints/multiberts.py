from typing import List, Optional

from .checkpoints import Checkpoint, Checkpoints


class MultiBERTCheckpoints(Checkpoints):
    """Intermediate checkpoints of the MultiBERTs (https://huggingface.co/google/multiberts-seed_0)."""

    all_seeds = [0, 1, 2, 3, 4]
    all_steps = list(range(0, 200, 20)) + list(range(200, 2001, 100))

    def __init__(self, step: Optional[List[int]] = None, seed: Optional[List[int]] = None, **kwargs):
        """Initialize the MultiBERTCheckpoints.

        Args:
            step (List[int], optional): List of steps (in thousands) to consider, uses all available steps if not
                specified.
            seed (List[int], optional): List of seeds to consider, uses all available seeds if not specified.
            **kwargs: Passed on to `Checkpoints` (e.g., device, clean_cache, torch_dtype).
        """
        ckpts = [Checkpoint(self.get_model_name(t, s), step=t, seed=s) for s in self.all_seeds for t in self.all_steps]
        kwargs.setdefault("model_class", "AutoModelForMaskedLM")
        super().__init__(ckpts, **kwargs)
        self._checkpoints = self.filter(step=step or None, seed=seed or None)._checkpoints

    @property
    def name(self) -> str:
        return "MultiBERTs"

    @staticmethod
    def last_step() -> int:
        """Last step of training."""
        return 2000

    @classmethod
    def final_checkpoints(cls, **kwargs) -> "MultiBERTCheckpoints":
        return cls(step=[cls.last_step()], **kwargs)

    @staticmethod
    def get_model_name(step: int, seed: int) -> str:
        """Name of the repository on the HF hub."""
        return f"google/multiberts-seed_{seed}-step_{step}k"
