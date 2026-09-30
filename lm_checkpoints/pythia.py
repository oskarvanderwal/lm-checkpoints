from typing import List, Optional

from .checkpoints import Checkpoint, Checkpoints


class PythiaCheckpoints(Checkpoints):
    """Checkpoints of the Pythia models (https://github.com/EleutherAI/pythia)."""

    sizes = ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
    all_steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000, 144000, 1000))

    def __init__(
        self,
        size: str = "14m",
        step: Optional[List[int]] = None,
        seed: Optional[List[int]] = None,
        deduped: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the PythiaCheckpoints.

        Args:
            size (str): Model size. Defaults to "14m".
            step (List[int], optional): List of steps to consider, uses all available steps if not specified.
            seed (List[int], optional): List of seeds to consider, uses all available seeds if not specified.
            deduped (bool, optional): Use the models trained on the deduplicated Pile. Defaults to False.
            **kwargs: Passed on to `Checkpoints` (e.g., device, clean_cache, torch_dtype).
        """
        if size not in self.sizes:
            raise ValueError(f"Unknown Pythia size {size!r}, choose from {self.sizes}.")
        self.size = size
        self.deduped = deduped

        # Multiple seeds are only available for the smaller, non-deduped models
        if deduped or size in ["1b", "1.4b", "2.8b", "6.9b", "12b"]:
            seeds = [0]
        else:
            seeds = list(range(10))

        ckpts = [
            Checkpoint(self.get_model_name(s), revision=f"step{t}", step=t, seed=s)
            for s in seeds
            for t in self.all_steps
        ]
        kwargs.setdefault("model_class", "AutoModelForCausalLM")
        super().__init__(ckpts, **kwargs)
        self._checkpoints = self.filter(step=step or None, seed=seed or None)._checkpoints

    @property
    def name(self) -> str:
        return f"Pythia {self.size}" + (" deduped" if self.deduped else "")

    @staticmethod
    def last_step() -> int:
        """Last step of training."""
        return 143000

    @classmethod
    def final_checkpoints(cls, **kwargs) -> "PythiaCheckpoints":
        return cls(step=[cls.last_step()], **kwargs)

    def get_model_name(self, seed: int) -> str:
        """Name of the repository on the HF hub."""
        name = f"EleutherAI/pythia-{self.size}"
        if self.deduped:
            name += "-deduped"
        if seed != 0:
            name += f"-seed{seed}"
        return name
