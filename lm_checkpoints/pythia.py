from lm_checkpoints import AbstractCheckpoints, Checkpoint
from itertools import product
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from typing import List, Dict


class PythiaCheckpoints(AbstractCheckpoints):
    """Class for iterating over Pythia checkpoints"""

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
            size (str): Model size. Defaults to "14m".
            step (List[int], optional): List of steps to consider, uses all available steps if not specified.
            seed (List[int], optional): List of seeds to consider, uses all available seeds if not specified.
            deduped (bool, optional): Specifies whether to use the deduped version of Pythia. Defaults to False.
        """
        super().__init__(**kwargs)

        self.deduped = deduped

        self._size = ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
        assert size in self._size
        self.size = size

        if deduped:
            raise NotImplementedError

        # Different seeds only available for the smaller models
        if self.size in ["1b", "1.4b", "2.8b", "6.9b", "12b"]:
            self._seeds = [0]
        else:
            self._seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        if seed:
            assert set(seed).issubset(set(self._seeds))
            self.seeds = seed
        else:
            self.seeds = self._seeds

        self._steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000, 144000, 1000))
        if step:
            assert set(step).issubset(set(self._steps))
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

    def get_checkpoint(self, seed, step) -> Checkpoint:
        model_name = self.get_model_name(seed)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=f"step{step}",
        )

        model = GPTNeoXForCausalLM.from_pretrained(
            model_name,
            revision=f"step{step}",
            low_cpu_mem_usage=self.low_cpu_mem_usage,
        )
        model.eval()
        model = model.to(self.device)

        commit_hash = self.get_revision_hash(model_name, f"step{step}")

        return Checkpoint(
            model, tokenizer=tokenizer, model_name=model_name, seed=seed, step=step, commit_hash=commit_hash, revision=f"step{step}"
        )