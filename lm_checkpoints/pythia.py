from lm_checkpoints import AbstractCheckpoints, Checkpoint
from itertools import product
from transformers import AutoTokenizer, GPTNeoXForCausalLM

class PythiaCheckpoints(AbstractCheckpoints):
    """Class for iterating over Pythia checkpoints"""
    def __init__(self, size: int = 70, step: list[int] = None, seed: list[int] = None, deduped: bool = False, **kwargs) -> None:
        """Initialize the PythiaCheckpoints.

        Args:
            size (int): Model size. Defaults to 70.
            step (List[int], optional): List of steps to consider, uses all available steps if not specified.
            seed (List[int], optional): List of seeds to consider, uses all available seeds if not specified.
            deduped (bool, optional): Specifies whether to use the deduped version of Pythia. Defaults to False.
        """
        super().__init__(**kwargs)

        self.deduped = deduped
        self.size = size
        if deduped:
            raise NotImplementedError

        self._seeds = [0, 1, 2, 3, 4]
        if seed:
            assert set(seed).issubset(set(self._seeds))
            self.seeds = seed
        else:
            self.seeds = self._seeds

        self._steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000,144000,1000))
        if step:
            assert set(step).issubset(set(self._steps))
            self.steps = step
        else:
            self.steps = self._steps

    @property
    def name(self) -> str:
        return f"Pythia {self.size}m"

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
            return f"EleutherAI/pythia-{self.size}m"
        else:
            return f"EleutherAI/pythia-{self.size}m-seed{seed}"

    @property
    def checkpoints(self) -> list[dict[str, int]]:          
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
                    low_cpu_mem_usage=True,
                    )
        model.eval()

        return Checkpoint(model, tokenizer=tokenizer, model_name=model_name, seed=seed, step=step)