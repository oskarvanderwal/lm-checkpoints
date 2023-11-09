from lm_checkpoints import AbstractCheckpoints, Checkpoint
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from itertools import product


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
        return f"MultiBERTs"

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
    def checkpoints(self) -> list[dict[str, int]]:
        """Returns all step and seed combinations that make up the checkpoints.

        Returns:
            list[dict[str, int]]: List of dicts (seed, step) describing each checkpoint.
        """
        return list(
            {"seed": p[0], "step": p[1]} for p in product(self.seeds, self.steps)
        )

    def __len__(self):
        return len(self.seeds) * len(self.steps)

    def get_checkpoint(self, seed, step) -> Checkpoint:
        model_name = self.get_model_name(step, seed)
        config = AutoConfig.from_pretrained(model_name)

        # Necessary here?
        config.output_scores = True

        tokenizer = AutoTokenizer.from_pretrained(model_name, device=self.device,)

        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            config=config,
            low_cpu_mem_usage=self.low_cpu_mem_usage,  # https://huggingface.co/docs/transformers/main_classes/model#large-model-loading
        )
        model.eval()
        model = model.to(self.device)

        return Checkpoint(
            model, tokenizer=tokenizer, model_name=model_name, seed=seed, step=step
        )
