from abc import ABC, abstractmethod
from itertools import product
import torch
from pathlib import Path
import numpy as np

def records_to_list(list_of_dicts: list[dict[str, int]] | dict[str, int]):
    """Transform a list of dictionaries to a dictionary of lists.
    If list_of_dicts is a dictionary, it will simply make a list of each values.
    From: https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists

    Args:
        list_of_dicts (list[dict[str, int]] | dict[str, int]): List of dictionaries, assuming each dictionary has the same keys.

    Returns:
        dict[list]: Dictionary of lists.
    """
    if not isinstance(list_of_dicts, list):
        list_of_dicts = [list_of_dicts]
    return {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}

def chunk(L, n): 
    '''
    Partition L into n chunks using every item in L and 
    such that the resulting chunks differ in size by at 
    most one element.
    
    >>> L = ['a', 'b', 'c', 'd']
    ['a', 'b', 'c', 'd']
    >>> chunk(L, 2)
    [['a', 'b'], ['c', 'd']]
    >>> chunk(L, 3)
    [['a'], ['b', 'c'], ['d']]
    >>> chunk(L, 4)
    [['a'], ['b'], ['c'], ['d']]
    >>> chunk(L, 5)
    [['a'], ['b'], [], ['c'], ['d']]
    '''
    size = len(L) / float(n) 
    I = lambda i: int(round(i))
    return [ L[I(size*i):I(size*(i+1))] for i in range(n) ]

class AbstractCheckpoints(ABC):
    """Abstract class for iterating over model checkpoints"""
    def __init__(self, cache_dir = None, device="cpu", **kwargs,):
        self._device = device
        if device == "cpu":
            self.device = torch.device("cpu")
        elif device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "mps":
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def split(self, n):
        """Convenience function for splitting checkpoints for e.g. parallel computing.
        If n > m (number of total checkpoints), it will return a list of m checkpoints objects instead.
        """
        total_length = len(self)
        chunk_indices = np.array_split(range(total_length), n)

        ckpts = []
        for ci in chunk_indices:
            if len(ci) > 0:
                start = ci[0].item()
                end = ci[-1].item()
                cfg = self.config
                cfg.update({k: set(v) for k,v in records_to_list(self.checkpoints[start:end+1]).items()})
                #ckpts.append(self.__class__(seeds=set(cfg["seed"]), steps=set(cfg["step"])))
                ckpts.append(self.__class__(**cfg))
        return ckpts
    
    @property
    @abstractmethod
    def name():
        pass
    
    @property
    @abstractmethod
    def checkpoints(self):
        pass

    @staticmethod
    @abstractmethod
    def last_step():
        pass

    @abstractmethod
    def get_checkpoint(self):
        pass

    @classmethod
    def final_checkpoints(cls, **kwargs):
        return cls(step=[cls.last_step()], **kwargs)

    @abstractmethod
    def __len__(self):
        pass

    def __iter__(self):
        for cfg in self.checkpoints:
            ckpt = self.get_checkpoint(**cfg)
            yield ckpt

class Checkpoint:
    """Convenience class for representing a checkpoint.
    Each checkpoint should at least have a model and model_name.
    """    
    def __init__(self, model, model_name, tokenizer=None, **kwargs):
        self._model = model
        self._tokenizer = tokenizer
        self._config = {"model_name": model_name}
        self._config.update(kwargs)

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        return self._model
    
    @property
    def tokenizer(self):
        if not self._tokenizer:
            raise ValueError("This checkpoint has no corresponding tokenizer.")
        return self._tokenizer