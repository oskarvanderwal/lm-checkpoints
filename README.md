# 🤖🚩lm-checkpoints
> Simple library for dealing with language model checkpoints.

*lm-checkpoints* should make it easier to work with intermediate training checkpoints that are provided for some language models (LMs). This allows you to iterate over the training steps, to define different subsets, to automatically clear the cache for previously seen checkpoints, etc. Nothing fancy, a simple wrapper for 🤗 models that should make it easier to study the training dynamics of LMs.

Install using `pip install lm-checkpoints`.

## Checkpoints
Currently implemented for the following models on HuggingFace:
- [The Pythia models](https://github.com/EleutherAI/pythia)
- [MultiBERTs](https://huggingface.co/google/multiberts-seed_0)

## Example
Say you want to compute some metrics for all model checkpoints of Pythia 160m, but only seed 0.

```
from lm_checkpoints import PythiaCheckpoints

for ckpt in PythiaCheckpoints(size=160,seed=[0]):
    # Do something with ckpt.model, ckpt.config or ckpt.tokenizer
    print(ckpt.config)
```

Or if you only want to load steps `0, 1, 2, 4, 8, 16` for all available seeds:
```
from lm_checkpoints import PythiaCheckpoints

for ckpt in PythiaCheckpoints(size=160,step=[0, 1, 2, 4, 8, 16]):
    # Do something with ckpt.model, ckpt.config or ckpt.tokenizer
    print(ckpt.config)
```

Alternatively, you may want to load all final checkpoints of MultiBERTs:
```
from lm_checkpoints import MultiBERTCheckpoints

for ckpt in MultiBERTCheckpoints.final_checkpoints():
    # Do something with ckpt.model, ckpt.config or ckpt.tokenizer
    print(ckpt.config)
```

### Loading "chunks" of checkpoints for parallel computations
It is possible to split the checkpoints in N "chunks", e.g., useful if you want to run computations in parallel:
```
chunks = []
for chunk in self.checkpoints.split(N):
    chunks.append(chunk)
```

### Dealing with limited disk space
In case you don't want the checkpoints to fill up your disk space, use `clean_cache=True` to delete earlier checkpoints when iterating over these models (NB: You have to redownload these if you run it again!):
```
from lm_checkpoints import PythiaCheckpoints

for ckpt in PythiaCheckpoints(size=14,clean_cache=True):
    # Do something with ckpt.model or ckpt.tokenizer
```
