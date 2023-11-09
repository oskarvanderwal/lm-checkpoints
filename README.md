# lm-checkpoints
Simple library for dealing with language model checkpoints.

## Example
Say you want to compute some metrics for all model checkpoints of Pythia 160m, but only seed 0.

```
from lm_checkpoints import PythiaCheckpoints

for ckpt in PythiaCheckpoints(size=160,seed=[0]):
    # Do something with ckpt.model or ckpt.tokenizer
```

Or if you only want to load steps `0, 1, 2, 4, 8, 16` for all available seeds:
```
from lm_checkpoints import PythiaCheckpoints

for ckpt in PythiaCheckpoints(size=160,step=[0, 1, 2, 4, 8, 16]):
    # Do something with ckpt.model or ckpt.tokenizer
```

Alternatively, you may want to load all final checkpoints of MultiBERTs:
```
from lm_checkpoints import MultiBERTCheckpoints

for ckpt in MultiBERTCheckpoints.final_checkpoints():
    # Do something with ckpt.model or ckpt.tokenizer
```