# 🤖🚩lm-checkpoints

> Simple library for dealing with language model checkpoints to study training dynamics.

**lm-checkpoints** should make it easier to work with intermediate training checkpoints that are provided for some language models (LMs), like MultiBERTs and Pythia. This library allows you to iterate over the training steps, to define different subsets, to automatically clear the cache for previously seen checkpoints, etc. Nothing fancy, simply a wrapper for 🤗 models that should make it easier to study their training dynamics.

Install using `pip install lm-checkpoints`.

## Checkpoints
Currently implemented for the following models on HuggingFace:
- [The Pythia models](https://github.com/EleutherAI/pythia)
- [MultiBERTs](https://huggingface.co/google/multiberts-seed_0)

## Example
Say you want to compute some metrics for all model checkpoints of Pythia 160m, but only seed 0.

```python
from lm_checkpoints import PythiaCheckpoints

for ckpt in PythiaCheckpoints(size=160,seed=[0]):
    # Do something with ckpt.model, ckpt.config or ckpt.tokenizer
    print(ckpt.config)
```

Or if you only want to load steps `0, 1, 2, 4, 8, 16` for all available seeds:
```python
from lm_checkpoints import PythiaCheckpoints

for ckpt in PythiaCheckpoints(size=160,step=[0, 1, 2, 4, 8, 16]):
    # Do something with ckpt.model, ckpt.config or ckpt.tokenizer
    print(ckpt.config)
```

Alternatively, you may want to load all final checkpoints of MultiBERTs:
```python
from lm_checkpoints import MultiBERTCheckpoints

for ckpt in MultiBERTCheckpoints.final_checkpoints():
    # Do something with ckpt.model, ckpt.config or ckpt.tokenizer
    print(ckpt.config)
```

### Loading "chunks" of checkpoints for parallel computations
It is possible to split the checkpoints in N "chunks", e.g., useful if you want to run computations in parallel:
```python
chunks = []
checkpoints = PythiaCheckpoints(size=160,seed=[0])
for chunk in checkpoints.split(N):
    chunks.append(chunk)
```

### Dealing with limited disk space
In case you don't want the checkpoints to fill up your disk space, use `clean_cache=True` to delete earlier checkpoints when iterating over these models (NB: You have to redownload these if you run it again!):
```python
from lm_checkpoints import PythiaCheckpoints

for ckpt in PythiaCheckpoints(size=14,clean_cache=True):
    # Do something with ckpt.model or ckpt.tokenizer
```
### Evaluating checkpoints using lm-evaluation-harness
If you install lm-checkpoints with the `eval` option (`pip install "lm-checkpoints[eval]"`), you can use the `evaluate` function to run [lm-evaluation-harness]() for all checkpoints:
```python
from lm_checkpoints import evaluate, PythiaCheckpoints

ckpts = PythiaCheckpoints(size=14, step=[0, 1, 2, 4], seed=[0], device="cuda")

evaluate(
    ckpts,
    tasks=["triviaqa", "crows_pairs_english"],
    output_dir="test_results",
    log_samples=True,
    skip_if_exists=True,
#    limit=5, # For testing purposes!
)
```

Or you can use the `evaluate_checkpoints` script:
```bash
evaluate_checkpoints pythia --output test_results --size 14 --seed 1 --step 0 1 2 --tasks blimp crows_pairs_english --device cuda --skip_if_exists
```

Both examples will create a subdirectory structure in `test_results/` for each model and step. This will contain a results json file (e.g., `results_crows_pairs_english,triviaqa.json`), and if using the `--log_samples` option, a json file containing the LM responses to the individual test items for each task (e.g., `samples_triviaqa.json`).