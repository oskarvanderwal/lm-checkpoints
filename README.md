# lm-checkpoints 🤖🚩
> Simple library for dealing with language model checkpoints for studying training dynamics.

**lm-checkpoints** should make it easier to work with intermediate training checkpoints that are provided for some language models (LMs), like MultiBERTs and Pythia. This library allows you to iterate over the training steps, to define different subsets, to automatically clear the cache for previously seen checkpoints, etc. Nothing fancy, simply a wrapper for 🤗 models that should make it easier to study their training dynamics.

Install using `pip install lm-checkpoints` or with [uv](https://docs.astral.sh/uv/):
```bash
uv add lm-checkpoints
```

## Checkpoints
Currently implemented for the following models on HuggingFace:
- [Pythia](https://github.com/EleutherAI/pythia) - 14m to 12b, multiple seeds
- [MultiBERTs](https://huggingface.co/google/multiberts-seed_0) - BERT with 5 seeds

## Usage examples
> [!NOTE]  
> The term seed here refers to the seed of the training run, not a random seed you would set for e.g., doing the evaluations.

Say you want to compute some metrics for all model checkpoints of Pythia 160m, but only seed 0.

```python
from lm_checkpoints import PythiaCheckpoints

for ckpt in PythiaCheckpoints(size="160m",seed=[0]):
    # Do something with ckpt.model, ckpt.config or ckpt.tokenizer
    print(ckpt.config)
```

Or if you only want to load steps `0, 1, 2, 4, 8, 16` for all available seeds:
```python
from lm_checkpoints import PythiaCheckpoints

for ckpt in PythiaCheckpoints(size="1.4b",step=[0, 1, 2, 4, 8, 16]):
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

### Device selection
Load models on CPU, CUDA, or Apple Silicon (MPS):
```python
ckpts = PythiaCheckpoints(size="14m", step=[0], seed=[0], device="cuda")  # or "cpu", "mps"
```

### Loading "chunks" of checkpoints for parallel computations
It is possible to split the checkpoints in N "chunks", e.g., useful if you want to run computations in parallel:
```python
chunks = []
checkpoints = PythiaCheckpoints(size="70m",seed=[0])
for chunk in checkpoints.split(N):
    chunks.append(chunk)
```

### Cache management
Control how checkpoints are cached using `cache_policy`:

```python
from lm_checkpoints import PythiaCheckpoints

# "keep" (default): Standard HF caching, keep all downloads
for ckpt in PythiaCheckpoints(size="14m", cache_policy="keep"):
    ...

# "previous": Delete previous checkpoint after loading next one
for ckpt in PythiaCheckpoints(size="14m", cache_policy="previous"):
    ...
```

#### Custom cache directory
Isolate checkpoints in a project-specific location:
```python
ckpts = PythiaCheckpoints(
    size="14m",
    cache_dir="/scratch/$USER/hf-lm-checkpoints"
)
```

#### Offline mode
Use `local_files_only=True` to only load from local cache (no downloads):
```python
ckpts = PythiaCheckpoints(size="14m", local_files_only=True)
```

### Applying evaluation functions
Use `map()` to apply any function to all checkpoints:
```python
from lm_checkpoints import PythiaCheckpoints

ckpts = PythiaCheckpoints(size="14m", step=[0, 1000, 2000], seed=[0])

# Simple iteration with results
for result in ckpts.map(lambda ckpt: my_eval(ckpt.model)):
    print(result)

# With checkpoint metadata
for config, result in ckpts.map(my_eval, include_config=True):
    print(f"Step {config['step']}: {result}")

# Collect all results as list of dicts
results = ckpts.map_collect(lambda ckpt: my_eval(ckpt.model))
# [{"model_name": "...", "step": 0, "seed": 0, "result": ...}, ...]
```

### Converting steps to tokens
Each checkpoint class provides methods to convert between training steps and tokens seen:
```python
ckpts = PythiaCheckpoints(size="14m", step=[1000], seed=[0])

# Get tokens seen at step 1000
tokens = ckpts.step_to_tokens(1000)  # ~2.1B tokens

# Find nearest step for a given token count
step = ckpts.tokens_to_step(5_000_000_000)  # Returns nearest available step
```
### Evaluation
Use `map()` with any evaluation framework:

```python
from lm_checkpoints import PythiaCheckpoints

ckpts = PythiaCheckpoints(size="14m", step=[0, 1000, 2000], seed=[0], device="cuda")

# With lm-evaluation-harness
from lm_eval.models.huggingface import HFLM
import lm_eval

for ckpt in ckpts:
    results = lm_eval.simple_evaluate(
        model=HFLM(pretrained=ckpt.model, tokenizer=ckpt.tokenizer),
        tasks=["hellaswag"],
    )

# With Inspect AI
from inspect_ai import eval
from inspect_ai.model import HuggingFaceModel

for ckpt in ckpts:
    eval(tasks, model=HuggingFaceModel(model=ckpt.model, tokenizer=ckpt.tokenizer))

# Or any custom evaluation
results = ckpts.map_collect(my_eval_function)
```

See `examples/` for ready-to-use evaluation scripts.