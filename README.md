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
- [OLMo](https://huggingface.co/allenai/OLMo-7B) - AI2's open language models (1B-32B)
- [Tri](https://huggingface.co/trillionlabs/Tri-70B-Intermediate-Checkpoints) - Trillion Labs (0.5B-70B)
- [OpenMoE](https://github.com/XueFuzhao/OpenMoE) - Mixture-of-Experts models (8B, 34B)

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

### OLMo, Tri, and OpenMoE
```python
from lm_checkpoints import OLMoCheckpoints, TriCheckpoints, OpenMoECheckpoints

# OLMo 7B at specific training steps
for ckpt in OLMoCheckpoints(size="7b", step=[1000, 2000, 3000]):
    print(ckpt.config)

# Tri 70B intermediate checkpoints (tokens in billions)
for ckpt in TriCheckpoints(size="70b", step=[160, 320]):
    print(ckpt.config)

# OpenMoE 8B at different token checkpoints
for ckpt in OpenMoECheckpoints(size="8b", step=[400, 600, 800]):
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

# "bounded": Prune oldest models when cache exceeds limit
for ckpt in PythiaCheckpoints(size="1.4b", cache_policy="bounded", max_cache_size_gb=50.0):
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
### Evaluating checkpoints using lm-evaluation-harness
If you install lm-checkpoints with the `eval` option (`pip install "lm-checkpoints[eval]"`), you can use the `evaluate` function to run [lm-evaluation-harness]() for all checkpoints:
```python
from lm_checkpoints import evaluate, PythiaCheckpoints

ckpts = PythiaCheckpoints(size="14m", step=[0, 1, 2, 4], seed=[0], device="cuda")

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
# Pythia
evaluate_checkpoints pythia --output test_results --size 14m --seed 1 --step 0 1 2 --tasks blimp crows_pairs_english --device cuda --skip_if_exists

# OLMo
evaluate_checkpoints olmo --output test_results --size 7b --step 1000 2000 --tasks hellaswag --device cuda

# With cache management
evaluate_checkpoints pythia --output test_results --size 70m --tasks triviaqa --cache_policy bounded --max_cache_size_gb 50
```

Both examples will create a subdirectory structure in `test_results/` for each model and step. This will contain a results json file (e.g., `results_crows_pairs_english,triviaqa.json`), and if using the `--log_samples` option, a json file containing the LM responses to the individual test items for each task (e.g., `samples_triviaqa.json`).