# lm-checkpoints 🤖🚩
> Simple library for dealing with language model checkpoints for studying training dynamics.

**lm-checkpoints** should make it easier to work with intermediate training checkpoints that are provided for some language models (LMs), like MultiBERTs and Pythia. This library allows you to iterate over the training steps, to define different subsets, to automatically clear the cache for previously seen checkpoints, etc. Nothing fancy, simply a wrapper for 🤗 models that should make it easier to study their training dynamics.

Install using `pip install lm-checkpoints`.

## Checkpoints
Built in for the following models on HuggingFace:
- [The Pythia models](https://github.com/EleutherAI/pythia) (`PythiaCheckpoints`, incl. `deduped=True` and the extra seeds)
- [MultiBERTs](https://huggingface.co/google/multiberts-seed_0) (`MultiBERTCheckpoints`)

Any other model works too, as long as its checkpoints are stored as **branches of a HF hub repo** (e.g., OLMo, LLM360 Amber, BLOOM intermediate checkpoints, or your own private repo) or as **subdirectories of a local directory** (e.g., the `checkpoint-500/` directories of the HF `Trainer`). See [other models and private checkpoints](#other-models-and-private-checkpoints).

## Usage examples
> [!NOTE]  
> The term seed here refers to the seed of the training run, not a random seed you would set for e.g., doing the evaluations.

Say you want to compute some metrics for all model checkpoints of Pythia 160m, but only seed 0.

```python
from lm_checkpoints import PythiaCheckpoints

for ckpt in PythiaCheckpoints(size="160m", seed=[0]):
    # Do something with ckpt.model or ckpt.tokenizer
    print(ckpt.step, ckpt.seed, ckpt.repo_id, ckpt.revision)
```

Or if you only want to load steps `0, 1, 2, 4, 8, 16` for all available seeds:
```python
from lm_checkpoints import PythiaCheckpoints

for ckpt in PythiaCheckpoints(size="1.4b", step=[0, 1, 2, 4, 8, 16]):
    # Do something with ckpt.model or ckpt.tokenizer
    print(ckpt.config)
```

Alternatively, you may want to load all final checkpoints of MultiBERTs:
```python
from lm_checkpoints import MultiBERTCheckpoints

for ckpt in MultiBERTCheckpoints.final_checkpoints():
    # Do something with ckpt.model or ckpt.tokenizer
    print(ckpt.config)
```

### Lazy loading
Nothing is downloaded or loaded until you access `ckpt.model` or `ckpt.tokenizer`, so it is cheap to list, filter and inspect checkpoints. `ckpt.config` holds the metadata of the checkpoint (repo, revision, step, seed, ...), not to be confused with `ckpt.model.config`.

Keyword arguments are passed on to `from_pretrained`, so you can for example load the models in half precision on a GPU:
```python
import torch
from lm_checkpoints import PythiaCheckpoints

ckpts = PythiaCheckpoints(size="1b", device="cuda", torch_dtype=torch.float16)
```

Want to load a checkpoint with another library (e.g., TransformerLens, vLLM)? Use `ckpt.repo_id` and `ckpt.revision`, or `ckpt.download()` to get the local path.

### Selecting checkpoints
Collections of checkpoints can be filtered, sliced, combined and split:
```python
ckpts = PythiaCheckpoints(size="70m")
early = ckpts.filter(step=[0, 1, 2, 4], seed=[0, 1])
first_ten = ckpts[:10]
final = ckpts.final()  # the last checkpoint of each seed
print(ckpts.steps, ckpts.seeds)
```

### Other models and private checkpoints
Discover the checkpoints stored as branches of a HF hub repo. The regex `pattern` should fully match the branch names; the group named `step` is parsed as the training step, and other named groups are stored in `ckpt.meta`:
```python
from lm_checkpoints import Checkpoints

ckpts = Checkpoints.from_hub(
    "allenai/OLMo-2-0425-1B",
    pattern=r"stage1-step(?P<step>\d+)-tokens(?P<tokens>\d+)B",
)
```
For private repos, log in with `huggingface-cli login` or pass `token=...`. Checkpoints of different training runs (seeds) can be combined with `+`:
```python
ckpts = Checkpoints.from_hub("me/run-a", seed=0) + Checkpoints.from_hub("me/run-b", seed=1)
```

Or use your own local checkpoints, e.g., from the HF `Trainer`:
```python
ckpts = Checkpoints.from_local("output/my-run", pattern=r"checkpoint-(?P<step>\d+)")
```

By default, models are loaded with `AutoModelForCausalLM`; use e.g. `model_class="AutoModelForMaskedLM"` for other architectures.

### Loading "chunks" of checkpoints for parallel computations
It is possible to split the checkpoints in N non-overlapping "chunks", e.g., useful if you want to run computations in parallel:
```python
checkpoints = PythiaCheckpoints(size="70m", seed=[0])
chunks = checkpoints.split(N)
```

### Dealing with limited disk space
In case you don't want the checkpoints to fill up your disk space, use `clean_cache=True` to delete each checkpoint from the HF cache once you move on to the next one (NB: You have to redownload these if you run it again!). Checkpoints that were already in your cache before are left alone:
```python
from lm_checkpoints import PythiaCheckpoints

for ckpt in PythiaCheckpoints(size="14m", clean_cache=True):
    # Do something with ckpt.model or ckpt.tokenizer
    ...
```
### Evaluating checkpoints with lm-evaluation-harness
lm-checkpoints does not wrap any evaluation framework, but since each checkpoint is just a repo (or path) and a revision, it is easy to combine with e.g. [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):
```python
import lm_eval
from lm_checkpoints import PythiaCheckpoints

for ckpt in PythiaCheckpoints(size="14m", seed=[0], step=[0, 1000, 143000]):
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={ckpt.repo_id},revision={ckpt.revision}",
        tasks=["lambada_openai"],
    )
```
See [`examples/evaluate_with_lm_eval.py`](examples/evaluate_with_lm_eval.py) for a version that skips checkpoints that were already evaluated (without downloading them) and saves the results together with the checkpoint's metadata.

## Development
```bash
pip install -e . pytest
pytest                # offline tests
pytest --run-network  # also tests that download small models from the HF hub
```
