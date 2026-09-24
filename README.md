# lola-alpha

This is LoLA's CALVIN ABC-D review repository for standalone inference and evaluation.

## Install

The validated platform is Linux, Python 3.11, an NVIDIA RTX A6000, driver
595.71.05 and PyTorch 2.11.0+cu130. 
Run the following commands from the repository root:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps .
python -m pip check
```

Install CALVIN's simulation and evaluation dependencies in the same Python
environment, with both `calvin_env` and `calvin_agent` available. Prepare the
CALVIN robot/scene assets, working headless EGL support and built `egl_check`
utility.

## Evaluate

Set the following paths to your local resources:

```sh
export CALVIN_ENV_ROOT=/path/to/calvin/calvin_env
export CALVIN_MODELS_ROOT=/path/to/calvin/calvin_models
export MODEL_WEIGHTS=/path/to/our-lola.safetensors
export COSMOS_DIR=/path/to/Cosmos3-Nano
export CALVIN_DATA_DIR=/path/to/task_ABC_D
export CALVIN_CONFIG_DIR="$CALVIN_MODELS_ROOT/conf"
```

- `CALVIN_ENV_ROOT`: directory containing `calvin_env/` and `egl_check/`.
- `CALVIN_MODELS_ROOT`: directory containing `calvin_agent/` and `conf/`.
- `MODEL_WEIGHTS`: the LoLA-alpha safetensors checkpoint.
- `COSMOS_DIR`: local Cosmos3-Nano architecture, tokenizer and image-processor
  resources. Base weight shards are not needed when the checkpoint includes all
  `vlm.*` weights.
- `CALVIN_DATA_DIR`: contains `validation/`, including `.hydra/merged_config.yaml`.
- `CALVIN_CONFIG_DIR`: contains `callbacks/rollout/tasks/new_playtable_tasks.yaml`
  and `annotations/new_playtable_validation.yaml`.

Run on a single GPU:

```sh
PYTHONPATH="$CALVIN_ENV_ROOT:$CALVIN_MODELS_ROOT" CUDA_VISIBLE_DEVICES=0 lola-alpha-eval \
  --checkpoint_path "$MODEL_WEIGHTS" \
  --vlm_path "$COSMOS_DIR" \
  --dataset_dir "$CALVIN_DATA_DIR" \
  --calvin_config_root "$CALVIN_CONFIG_DIR" \
  --get_sequences \
  --eval_dir ./results-seed-1 \
  --num_sequences 1000 --episode_length 360 --seed 1
```

`--get_sequences` directly calls CALVIN's
`calvin_agent.evaluation.multistep_sequences.get_sequences(1000)`; it is also
the default when no sequence source is specified. No sequence JSON is required.
For a short run, `--num_sequences N` selects the first N of these fixed 1000
sequences, independently of the inference `--seed`. To reuse an existing JSON,
replace `--get_sequences` with `--eval_sequences_path /path/to/sequences.json`.

To reproduce the results below, repeat with seeds **1, 10, 100, 1000**, changing both `--seed` and `--eval_dir`.
Each run writes `results_all.json`, `summary_metrics.json` and
`evaluation_run.json`; use a new output directory for every run.

## Results

Evaluated with `lola-alpha-Calvin-ABC_D_0923.safetensors`, one GPU
per run, 1000 sequences per seed and a 360-step limit per subtask. Inference
uses 3 integration steps and executes 8 actions per prediction. Scores are
the mean number of completed tasks per sequence (maximum 5).

| Seed | Mean Completed Tasks |
| --- | ---: |
| 1 | 4.708 |
| 10 | 4.738 |
| 100 | 4.727 |
| 1000 | 4.704 |

Seeds 1/10/100/1000: **4.7193 +/- 0.0160** (mean +/- sample standard deviation).