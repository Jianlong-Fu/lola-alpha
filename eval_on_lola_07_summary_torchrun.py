"""Summary evaluation, modified for fixed standalone inference."""

import argparse
from collections import deque
from datetime import timedelta
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch

from lola_alpha.evaluation_utils import (
    INITIAL_STATE_HASH, get_env_state_for_initial_condition, seed_everything, sequence_seed,
)
from lola_alpha.processor import Processor, SummaryHistory, unnormalize_actions


def make_calvin_env(validation_path, show_gui=False):
    from calvin_env.envs.play_table_env import get_env
    from calvin_env.utils.utils import get_egl_device_id

    if torch.cuda.is_available():
        os.environ["EGL_VISIBLE_DEVICES"] = str(get_egl_device_id(torch.cuda.current_device()))
    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": [],
        "state_obs": ["robot_obs"], "actions": ["rel_actions"], "language": ["language"],
    }
    return get_env(validation_path, obs_space=observation_space, show_gui=show_gui)


def rollout(env, policy, processor, history, task_oracle, subtask, annotation, episode_length):
    observation = env.get_obs()
    history.begin_subtask(observation["robot_obs"])
    actions = deque()
    start_info = env.get_info()
    for _ in range(episode_length):
        if not actions:
            batch = processor(observation, annotation, history)
            predicted = policy.predict_action_chunk(batch)
            actions.extend(unnormalize_actions(predicted, policy.normalization)[0, :8].detach().cpu().numpy())
        observation, _, _, info = env.step(actions.popleft())
        if task_oracle.get_task_info_for_set(start_info, info, {subtask}):
            history.complete_subtask(annotation)
            return True
        history.record_nonterminal_state(observation["robot_obs"])
    return False


def evaluate_sequence(env, policy, processor, history, task_oracle, sequence, annotations,
                      initial_state_converter, episode_length):
    robot_obs, scene_obs = initial_state_converter(sequence["initial_state"])
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    history.reset()
    successes = 0
    for subtask in sequence["action_sequence"]:
        if not rollout(env, policy, processor, history, task_oracle, subtask,
                       annotations[subtask][0], episode_length):
            break
        successes += 1
    return successes


def summarize(results):
    values = list(results.values())
    count = len(values)
    return {
        "num_sequences": count,
        "mean_completed_tasks": sum(values) / count if count else None,
        "success_rates": {
            f"{level}/5": sum(value >= level for value in values) / count if count else None
            for level in range(1, 6)
        },
    }


def load_eval_sequences(path, num_sequences):
    if num_sequences <= 0:
        raise ValueError("num_sequences must be positive")
    if path is not None:
        sequences = json.loads(Path(path).read_text())
    else:
        if num_sequences > 1000:
            raise ValueError("CALVIN generation supports at most 1000 evaluation sequences")
        from calvin_agent.evaluation.multistep_sequences import get_sequences

        sequences = {
            f"seq_{index}": {"initial_state": initial_state, "action_sequence": list(tasks)}
            for index, (initial_state, tasks) in enumerate(get_sequences(1000, num_workers=4))
        }
    missing = [f"seq_{index}" for index in range(num_sequences) if f"seq_{index}" not in sequences]
    if missing:
        raise ValueError(f"Missing evaluation sequence: {missing[0]}")
    return {f"seq_{index}": sequences[f"seq_{index}"] for index in range(num_sequences)}


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate the fixed LoLA-alpha model on CALVIN")
    parser.add_argument("--checkpoint_path", required=True, help="Exported safetensors checkpoint")
    parser.add_argument("--vlm_path", required=True, help="Local Cosmos3-Nano model and processor")
    parser.add_argument("--dataset_dir", required=True, help="CALVIN directory containing validation/")
    parser.add_argument("--calvin_config_root", required=True, help="CALVIN configuration directory containing annotations/ and callbacks/")
    sequence_source = parser.add_mutually_exclusive_group()
    sequence_source.add_argument("--get_sequences", action="store_true",
                                 help="Use CALVIN get_sequences(1000); this is the default without a JSON path")
    sequence_source.add_argument("--eval_sequences_path", help="Use an existing sequence JSON instead of CALVIN generation")
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--num_sequences", type=int, default=1000, help="Evaluate the first N sequences (default: 1000)")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episode_length", type=int, default=360, help="Maximum environment steps per subtask")
    parser.add_argument("--env_factory", default="eval_on_lola_07_summary_torchrun:make_calvin_env",
                        help="module:callable accepting (validation_path, show_gui=False)")
    return parser


def main():
    args = build_parser().parse_args()
    if args.num_sequences <= 0 or args.episode_length <= 0:
        raise ValueError("num_sequences and episode_length must be positive")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    destination = Path(args.eval_dir)
    if (destination / "results_all.json").exists():
        raise FileExistsError(f"Use a new evaluation directory: {destination}")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        os.environ["EGL_DEVICE_ID"] = str(local_rank)
    env = None
    try:
        if world_size > 1:
            torch.distributed.init_process_group(
                backend="nccl" if device.type == "cuda" else "gloo",
                timeout=timedelta(minutes=60),
            )
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from lola_alpha.model import load_policy

        seed_everything(args.seed)
        config_root = Path(args.calvin_config_root)
        task_oracle = instantiate(OmegaConf.load(config_root / "callbacks/rollout/tasks/new_playtable_tasks.yaml"))
        annotations = OmegaConf.load(config_root / "annotations/new_playtable_validation.yaml")
        sequences = load_eval_sequences(args.eval_sequences_path, args.num_sequences)
        if rank == 0:
            from safetensors import safe_open

            checkpoint_metadata = None
            if Path(args.checkpoint_path).is_file():
                with safe_open(args.checkpoint_path, framework="pt", device="cpu") as checkpoint:
                    checkpoint_metadata = checkpoint.metadata()
            destination.mkdir(parents=True, exist_ok=True)
            runtime = {
                "arguments": vars(args), "world_size": world_size,
                "initial_state_hash": INITIAL_STATE_HASH,
                "checkpoint_metadata": checkpoint_metadata,
                "source_sha256": {
                    name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
                    for name in ("lola_alpha.model", "lola_alpha.processor", "lola_alpha.dit")
                },
                "versions": {name: importlib.metadata.version(name) for name in (
                    "lola-alpha", "torch", "torchvision", "transformers", "diffusers",
                    "numpy", "gym", "pybullet", "hydra-core", "omegaconf",
                )},
            }
            (destination / "evaluation_run.json").write_text(json.dumps(runtime, indent=2) + "\n")
        factory_module, factory_name = args.env_factory.split(":", 1)
        factory = getattr(importlib.import_module(factory_module), factory_name)
        env = factory(Path(args.dataset_dir) / "validation", show_gui=False)
        policy = load_policy(args.checkpoint_path, args.vlm_path, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            print(
                f"MODEL_LOADED allocated_GiB={torch.cuda.memory_allocated(device) / 2**30:.3f} "
                f"reserved_GiB={torch.cuda.memory_reserved(device) / 2**30:.3f} "
                f"peak_allocated_GiB={torch.cuda.max_memory_allocated(device) / 2**30:.3f}",
                flush=True,
            )
        processor = Processor(args.vlm_path, device, policy.normalization)
        history = SummaryHistory(policy.model.state_encoder.history_null_state, policy.normalization)
        results = {}
        for offset in range(0, args.num_sequences, world_size):
            index = offset + rank
            local = {}
            if index < args.num_sequences:
                key = f"seq_{index}"
                seed_everything(sequence_seed(args.seed, index))
                local[key] = evaluate_sequence(
                    env, policy, processor, history, task_oracle, sequences[key], annotations,
                    get_env_state_for_initial_condition, args.episode_length,
                )
            if world_size > 1:
                gathered = [None] * world_size if rank == 0 else None
                torch.distributed.gather_object(local, gathered, dst=0)
            else:
                gathered = [local]
            if rank == 0:
                for payload in gathered:
                    results.update(payload)
                print(f"{len(results)}/{args.num_sequences}: mean={sum(results.values()) / len(results):.4f}", flush=True)
        if rank == 0:
            destination = Path(args.eval_dir)
            destination.mkdir(parents=True, exist_ok=True)
            summary = summarize(results)
            (destination / "results_all.json").write_text(json.dumps(results, indent=2) + "\n")
            (destination / "summary_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2))
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()