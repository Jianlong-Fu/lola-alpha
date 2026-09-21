import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from safetensors.torch import load_file, save_file
import torch
from torch import nn

from eval_on_lola_07_summary_torchrun import (
    build_parser, evaluate_sequence, get_env_state_for_initial_condition,
    load_eval_sequences, rollout, sequence_seed, summarize,
)
from export_checkpoint import export_checkpoint, normalize_name
from lola_alpha.evaluation_utils import fnv1_32, seed_everything
from lola_alpha.model import ActionEncoder, ActionModel, StateEncoder, load_policy
from lola_alpha.processor import (
    ACTION_MEAN, STATE_MEAN, Processor, SummaryHistory, append_empty_token,
    format_task, normalize_state, unnormalize_actions,
)


def distributed_worker(directory):
    import eval_on_lola_07_summary_torchrun as evaluator

    policy = CountingPolicy()
    policy.model = types.SimpleNamespace(state_encoder=types.SimpleNamespace(history_null_state=torch.zeros(7)))
    env = Environment()
    env.close = lambda: None
    oracle = types.SimpleNamespace(get_task_info_for_set=lambda *args: True)
    modules = {}
    for name in ("hydra", "hydra.utils", "omegaconf", "calvin_agent", "calvin_agent.evaluation",
                 "calvin_agent.evaluation.utils", "test_environment"):
        modules[name] = types.ModuleType(name)
    modules["hydra.utils"].instantiate = lambda config: oracle
    modules["omegaconf"].OmegaConf = types.SimpleNamespace(load=lambda path: {"task": ["annotation"]})
    modules["calvin_agent.evaluation.utils"].get_env_state_for_initial_condition = lambda state: (None, None)
    modules["test_environment"].create = lambda *args, **kwargs: env
    arguments = ["eval", "--checkpoint_path", "unused", "--vlm_path", "unused", "--dataset_dir", directory,
                 "--calvin_config_root", directory, "--eval_sequences_path", str(Path(directory) / "sequences.json"),
                 "--eval_dir", str(Path(directory) / "results"), "--num_sequences", "3",
                 "--env_factory", "test_environment:create"]
    with patch.dict(sys.modules, modules), patch.object(sys, "argv", arguments), patch(
        "lola_alpha.model.load_policy", return_value=policy,
    ), patch.object(evaluator, "Processor", lambda *args: lambda *items: {}), patch.object(
        evaluator, "get_env_state_for_initial_condition", lambda state: (None, None),
    ):
        evaluator.main()


class Environment:
    def __init__(self):
        self.steps = 0
        self.actions = []

    def reset(self, **kwargs):
        self.steps = 0
        self.actions.clear()

    def get_obs(self):
        return {
            "robot_obs": np.full(15, self.steps, dtype=np.float32),
            "rgb_obs": {
                "rgb_static": np.zeros((200, 200, 3), dtype=np.uint8),
                "rgb_gripper": np.ones((84, 84, 3), dtype=np.uint8),
            },
        }

    def get_info(self):
        return {"step": self.steps}

    def step(self, action):
        self.steps += 1
        self.actions.append(action)
        return self.get_obs(), 0, False, self.get_info()


class CountingPolicy:
    def __init__(self):
        self.calls = 0

    def predict_action_chunk(self, batch):
        self.calls += 1
        return torch.arange(16).float()[None, :, None].expand(1, 16, 7)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.history = SummaryHistory(normalize_state(torch.zeros(7)))

    def test_dependency_lock_matches_metadata(self):
        root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((root / "pyproject.toml").read_text())
        locked = {}
        for line in (root / "requirements-lock.txt").read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            requirement = Requirement(line)
            name = canonicalize_name(requirement.name)
            self.assertNotIn(name, locked)
            self.assertIsNone(requirement.url)
            self.assertIsNone(requirement.marker)
            specifiers = list(requirement.specifier)
            self.assertEqual(len(specifiers), 1)
            self.assertEqual(specifiers[0].operator, "==")
            self.assertNotIn("*", specifiers[0].version)
            locked[name] = requirement.specifier
        declared = (
            metadata["build-system"]["requires"]
            + metadata["project"]["dependencies"]
            + metadata["project"]["optional-dependencies"]["eval"]
        )
        for value in declared:
            requirement = Requirement(value)
            self.assertEqual(locked[canonicalize_name(requirement.name)], requirement.specifier)
        self.assertNotIn("deepspeed", locked)

    def test_legacy_initial_state(self):
        initial = {
            "led": 1, "lightbulb": 1, "slider": "left", "drawer": "open",
            "red_block": "slider_left", "blue_block": "table", "pink_block": "slider_right",
            "grasped": 0,
        }
        self.assertEqual(fnv1_32(""), 0)
        self.assertEqual(fnv1_32("a"), 1627429043)
        self.assertEqual(fnv1_32(str(initial.values())), 2138410836)
        before = np.random.get_state()
        robot, scene = get_env_state_for_initial_condition(initial)
        after = np.random.get_state()
        self.assertEqual(robot.shape, (15,))
        np.testing.assert_array_equal(scene[[11, 17, 23]], [
            1.8859410064104645, 1.488143437705632, 1.4683986863194542,
        ])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_history_rollover_and_reset(self):
        self.history.begin_subtask(torch.zeros(7))
        for index in range(80):
            self.history.record_nonterminal_state(torch.full((7,), index + 1))
        self.history.complete_subtask("first")
        self.history.begin_subtask(torch.full((7,), 100))
        history = self.history.build("cpu")
        self.assertEqual(len(history), 6)
        self.assertEqual(history["hist_transition_total_length"].item(), 81)
        self.assertEqual(history["hist_transition_frame_mask"].sum().item(), 32)
        self.assertEqual(history["hist_task_total_length"].item(), 1)
        self.assertTrue(torch.equal(history["hist_task_states"][0, 0], self.history.null))
        self.history.reset()
        self.assertEqual(self.history.completed, [])
        self.assertEqual(self.history.transition_length, 0)
        self.assertEqual(len(self.history.transition), 0)

    def test_completed_annotations_capped(self):
        for index in range(6):
            self.history.begin_subtask(torch.zeros(7))
            self.history.complete_subtask(str(index))
        self.assertEqual(self.history.completed, ["2", "3", "4", "5"])

    def test_text_and_empty_token(self):
        self.assertEqual(format_task("close drawer", []), "close drawer")
        self.assertEqual(format_task("close drawer", ["open drawer"]),
                         "Perform task: close drawer. Completed: 1. open drawer")
        batch = append_empty_token({
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "mm_token_type_ids": torch.tensor([[1, 0]]),
        })
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 151645]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1]])
        self.assertEqual(batch["mm_token_type_ids"].tolist(), [[1, 0, 0]])

    def test_normalization_and_gripper(self):
        self.assertTrue(torch.equal(normalize_state(STATE_MEAN), torch.zeros(7)))
        prediction = torch.zeros(1, 16, 7, dtype=torch.bfloat16)
        prediction[..., -1] = -1
        output = unnormalize_actions(prediction)
        self.assertTrue(torch.equal(output[0, 0, :6], torch.tensor(ACTION_MEAN[:6])))
        self.assertTrue(torch.all(output[..., -1] == -1))

    def test_single_item_processor(self):
        fake = types.SimpleNamespace(
            image_processor=types.SimpleNamespace(size={}),
            tokenizer=types.SimpleNamespace(padding_side="right"),
        )
        calls = []
        def encode(messages, **kwargs):
            calls.append((messages, kwargs))
            return {"input_ids": torch.tensor([[5, 6]]), "attention_mask": torch.ones(1, 2, dtype=torch.long),
                    "mm_token_type_ids": torch.tensor([[1, 0]]), "pixel_values": torch.zeros(2, 12),
                    "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]])}
        fake.apply_chat_template = encode
        with patch("transformers.AutoProcessor.from_pretrained", return_value=fake):
            processor = Processor("unused", "cpu")
        self.history.begin_subtask(torch.zeros(7))
        batch = processor(Environment().get_obs(), "open drawer", self.history)
        content = calls[0][0][0][0]["content"]
        self.assertEqual([item["image"].size for item in content[:2]], [(200, 200), (84, 84)])
        self.assertEqual(content[2]["text"], "open drawer")
        self.assertEqual(calls[0][1]["processor_kwargs"], {"text_kwargs": {"padding": True}})
        self.assertEqual(fake.image_processor.size, {"longest_edge": 230400, "shortest_edge": 16384})
        self.assertEqual(batch["observation.state"].shape, (1, 7))
        self.assertEqual(batch["hist_task_states"].shape, (1, 32, 7))
        self.assertEqual(batch["input_ids"].shape, batch["mm_token_type_ids"].shape)

    def test_rollout_chunk_and_success_boundary(self):
        env, policy = Environment(), CountingPolicy()
        oracle = types.SimpleNamespace(get_task_info_for_set=lambda start, current, tasks: current["step"] == 9)
        self.assertTrue(rollout(env, policy, lambda *args: {}, self.history, oracle, "task", "annotation", 20))
        self.assertEqual(policy.calls, 2)
        self.assertEqual([action[-1] for action in env.actions], [0, 1, 2, 3, 4, 5, 6, 7, 0])
        self.assertEqual(self.history.transition_length, 9)
        self.assertEqual(self.history.task_length, 0)
        torch.testing.assert_close(self.history.transition[-1], normalize_state(torch.full((7,), 8)))

    def test_sequence_stops_on_failure(self):
        env, policy = Environment(), CountingPolicy()
        oracle = types.SimpleNamespace(get_task_info_for_set=lambda start, current, tasks: "first" in tasks)
        sequence = {"initial_state": {}, "action_sequence": ["first", "second", "third"]}
        result = evaluate_sequence(
            env, policy, lambda *args: {}, self.history, oracle, sequence,
            {task: [task] for task in sequence["action_sequence"]}, lambda state: (None, None), 3,
        )
        self.assertEqual(result, 1)
        self.assertEqual(env.steps, 4)
        self.assertEqual(self.history.completed, ["first"])

    def test_calvin_sequence_generation(self):
        initial = {"led": 1, "lightbulb": 0}
        generated = [(initial, ("first", "second", "third", "fourth", "fifth"))] * 1000
        modules = {name: types.ModuleType(name) for name in (
            "calvin_agent", "calvin_agent.evaluation", "calvin_agent.evaluation.multistep_sequences",
        )}
        generator = Mock(return_value=generated)
        modules["calvin_agent.evaluation.multistep_sequences"].get_sequences = generator
        with patch.dict(sys.modules, modules):
            sequences = load_eval_sequences(None, 2)
            generator.assert_called_once_with(1000, num_workers=4)
            with self.assertRaisesRegex(ValueError, "at most 1000"):
                load_eval_sequences(None, 1001)
        self.assertEqual(list(sequences), ["seq_0", "seq_1"])
        self.assertEqual(list(sequences["seq_0"]["initial_state"].items()), list(initial.items()))
        self.assertEqual(sequences["seq_0"]["action_sequence"], list(generated[0][1]))

    def test_sequence_json_override(self):
        sequences = {f"seq_{index}": {"initial_state": {}, "action_sequence": ["task"] * 5} for index in range(3)}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"calvin_agent": None}):
            path = Path(directory) / "sequences.json"
            path.write_text(json.dumps(sequences))
            self.assertEqual(load_eval_sequences(path, 2), {key: sequences[key] for key in ("seq_0", "seq_1")})
            with self.assertRaisesRegex(ValueError, "Missing evaluation sequence: seq_3"):
                load_eval_sequences(path, 4)
            with self.assertRaises(FileNotFoundError):
                load_eval_sequences(Path(directory) / "missing.json", 2)

    def test_sequence_source_cli(self):
        arguments = ["--checkpoint_path", "unused", "--vlm_path", "unused", "--dataset_dir", "unused",
                     "--calvin_config_root", "unused", "--eval_dir", "unused"]
        parser = build_parser()
        self.assertIsNone(parser.parse_args(arguments).eval_sequences_path)
        self.assertTrue(parser.parse_args(arguments + ["--get_sequences"]).get_sequences)
        self.assertEqual(parser.parse_args(arguments + ["--eval_sequences_path", "sequences.json"]).eval_sequences_path,
                         "sequences.json")
        with patch("sys.stderr"), self.assertRaises(SystemExit) as failure:
            parser.parse_args(arguments + ["--get_sequences", "--eval_sequences_path", "sequences.json"])
        self.assertEqual(failure.exception.code, 2)

    def test_summary_seed_and_cli(self):
        summary = summarize({"seq_0": 5, "seq_1": 2, "seq_2": 0})
        self.assertEqual(summary["mean_completed_tasks"], 7 / 3)
        self.assertEqual(summary["success_rates"]["3/5"], 1 / 3)
        self.assertNotEqual(sequence_seed(0, 1), sequence_seed(0, 2))
        self.assertEqual(sequence_seed(0, 1), sequence_seed(0, 1))
        for index, expected in enumerate((16294208416658607535, 10451216379200822465, 10905525725756348110)):
            with self.subTest(index=index):
                self.assertEqual(sequence_seed(0, index), expected)
                self.assertEqual(sequence_seed(2**64, index), expected)
                self.assertEqual(sequence_seed(0, index + 2**64), expected)
        options = build_parser()._option_string_actions
        self.assertNotIn("--training_config", options)
        self.assertNotIn("--allow_summary_variant", options)

    def test_seed_everything_preserves_seed_width(self):
        with patch("lola_alpha.evaluation_utils.random.seed") as python_seed, patch(
            "lola_alpha.evaluation_utils.np.random.seed",
        ) as numpy_seed, patch("lola_alpha.evaluation_utils.torch.manual_seed") as torch_seed, patch(
            "lola_alpha.evaluation_utils.torch.cuda.is_available", return_value=False,
        ):
            for seed in (-1, 0, 2**32 - 1, 2**32, 2**64 - 1):
                with self.subTest(seed=seed):
                    seed_everything(seed)
                    python_seed.assert_called_with(seed)
                    numpy_seed.assert_called_with(seed % (2**32))
                    torch_seed.assert_called_with(seed)

    def test_fixed_model_topology(self):
        with torch.device("meta"):
            model = ActionModel()
        state = model.state_dict()
        self.assertEqual(len(state), 732)
        self.assertEqual(state["vlm_bridge.input_proj.weight"].shape, (2048, 4096))
        self.assertEqual(state["state_encoder.segment_pool.length_mlp.0.weight"].shape, (1024, 64))
        self.assertFalse(any(name.startswith("dit.single_blocks.11.ctx_shared_ff") for name in state))

    def test_empty_summary(self):
        encoder = StateEncoder().eval()
        self.history.begin_subtask(torch.zeros(7))
        with torch.no_grad():
            arm, grip, present = encoder(self.history.build("cpu"))
        self.assertEqual(arm.shape, (1, 2, 1024))
        self.assertTrue(torch.isfinite(arm).all())
        self.assertTrue(torch.isfinite(grip).all())
        self.assertFalse(present.item())

    def test_export_roundtrip(self):
        source = {"module.policy.model.weight": torch.randn(2, 3), "policy.vlm.weight": torch.randn(3, 4)}
        fake = types.ModuleType("deepspeed.utils.zero_to_fp32")
        fake.get_fp32_state_dict_from_zero_checkpoint = lambda *args, **kwargs: source
        modules = {"deepspeed": types.ModuleType("deepspeed"), "deepspeed.utils": types.ModuleType("deepspeed.utils"),
                   "deepspeed.utils.zero_to_fp32": fake}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            output = Path(directory) / "model.safetensors"
            export_checkpoint(Path(directory) / "step_1", output)
            restored = load_file(str(output))
            for name, value in source.items():
                self.assertTrue(torch.equal(restored[normalize_name(name)], value))

    def test_export_inference_dtypes(self):
        source = {
            "model.action_encoder.weight": torch.randn(2, 3),
            "model.state_encoder.weight": torch.randn(2, 3),
            "model.dit.weight": torch.randn(2, 3),
            "vlm.weight": torch.randn(2, 3),
        }
        fake = types.ModuleType("deepspeed.utils.zero_to_fp32")
        fake.get_fp32_state_dict_from_zero_checkpoint = lambda *args, **kwargs: source
        modules = {"deepspeed": types.ModuleType("deepspeed"), "deepspeed.utils": types.ModuleType("deepspeed.utils"),
                   "deepspeed.utils.zero_to_fp32": fake}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            output = Path(directory) / "model.safetensors"
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                export_checkpoint(Path(directory) / "step_1", output, True, "wrong")
            self.assertFalse(output.exists())
            export_checkpoint(Path(directory) / "step_1", output, True)
            restored = load_file(str(output))
            for name, value in source.items():
                dtype = torch.float32 if "encoder" in name else torch.bfloat16
                self.assertEqual(restored[name].dtype, dtype)
                torch.testing.assert_close(restored[name], value.to(dtype), rtol=0, atol=0)

    def test_load_dtype_and_trained_vlm(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.action_encoder = nn.Linear(2, 2)
                self.state_encoder = nn.Linear(2, 2)
                self.vlm_bridge = nn.Linear(2, 2)
        class TinyVLM(nn.Module):
            def __init__(self, config):
                super().__init__()
                if torch.empty(0).device.type != "cpu":
                    raise RuntimeError("VLM buffers must be initialized on CPU")
                self.register_buffer("rotary_frequency", 1.0 / (10000 ** (torch.arange(18).float() / 18)), persistent=False)
                self.weight = nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))
                self.language_model = nn.Module()
                self.language_model.layers = nn.ModuleList([nn.Identity() for _ in range(36)])
                self.language_model.norm = nn.Identity()
        weights = {f"model.{name}": value for name, value in TinyModel().state_dict().items()}
        weights["vlm.weight"] = torch.full((2, 2), 0.123456789)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(weights, str(path))
            with patch("lola_alpha.model.ActionModel", TinyModel), patch(
                "transformers.models.cosmos3_omni.modeling_cosmos3_omni.Cosmos3OmniModel", TinyVLM,
            ), patch("transformers.AutoConfig.from_pretrained", return_value=types.SimpleNamespace()):
                policy = load_policy(path, "unused", "cpu")
                if torch.cuda.is_available():
                    cuda_policy = load_policy(path, "unused", "cuda:0")
                    self.assertEqual(cuda_policy.vlm.weight.device.type, "cuda")
                    torch.testing.assert_close(
                        cuda_policy.vlm.rotary_frequency.cpu(), policy.vlm.rotary_frequency, rtol=0, atol=0,
                    )
                invalid = dict(weights)
                invalid["vlm.unexpected"] = torch.ones(1)
                save_file(invalid, str(path))
                with self.assertRaisesRegex(RuntimeError, "VLM checkpoint mismatch"):
                    load_policy(path, "unused", "cpu")
            self.assertFalse(policy.training)
            torch.testing.assert_close(policy.model.action_encoder.weight, weights["model.action_encoder.weight"], rtol=0, atol=0)
            self.assertEqual(policy.model.state_encoder.weight.dtype, torch.float32)
            self.assertEqual(policy.model.vlm_bridge.weight.dtype, torch.bfloat16)
            torch.testing.assert_close(policy.vlm.weight, weights["vlm.weight"].bfloat16(), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA precision check")
    def test_latent_projection_runs_in_fp32(self):
        class Bridge(nn.Module):
            def forward(self, features):
                return features[:, :-1], features[:, -1]
        class HistoryEncoder(nn.Module):
            def forward(self, history):
                features = history["features"]
                summary = features.new_zeros(1, 2, 1024)
                return summary, summary, torch.zeros(1, dtype=torch.bool, device=features.device)
        class ToyDiT(nn.Module):
            def __init__(self):
                super().__init__()
                for name in ("vlm_start_emb", "vlm_end_emb", "hist_start_emb", "hist_end_emb", "previous_task_end_emb"):
                    setattr(self, name, nn.Parameter(torch.zeros(1, 1, 1024)))
                self.arm_out_proj = nn.Linear(1024, 48)
                self.gripper_out_proj = nn.Linear(1024, 8)
            def forward(self, arm, grip, *args):
                return arm, grip
        model = ActionModel.__new__(ActionModel)
        nn.Module.__init__(model)
        model.vlm_bridge = Bridge()
        model.state_encoder = HistoryEncoder()
        model.action_encoder = ActionEncoder().cuda()
        model.dit = ToyDiT().cuda().bfloat16()
        model.arm_dit_to_latent = nn.Linear(1024, 256).cuda().bfloat16()
        model.grip_dit_to_latent = nn.Linear(1024, 128).cuda().bfloat16()
        dtypes = []
        handle = model.arm_dit_to_latent.register_forward_hook(lambda module, inputs, output: dtypes.append(output.dtype))
        features = torch.randn(1, 3, 1024, device="cuda", dtype=torch.bfloat16)
        try:
            actions = model.sample_actions(features, features.new_zeros(1, 7), {"features": features})
        finally:
            handle.remove()
        self.assertEqual(dtypes, [torch.float32] * 3)
        self.assertEqual(actions.shape, (1, 16, 7))
        self.assertTrue(torch.isfinite(actions).all())

    @unittest.skipUnless(importlib.util.find_spec("deepspeed"), "Optional export dependency")
    def test_real_zero_export(self):
        import deepspeed

        weights = {"model.weight": torch.arange(6).reshape(2, 3).float(), "vlm.weight": torch.arange(5).float()}
        buffer_name = "model.state_encoder.history_null_state"
        with tempfile.TemporaryDirectory() as directory:
            tag = Path(directory) / "step_1"
            tag.mkdir()
            for rank in range(2):
                partitions = []
                for value in weights.values():
                    flat = value.flatten()
                    partition_size = (flat.numel() + 1) // 2
                    padded = torch.nn.functional.pad(flat, (0, partition_size * 2 - flat.numel()))
                    partitions.append(padded[rank * partition_size:(rank + 1) * partition_size])
                torch.save({"optimizer_state_dict": {
                    "zero_stage": 3, "partition_count": 2, "fp32_flat_groups": [torch.cat(partitions)],
                }}, tag / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")
                torch.save({
                    "module": {buffer_name: torch.arange(7).float()}, "buffer_names": [buffer_name],
                    "param_shapes": [{name: value.shape for name, value in weights.items()}],
                    "shared_params": {}, "ds_version": deepspeed.__version__,
                }, tag / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt")
            destination = Path(directory) / "model.safetensors"
            export_checkpoint(tag, destination)
            restored = load_file(str(destination))
            for name, value in weights.items():
                torch.testing.assert_close(restored[name], value, rtol=0, atol=0)
            torch.testing.assert_close(restored[buffer_name], torch.arange(7).float())

    def test_torchrun_uneven_cpu_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            sequences = {f"seq_{index}": {"initial_state": {}, "action_sequence": ["task"] * 5} for index in range(3)}
            (Path(directory) / "sequences.json").write_text(json.dumps(sequences))
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                           "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
            result = subprocess.run(
                [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                 str(Path(__file__).resolve()), "--smoke-rank", directory],
                env=environment, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((Path(directory) / "results/summary_metrics.json").read_text())
            self.assertEqual(summary["num_sequences"], 3)
            self.assertEqual(summary["mean_completed_tasks"], 5)
            results = json.loads((Path(directory) / "results/results_all.json").read_text())
            self.assertEqual(results, {"seq_0": 5, "seq_1": 5, "seq_2": 5})
            runtime = json.loads((Path(directory) / "results/evaluation_run.json").read_text())
            self.assertEqual(set(runtime["source_sha256"]), {"lola_alpha.model", "lola_alpha.processor", "lola_alpha.dit"})
            self.assertTrue(all(len(value) == 64 for value in runtime["source_sha256"].values()))


if __name__ == "__main__":
    if "--smoke-rank" in sys.argv:
        distributed_worker(sys.argv[-1])
    else:
        unittest.main()