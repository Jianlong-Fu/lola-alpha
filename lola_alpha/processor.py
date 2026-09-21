"""LoLA preprocessing, modified for standalone inference. Original notices: LICENSE."""

from collections import deque

import numpy as np
from PIL import Image
import torch


STATE_MEAN = (
    0.03989503970639779, -0.11164562497145028, 0.5003418895634643,
    1.0466402032966404, -0.0814058314045175, 1.5845884965835066, 0.0501676678533357,
)
STATE_STD = (
    0.1438797839580311, 0.09922964049446895, 0.05536321127712049,
    2.894355683052885, 0.13044955739834957, 0.571569553397238, 0.03089786611712422,
)
ACTION_MEAN = (
    0.0010866473540853921, 0.010116912107051651, -0.008357305807614474,
    -0.0026971741838111114, 0.0009072716730321385, -0.004831478727161528, -0.08337152609143453,
)
ACTION_STD = (
    0.24954695226515133, 0.20441892702966666, 0.2120733907251881,
    0.15890847888600657, 0.17389288232259603, 0.35485703353177755, 0.9965185357007168,
)


def normalize_state(raw_state):
    state = torch.as_tensor(raw_state, dtype=torch.float32).detach().cpu().reshape(-1)[:7]
    return (state - state.new_tensor(STATE_MEAN)) / (state.new_tensor(STATE_STD) + 1e-8)


def unnormalize_actions(actions):
    mean = torch.tensor(ACTION_MEAN, device=actions.device, dtype=torch.float32)
    std = torch.tensor(ACTION_STD, device=actions.device, dtype=torch.float32)
    output = actions * std + mean
    output[..., -1] = actions[..., -1].to(output.dtype)
    return output


def format_task(task, completed):
    if not completed:
        return task
    numbered = ", ".join(f"{index}. {annotation}" for index, annotation in enumerate(completed, 1))
    return f"Perform task: {task}. Completed: {numbered}"


def append_empty_token(batch):
    for name, fill in (("input_ids", 151645), ("attention_mask", 1), ("mm_token_type_ids", 0)):
        if name in batch:
            value = batch[name]
            batch[name] = torch.cat((value, value.new_full((value.shape[0], 1), fill)), dim=1)
    return batch


class Processor:
    def __init__(self, vlm_path, device):
        from transformers import AutoProcessor

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(str(vlm_path), local_files_only=True)
        image_processor = self.processor.image_processor
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            image_processor.size = {**size, "longest_edge": 230400, "shortest_edge": 16384}
        elif size is not None and hasattr(size, "longest_edge"):
            size.longest_edge = 230400
            size.shortest_edge = 16384
        else:
            image_processor.max_pixels = 230400
            image_processor.min_pixels = 16384
        self.processor.tokenizer.padding_side = "left"

    def __call__(self, observation, task, history):
        content = [
            {"type": "image", "image": Image.fromarray(np.asarray(observation["rgb_obs"][camera]))}
            for camera in ("rgb_static", "rgb_gripper")
        ]
        content.append({"type": "text", "text": format_task(task, history.completed)})
        encoded = self.processor.apply_chat_template(
            [[{"role": "user", "content": content}]],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            processor_kwargs={"text_kwargs": {"padding": True}},
        )
        batch = append_empty_token({key: encoded[key] for key in (
            "input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids",
        ) if key in encoded})
        batch = {key: value.to(self.device) for key, value in batch.items()}
        batch["observation.state"] = normalize_state(observation["robot_obs"]).unsqueeze(0).to(self.device)
        batch.update(history.build(self.device))
        return batch


class SummaryHistory:
    def __init__(self, null_state):
        self.null = null_state.detach().float().cpu()
        self.transition = deque(maxlen=32)
        self.task = deque(maxlen=32)
        self.reset()

    def reset(self):
        self.transition.clear()
        self.task.clear()
        self.transition_length = 0
        self.task_length = 0
        self.completed = []

    def begin_subtask(self, raw_state):
        self.task.append(normalize_state(raw_state))
        self.task_length = 1

    def record_nonterminal_state(self, raw_state):
        self.task.append(normalize_state(raw_state))
        self.task_length += 1

    def complete_subtask(self, annotation):
        self.transition.extend(self.task)
        self.transition_length += self.task_length
        self.task.clear()
        self.task_length = 0
        self.completed = (self.completed + [annotation])[-4:]

    def build(self, device):
        result = {}
        for segment, values, length in (
            ("transition", self.transition, self.transition_length),
            ("task", self.task, self.task_length),
        ):
            states = self.null.view(1, 7).expand(32, 7).clone()
            mask = torch.zeros(32, dtype=torch.bool)
            if values:
                states[-len(values):] = torch.stack(list(values))
                mask[-len(values):] = True
            result[f"hist_{segment}_states"] = states.unsqueeze(0).to(device)
            result[f"hist_{segment}_frame_mask"] = mask.unsqueeze(0).to(device)
            result[f"hist_{segment}_total_length"] = torch.tensor([length], device=device, dtype=torch.long)
        return result