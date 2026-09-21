"""Deterministic CALVIN scene initialization and per-sequence random seeding."""

import random

import fnvhash
import numpy as np
import torch


INITIAL_STATE_HASH = "fnv1_32_seed0_utf16le"
_UINT64_MODULUS = 2**64
_SPLITMIX64_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX64_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_SPLITMIX64_MULTIPLIER_2 = 0x94D049BB133111EB


def fnv1_32(value):
    """Match legacy pyhash string hashing: zero initial value and UTF-16-LE."""
    return fnvhash.fnv1_32(value.encode("utf-16-le"), hval_init=0)


def get_env_state_for_initial_condition(initial_condition):
    """Convert a CALVIN sequence's symbolic initial state into env.reset arrays.

    The ordered input dict describes lights, drawer/slider positions and block
    locations such as "table" or "slider_left". Returns robot_obs (15,) and
    scene_obs (24,) using CALVIN's fixed evaluation start pose and slot positions.

    A local RNG deterministically chooses table slots and block yaw angles,
    independently of the inference seed and without changing NumPy's global RNG.
    Preserve input field order: the legacy scene hash uses str(dict.values()).
    """
    robot_obs = np.array([
        0.02586889, -0.2313129, 0.5712808, 3.09045411, -0.02908596,
        1.50013585, 0.07999963, -1.21779124, 1.03987629, 2.11978254,
        -2.34205014, -0.87015899, 1.64119093, 0.55344928, 1.0,
    ])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    generator = np.random.RandomState(fnv1_32(str(initial_condition.values())))
    generator.shuffle(block_table)
    positions = {
        "slider_left": np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01]),
        "slider_right": np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01]),
    }
    # Slots 0:6 hold scene joints/lights; three xyz + Euler-angle block poses follow.
    scene_obs = np.zeros(24)
    scene_obs[0] = 0.28 if initial_condition["slider"] == "left" else 0.0
    scene_obs[1] = 0.22 if initial_condition["drawer"] == "open" else 0.0
    scene_obs[3] = 0.088 if initial_condition["lightbulb"] == 1 else 0.0
    scene_obs[4] = initial_condition["lightbulb"]
    scene_obs[5] = initial_condition["led"]
    # Table blocks take distinct shuffled slots in red/blue/pink order.
    for name, offset, table_index in (
        ("red_block", 6, 0),
        ("blue_block", 12, int(initial_condition["red_block"] == "table")),
        ("pink_block", 18, 1),
    ):
        scene_obs[offset:offset + 3] = positions.get(initial_condition[name], block_table[table_index])
        scene_obs[offset + 5] = generator.uniform(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    return robot_obs, scene_obs


def sequence_seed(seed, index):
    """Apply SplitMix64's finalizer to the historical seed/index combination.

    Constants and shifts follow https://prng.di.unimi.it/splitmix64.c.
    The initial XOR combines the run seed with the global sequence index,
    keeping each rollout's seed independent of worker rank and GPU count.
    """
    value = (seed % _UINT64_MODULUS) ^ ((index + _SPLITMIX64_INCREMENT) % _UINT64_MODULUS)
    value = ((value ^ (value >> 30)) * _SPLITMIX64_MULTIPLIER_1) % _UINT64_MODULUS
    value = ((value ^ (value >> 27)) * _SPLITMIX64_MULTIPLIER_2) % _UINT64_MODULUS
    return value ^ (value >> 31)


def seed_everything(seed):
    """Seed Python and Torch directly; NumPy's legacy RNG needs a 32-bit seed."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)