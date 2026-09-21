import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory


def normalize_name(name):
    prefixes = ("module.", "_forward_module.")
    while name.startswith(prefixes):
        name = name.split(".", 1)[1]
    return name.removeprefix("policy.")


def export_checkpoint(checkpoint_path, output, inference_dtypes=False, expected_sha256=None):
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    from safetensors.torch import save_file
    import torch

    path = Path(checkpoint_path)
    if (path / "latest").is_file():
        root, tag = path, (path / "latest").read_text().strip()
    else:
        root, tag = path.parent, path.name
    state = get_fp32_state_dict_from_zero_checkpoint(
        str(root), tag=tag, exclude_frozen_parameters=True, lazy_mode=True,
    )
    digest = hashlib.sha256()
    tensors = {}
    for name in sorted(state):
        value = state[name].contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(memoryview(value.view(torch.uint8).reshape(-1).numpy()))
        normalized = normalize_name(name)
        if normalized in tensors:
            raise ValueError(f"Duplicate normalized tensor: {normalized}")
        if inference_dtypes and value.is_floating_point():
            dtype = torch.float32 if normalized.startswith(("model.action_encoder.", "model.state_encoder.")) else torch.bfloat16
            value = value.to(dtype=dtype)
        tensors[normalized] = value.clone()
    source_sha256 = digest.hexdigest()
    if expected_sha256 is not None and source_sha256 != expected_sha256:
        raise ValueError(f"Source checkpoint SHA256 mismatch: {source_sha256} != {expected_sha256}")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        result = Path(temporary) / "model.safetensors"
        save_file(tensors, str(result), metadata={
            "source_tag": tag, "source_state_sha256": source_sha256,
            "precision": "bf16_with_fp32_encoders" if inference_dtypes else "source",
        })
        result.replace(destination)
    print(f"Saved {len(tensors)} tensors to {destination}")


def main():
    parser = argparse.ArgumentParser(description="Export a ZeRO checkpoint to one safetensors file")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inference-dtypes", action="store_true")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    export_checkpoint(args.checkpoint_path, args.output, args.inference_dtypes, args.expected_sha256)


if __name__ == "__main__":
    main()