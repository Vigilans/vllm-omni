# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Published Qwen-Image-2.1 Comfy checkpoint names and quantization metadata."""

import json
import re

from safetensors import safe_open


def encoder_name(name: str) -> str:
    for prefix in ("model.layers.", "model.embed_tokens.", "model.norm."):
        if name.startswith(prefix):
            return "model.language_model." + name.removeprefix("model.")
    return name


def transformer_layer_names(name: str) -> tuple[str, ...]:
    for projection in ("to_q", "to_k", "to_v"):
        name = name.replace(f".attn.{projection}.", ".attn.to_qkv.")
    if ".img_mlp.gate_up." in name:
        return tuple(name.replace(".gate_up.", f".{part}.") for part in ("gate_layer", "proj"))
    return (name,)


def layer_metadata(path: str, *, transformer: bool = False) -> dict[str, dict]:
    result = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            if not name.endswith(".comfy_quant"):
                continue
            metadata = json.loads(bytes(checkpoint.get_tensor(name).tolist()))
            names = transformer_layer_names(name) if transformer else (encoder_name(name),)
            for mapped in names:
                prefix = mapped.removesuffix(".comfy_quant")
                if prefix in result and result[prefix] != metadata:
                    raise ValueError(f"Fused projection {prefix} has incompatible checkpoint formats")
                result[prefix] = metadata
    return result


def transformer_weights(name, tensor):
    if ".img_mlp.gate_up." in name:
        gate, up = tensor.chunk(2, dim=0)
        yield name.replace(".gate_up.", ".gate_layer."), gate
        yield name.replace(".gate_up.", ".proj."), up
    else:
        yield name, tensor


def vae_name(name: str) -> str:
    if name.startswith("conv1."):
        return name.replace("conv1.", "quant_conv.", 1)
    if name.startswith("conv2."):
        return name.replace("conv2.", "post_quant_conv.", 1)
    name = re.sub(r"^(encoder|decoder)\.conv1\.", r"\1.conv_in.", name)
    name = re.sub(r"\.(middle)\.([02])\.", lambda m: f".mid_block.resnets.{int(m[2]) // 2}.", name)
    name = name.replace(".middle.1.", ".mid_block.attentions.0.")
    name = re.sub(
        r"^encoder\.downsamples\.(\d+)\.downsamples\.(\d+)\.",
        lambda m: f"encoder.down_blocks.{m[1]}." + (f"resnets.{m[2]}." if int(m[2]) < 2 else "downsampler."),
        name,
    )
    name = re.sub(
        r"^decoder\.upsamples\.(\d+)\.upsamples\.(\d+)\.",
        lambda m: f"decoder.up_blocks.{m[1]}." + (f"resnets.{m[2]}." if int(m[2]) < 3 else "upsampler."),
        name,
    )
    for source, target in (
        (".residual.0.", ".norm1."), (".residual.2.", ".conv1."),
        (".residual.3.", ".norm2."), (".residual.6.", ".conv2."),
        (".shortcut.", ".conv_shortcut."), (".head.0.", ".norm_out."), (".head.2.", ".conv_out."),
    ):
        name = name.replace(source, target)
    return name


def vae_weight(name, tensor):
    # Comfy stores spatial convolutions with a singleton temporal kernel axis.
    if tensor.ndim == 5 and tensor.shape[2] == 1:
        tensor = tensor.squeeze(2)
    return vae_name(name), tensor
