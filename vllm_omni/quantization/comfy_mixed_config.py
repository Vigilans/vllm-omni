# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Published Comfy mixed weights for replicated text encoders."""

import torch
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.utils import set_weight_attrs

from vllm_omni.quantization.int8_convrot_config import DiffusionInt8ConvRotConfig


class ComfyMixedConfig(QuantizationConfig):
    is_checkpoint_quantized = True

    def __init__(self, layer_configs: dict[str, dict]):
        super().__init__()
        self.layer_configs = layer_configs
        self.fp8 = Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic")
        self.int8 = DiffusionInt8ConvRotConfig(
            layer_configs={name: value for name, value in layer_configs.items() if value["format"] == "int8_tensorwise"}
        )

    @classmethod
    def get_name(cls):
        return "comfy_mixed"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 100

    @classmethod
    def get_config_filenames(cls):
        return []

    @classmethod
    def from_config(cls, config):
        return cls(config["layer_configs"])

    def get_quant_method(self, layer, prefix):
        if not isinstance(layer, LinearBase):
            return None
        metadata = self.layer_configs.get(prefix)
        if metadata is None:
            return UnquantizedLinearMethod()
        if metadata["format"] == "float8_e4m3fn":
            return self.fp8.get_quant_method(layer, prefix)
        if metadata["format"] == "nvfp4" and isinstance(layer, ReplicatedLinear):
            return ComfyNVFP4LinearMethod()
        if metadata["format"] == "int8_tensorwise":
            return self.int8.get_quant_method(layer, prefix)
        if metadata["format"] == "asym_w4a8_int8" and isinstance(layer, ReplicatedLinear):
            return ComfyW4A8LinearMethod(metadata)
        raise ValueError(f"Unsupported Comfy encoder projection {prefix}: {metadata}")


class ComfyW4A8LinearMethod(LinearMethodBase):
    """Keep published packed codes and scales for comfy-kitchen's W4A8 kernel."""

    def __init__(self, metadata: dict):
        if not metadata.get("convrot", False):
            raise ValueError("Comfy W4A8 requires ConvRot weights")
        self.group_size = metadata["group_size"]
        self.convrot_groupsize = metadata.get("convrot_groupsize", 256)

    def create_weights(
        self, layer, input_size_per_partition, output_partition_sizes,
        input_size, output_size, params_dtype, **extra_weight_attrs,
    ):
        shapes = {
            "weight": ((output_size, input_size // 2), torch.int8),
            "weight_s_rel": ((output_size, input_size // self.group_size), torch.float8_e4m3fn),
            "weight_s_channel": ((output_size,), torch.float32),
            "weight_codebook": ((16,), torch.float32),
        }
        for name, (shape, dtype) in shapes.items():
            parameter = torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            set_weight_attrs(parameter, extra_weight_attrs)
            layer.register_parameter(name, parameter)

    def apply(self, layer, x, bias=None):
        from comfy_kitchen.tensor.w4a8_int8 import w4a8_int8_linear

        shape = x.shape
        output = w4a8_int8_linear(
            x.reshape(-1, shape[-1]).contiguous(), layer.weight,
            layer.weight_s_rel, layer.weight_s_channel,
            codebook=layer.weight_codebook, bias=bias,
            group_size=self.group_size, convrot_groupsize=self.convrot_groupsize,
            out_dtype=x.dtype,
        )
        return output.reshape(*shape[:-1], layer.weight.shape[0])


class ComfyInt8Embedding(torch.nn.Module):
    """Gather and unrotate requested rows without expanding the vocabulary table."""

    def __init__(self, num_embeddings: int, embedding_dim: int, metadata: dict, dtype: torch.dtype):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.orig_dtype = dtype
        self.metadata = metadata
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=torch.int8), requires_grad=False
        )
        self.weight_scale = torch.nn.Parameter(
            torch.empty(num_embeddings, 1, dtype=torch.float32), requires_grad=False
        )

    def forward(self, input_ids):
        from comfy_kitchen.tensor import TensorWiseINT8Layout

        params = TensorWiseINT8Layout.Params(
            scale=self.weight_scale, orig_dtype=self.orig_dtype,
            orig_shape=tuple(self.weight.shape), is_weight=True,
            convrot=self.metadata.get("convrot", False),
            convrot_groupsize=self.metadata.get("convrot_groupsize", 256),
        )
        return TensorWiseINT8Layout.dequantize_embedding(self.weight, params, input_ids)


class ComfyNVFP4LinearMethod(LinearMethodBase):
    def create_weights(
        self, layer, input_size_per_partition, output_partition_sizes,
        input_size, output_size, params_dtype, **extra_weight_attrs,
    ):
        shapes = {
            "weight": ((output_size, input_size // 2), torch.uint8),
            "weight_scale": ((output_size, input_size // 16), torch.float8_e4m3fn),
            "weight_scale_2": ((1,), torch.float32),
        }
        for name, (shape, dtype) in shapes.items():
            parameter = torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            set_weight_attrs(parameter, extra_weight_attrs)
            layer.register_parameter(name, parameter)
        layer.orig_dtype = params_dtype

    def apply(self, layer, x, bias=None):
        import comfy_kitchen as ck

        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1]).contiguous()
        scale = (x.float().abs().amax() / (448 * 6)).clamp_min(1e-12)
        packed, block_scale = ck.quantize_nvfp4(x, scale, pad_16x=True)
        output = ck.scaled_mm_nvfp4(
            packed,
            layer.weight,
            scale,
            layer.weight_scale_2,
            block_scale,
            layer.weight_scale,
            bias=bias,
            out_dtype=x.dtype,
        )
        return output[:x.shape[0], :layer.weight.shape[0]].reshape(*original_shape[:-1], layer.weight.shape[0])
