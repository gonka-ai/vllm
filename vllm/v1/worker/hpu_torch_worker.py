# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The stock model runner on an Intel Gaudi card.

Select it with ``--worker-cls vllm.v1.worker.hpu_torch_worker.HPUTorchWorker
--enforce-eager --no-async-scheduling``. The card is used as a plain torch
device: no CUDA graphs, no Triton kernels, attention in torch
(``torch_paged_attn``). ``VLLM_HPU_TORCH_COMPILE=1`` compiles the forward with
the Habana backend, one static graph per batch size; the attention calls stay
outside the graph.
"""

import os
from contextlib import AbstractContextManager, nullcontext

import torch

from vllm.config import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import attention
from vllm.model_executor.layers.attention.attention import Attention
from vllm.platforms import current_platform
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.block_table import BlockTable, SlotMappingMode
from vllm.v1.worker.cpu_model_runner import _torch_cuda_wrapper
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import (
    Worker,
    _num_workspace_lanes,
    init_worker_distributed_environment,
)
from vllm.v1.worker.worker_base import CompilationTimes
from vllm.v1.worker.workspace import init_workspace_manager

logger = init_logger(__name__)

_ATTN_BACKEND = "vllm.v1.attention.backends.torch_paged_attn.TorchPagedAttentionBackend"


def _compute_slot_mapping(
    self: BlockTable,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """BlockTable.compute_slot_mapping without the Triton kernel."""
    if self.slot_mapping_mode == SlotMappingMode.NONE:
        return
    assert self.dcp_world_size == 1, "context parallelism is not supported"
    num_tokens = positions.shape[0]
    # The request of every token, with no readback from the device.
    ends = query_start_loc[1 : num_reqs + 1]
    tokens = torch.arange(num_tokens, device=positions.device, dtype=ends.dtype)
    req_ids = (tokens.unsqueeze(1) >= ends.unsqueeze(0)).sum(dim=1)
    block_ids = (positions // self.kv_cache_block_size) * self.blocks_per_kv_block + (
        positions % self.kv_cache_block_size
    ) // self.block_size
    table = self.block_table.gpu
    block_numbers = table.view(-1).index_select(0, req_ids * table.shape[1] + block_ids)
    slot_mapping = self.slot_mapping.gpu
    slot_mapping[:num_tokens] = (
        block_numbers * self.block_size + positions % self.block_size
    )
    slot_mapping[num_tokens:] = PAD_SLOT_ID


def _memory_usage(cls, device: torch.device | None = None) -> float:
    free, total = torch.hpu.mem_get_info()
    return float(total - free)


def _attn_backend(cls, *args, **kwargs) -> str:
    return _ATTN_BACKEND


def _use_torch_kernels() -> None:
    """Swap the pieces that need a kernel the card does not have."""
    platform = type(current_platform)
    platform.get_attn_backend_cls = classmethod(  # type: ignore[method-assign,assignment]
        _attn_backend
    )
    platform.get_current_memory_usage = classmethod(  # type: ignore[method-assign,assignment]
        _memory_usage
    )
    # vllm-gaudi replaces Attention.forward with one for its own backend.
    stock_forward = getattr(Attention, "_vllm_gaudi_original_forward", None)
    if stock_forward is not None:
        Attention.forward = stock_forward  # type: ignore[method-assign]
    BlockTable.compute_slot_mapping = (  # type: ignore[method-assign]
        _compute_slot_mapping
    )
    # A compiled forward must not trace into the attention calls: they are
    # keyed by layer name, so each layer would get its own compiled copy.
    for name in ("unified_kv_cache_update", "unified_attention_with_output"):
        setattr(attention, name, torch.compiler.disable(getattr(attention, name)))
    torch.accelerator.synchronize = lambda *args, **kwargs: torch.hpu.synchronize()
    torch.accelerator.empty_cache = lambda *args, **kwargs: None


class HPUTorchModelRunner(GPUModelRunner):
    def __init__(self, vllm_config, device: torch.device):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

    def load_model(self, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        if os.environ.get("VLLM_HPU_TORCH_COMPILE") == "1":
            # One static graph per batch size.
            torch._dynamo.config.recompile_limit = 256
            self.model.forward = torch.compile(
                self.model.forward, backend="hpu_backend", dynamic=False
            )

    def _init_device_properties(self) -> None:
        pass

    def _sync_device(self) -> None:
        torch.hpu.synchronize()

    def _zero_block_ids(self, block_ids: list[int]) -> None:
        seen: set[int] = set()
        for group in self.kv_cache_config.kv_cache_groups:
            if not isinstance(group.kv_cache_spec, FullAttentionSpec):
                continue
            for layer_name in group.layer_names:
                ctx = self.compilation_config.static_forward_context.get(layer_name)
                kv = getattr(ctx, "kv_cache", None)
                if not isinstance(kv, torch.Tensor) or id(kv) in seen:
                    continue
                seen.add(id(kv))
                kv[block_ids] = 0

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        return sampled_token_ids.tolist()


class HPUTorchWorker(Worker):
    def __init__(self, *args, **kwargs):
        _use_torch_kernels()
        super().__init__(*args, **kwargs)
        self.parallel_config.disable_custom_all_reduce = True

    def init_device(self):
        self.device = torch.device("hpu")
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            "hccl",
        )
        set_random_seed(self.model_config.seed)
        init_workspace_manager(
            self.device, 1, _num_workspace_lanes(self.vllm_config, False)
        )
        self.model_runner = HPUTorchModelRunner(self.vllm_config, self.device)

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        return nullcontext()

    def sleep(self, level: int = 1) -> None:
        logger.warning("sleep mode is not supported on HPU, ignore it.")

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("sleep mode is not supported on HPU, ignore it.")

    def determine_available_memory(self) -> int:
        if self.cache_config.kv_cache_memory_bytes is not None:
            return self.cache_config.kv_cache_memory_bytes
        torch.hpu.synchronize()
        free, total = torch.hpu.mem_get_info()
        kv_cache_memory = int(
            total * self.cache_config.gpu_memory_utilization - (total - free)
        )
        if kv_cache_memory <= 0:
            raise ValueError(
                f"No device memory left for the KV cache: {format_gib(free)} of "
                f"{format_gib(total)} GiB free at gpu_memory_utilization="
                f"{self.cache_config.gpu_memory_utilization}."
            )
        logger.info("HPU KV cache: %s GiB", format_gib(kv_cache_memory))
        return kv_cache_memory

    def compile_or_warm_up_model(self) -> CompilationTimes:
        set_random_seed(self.model_config.seed)
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )
