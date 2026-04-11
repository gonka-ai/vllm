"""Per-round layer hooks for structure breaking.

Applies transformations between transformer layers to break
the model learned output structure.

PATCHED: Replaced ContextVar with simple global bool for torch.compile compatibility.
torch.compile/dynamo cannot trace ContextVar.get() calls.
"""
from contextlib import contextmanager
from typing import List

import torch

from .gpu_random import generate_householder_vector, apply_householder

# Simple global flag instead of ContextVar (dynamo-compatible)
# Safe because PoC forward is synchronous and single-threaded
_poc_forward_active_flag: bool = False


@contextmanager
def poc_forward_context():
    """Context manager for PoC forward passes.

    Hooks only transform hidden states when this context is active.
    This allows inference and PoC to coexist without interference.
    """
    global _poc_forward_active_flag
    _poc_forward_active_flag = True
    try:
        yield
    finally:
        _poc_forward_active_flag = False


def is_poc_forward_active() -> bool:
    """Check if PoC forward context is active."""
    return _poc_forward_active_flag


class LayerHouseholderHook:
    """Per-round Householder reflections applied between transformer layers."""

    def __init__(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ):
        self.hooks: List = []
        self.reflection_vectors: List[torch.Tensor] = []
        self.block_hash = block_hash

    def _find_layers(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Find transformer layers in a model-agnostic way."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return list(model.transformer.h)
        elif hasattr(model, "layers"):
            return list(model.layers)
        return []

    def _setup(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ):
        """Setup hooks on all transformer layers."""
        layers = self._find_layers(model)
        self.num_total_layers = len(layers)

        for i in range(len(layers)):
            seed_str = f"{block_hash}_layer_{i}_householder"
            v = generate_householder_vector(seed_str, hidden_size, device)
            self.reflection_vectors.append(v)

            hook = layers[i].register_forward_hook(self._create_hook(i))
            self.hooks.append(hook)

    def _create_hook(self, layer_idx: int):
        """Create a forward hook that applies Householder reflection.

        Hook only transforms when poc_forward_context is active.
        Uses simple global bool check instead of ContextVar for dynamo compat.
        """
        def hook(module, input, output):
            if not is_poc_forward_active():
                return output

            v = self.reflection_vectors[layer_idx]

            def transform(x):
                return apply_householder(x, v.to(x.dtype))

            if isinstance(output, tuple):
                if len(output) >= 2:
                    hidden = output[0]
                    residual = output[1]
                    rest = output[2:] if len(output) > 2 else ()
                    transformed_hidden = transform(hidden)
                    transformed_residual = transform(residual)
                    return (transformed_hidden, transformed_residual) + rest
                else:
                    hidden = output[0]
                    transformed = transform(hidden)
                    return (transformed,)
            else:
                transformed = transform(output)
                return transformed

        return hook

    def detach(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.reflection_vectors = []

    @property
    def num_layers(self) -> int:
        """Number of layers with hooks attached."""
        return len(self.hooks)
