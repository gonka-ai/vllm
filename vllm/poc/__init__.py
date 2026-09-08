# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire types shared by the engine seams and the gonka_poc plugin.

The PoC implementation itself lives in the plugin; this package holds only
the types that must be one class per process and therefore cannot be
resolved per-caller.
"""

from .poc_params import PoCParams

__all__ = ["PoCParams"]
