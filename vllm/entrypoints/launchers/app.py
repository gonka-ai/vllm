# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings
from argparse import Namespace

from fastapi import FastAPI

from vllm.config import ModelConfig
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.entrypoints.serve.middleware.register import init_entrypoints_middleware
from vllm.entrypoints.serve.sagemaker.api_router import sagemaker_standards_bootstrap
from vllm.plugins.endpoint_plugins.interface import attach_endpoint_plugins
from vllm.tasks import FALLBACK_SUPPORTED_TASKS, SupportedTask

from .api_server.routers import register_api_routers
from .utils.server_utils import lifespan


def build_app(
    args: Namespace,
    supported_tasks: tuple["SupportedTask", ...] | None = None,
    model_config: ModelConfig | None = None,
) -> FastAPI:
    if supported_tasks is None:
        warnings.warn(
            "The 'supported_tasks' parameter was not provided to "
            "build_app and will be required in a future version. "
            "Defaulting to ('generate',).",
            DeprecationWarning,
            stacklevel=2,
        )
        supported_tasks = FALLBACK_SUPPORTED_TASKS

    if args.disable_fastapi_docs:
        app = FastAPI(
            openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan
        )
    elif args.enable_offline_docs:
        app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
    else:
        app = FastAPI(lifespan=lifespan)
    app.state.args = args
    app.root_path = args.root_path

    register_api_routers(args, app, supported_tasks, model_config)

    # Endpoint plugins are attached last so their routes are registered after all core
    # routers. This runs even for the CPU only render server. A plugin eligible for
    # the `render` task still gets its routes registered. It receives
    # `engine_client=None` at Phase B (see `_init_endpoint_plugins_state`).
    attach_endpoint_plugins(app, supported_tasks)

    init_exception_handler(app)
    init_entrypoints_middleware(args, app, supported_tasks)
    app = sagemaker_standards_bootstrap(app)

    # PoC routes: registered directly, not via include_router — FastAPI's
    # _IncludedRouter crashes prometheus route-name lookup (0.20 fix kept).
    # The gonka_poc plugin is required — there is no in-tree implementation.
    # On this base the app is built here, not in the deprecated
    # entrypoints/openai/api_server.py, so the registration moved with it.
    # A cleaner home is upstream's endpoint-plugin interface
    # (attach_endpoint_plugins above); doing that needs changes in the plugin
    # and is the right shape if this line ever becomes a release line.
    from gonka_poc.poc.routes import router as poc_router

    for _poc_route in poc_router.routes:
        app.add_api_route(
            _poc_route.path,
            _poc_route.endpoint,
            methods=list(_poc_route.methods),
            name=_poc_route.name,
        )
    app.state.poc_enabled = True
    # Decode-PoC is the canonical scheme; per-request max_tokens still
    # selects prefill-only (max_tokens == 0).
    app.state.poc_decode = True
    # Mining rounds (init/generate) need the PoC gate that the plugin's own
    # entrypoint used to install; on this base the app is built here.
    from gonka_poc.entrypoint.gating import (
        DEFAULT_BLOCKED_PREFIXES, PoCGate, install_gating_middleware)
    _gate = PoCGate()
    app.state.gonka_gate = _gate
    install_gating_middleware(
        app, gate=_gate, blocked_prefixes=DEFAULT_BLOCKED_PREFIXES)
    return app
