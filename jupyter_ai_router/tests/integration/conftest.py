# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
"""Fixtures for the jupyter_ai_router RTC integration matrix.

These tests boot a *real* jupyter_server (via pytest-jupyter's ``jp_serverapp``)
with the ``jupyter_ai_router`` and ``jupyterlab_chat`` server extensions enabled,
plus whichever RTC provider is installed in the current environment. The same
test module runs unchanged in every matrix environment; only the set of
installed packages (and the injected ``EXPECTED_RTC_PROVIDER``) differs. See
``noxfile.py``.

``pytest_plugins = ("pytest_jupyter.jupyter_server",)`` is declared in the
top-level ``conftest.py`` of the package, so ``jp_serverapp`` / ``jp_fetch`` /
``jp_asyncio_loop`` are available here without re-declaring it (``pytest_plugins``
is only allowed in a top-level conftest).
"""
from __future__ import annotations

import importlib.util

import pytest
from traitlets.config import Config

# Provider extension module names, in resolution order. Exactly one of these is
# installed per RTC matrix environment (none for the RTC-free environment).
_RTC_PROVIDERS = ("jupyter_server_ydoc", "jupyter_server_documents")


@pytest.fixture
def jp_server_config():
    """Enable ``jupyter_ai_router`` + ``jupyterlab_chat`` and any *installed*
    RTC provider.

    This is environment-adaptive setup (not an assertion): each matrix env has
    exactly one provider installed, so the resolved transport reflects the env.
    The expected provider is asserted separately from ``EXPECTED_RTC_PROVIDER``.
    """
    extensions = {
        "jupyter_ai_router": True,
        "jupyterlab_chat": True,
    }
    # Enable jupyterlab too (if installed) so providers that assume it is present
    # load cleanly.
    if importlib.util.find_spec("jupyterlab") is not None:
        extensions["jupyterlab"] = True
    for name in _RTC_PROVIDERS:
        if importlib.util.find_spec(name) is not None:
            extensions[name] = True
    return Config({"ServerApp": {"jpserver_extensions": extensions}})
