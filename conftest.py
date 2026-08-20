"""Shared pytest configuration for jupyter_ai_router.

Every test runs against a **real** booted ``jupyter_server`` (via pytest-jupyter's
``jp_serverapp``) with the real ``jupyter_ai_router`` and ``jupyterlab_chat``
server extensions enabled -- plus whichever RTC provider is installed in the
current environment. There is no separate "unit" vs "integration" split: the
same suite runs unchanged in every RTC matrix environment (see ``noxfile.py``),
and the active transport (``WsChatModel`` vs ``YChat``) is whatever the real
server resolves.

``pytest_plugins`` must live in a top-level conftest, which is why it is here.
"""
import importlib.util

import pytest
from traitlets.config import Config

pytest_plugins = ("pytest_jupyter.jupyter_server",)

# RTC provider extension module names, in resolution order. Exactly one is
# installed per RTC matrix env (none for the RTC-free env). ``jupyter_server_fileid``
# is required by both providers (they resolve documents by file id).
_RTC_PROVIDERS = ("jupyter_server_ydoc", "jupyter_server_documents")


@pytest.fixture
def jp_server_config():
    """Enable the router, jupyterlab_chat, and any *installed* RTC provider.

    Environment-adaptive setup (not an assertion): each matrix env has exactly
    one provider installed, so the resolved transport reflects the env. The
    expected provider is asserted separately from ``EXPECTED_RTC_PROVIDER``.
    """
    extensions = {
        "jupyter_ai_router": True,
        "jupyterlab_chat": True,
    }
    if importlib.util.find_spec("jupyterlab") is not None:
        extensions["jupyterlab"] = True
    # The RTC providers resolve documents via the file-id manager.
    if importlib.util.find_spec("jupyter_server_fileid") is not None:
        extensions["jupyter_server_fileid"] = True
    for name in _RTC_PROVIDERS:
        if importlib.util.find_spec(name) is not None:
            extensions[name] = True
    return Config({"ServerApp": {"jpserver_extensions": extensions}})
