# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
"""RTC integration matrix for jupyter_ai_router.

Issue: https://github.com/jupyter-ai-contrib/jupyter-ai-router/issues/48

Runs the SAME test suite (``jupyter_ai_router/tests``) under three different
dependency sets so the router's three lifecycle observers (init, message, stop)
are exercised end-to-end against a real ``jupyterlab_chat.ChatManager`` in every
supported transport:

    - ``no_rtc``      -- RTC-free, WebSocket ``WsChatModel`` (just ``jupyterlab_chat``)
    - ``rtc_jcollab`` -- RTC via ``jupyter_collaboration`` (``jupyter_server_ydoc``)
    - ``rtc_jsd``     -- RTC via ``jupyter_server_documents``

Each session installs its packages and injects the expected provider via
``EXPECTED_RTC_PROVIDER``; the integration tests adapt to whichever transport is
active and additionally assert the real server resolved the expected provider.

Usage::

    nox -l                                  # list sessions
    nox -s integration                      # run all three environments
    nox -s "integration(env='no_rtc')"      # run one environment
"""
import nox

# Prefer uv for fast env creation, fall back to virtualenv.
nox.options.default_venv_backend = "uv|virtualenv"

_TESTS = "jupyter_ai_router/tests"

# env name -> (expected provider module or None, extra packages providing it)
_RTC_ENVS = {
    "no_rtc": (None, []),
    "rtc_jcollab": ("jupyter_server_ydoc", ["jupyter_collaboration>=4,<6"]),
    "rtc_jsd": ("jupyter_server_documents", ["jupyter_server_documents"]),
}


@nox.session
@nox.parametrize("env", list(_RTC_ENVS))
def integration(session: nox.Session, env: str) -> None:
    """Run the full test suite against one RTC environment."""
    expected, extra = _RTC_ENVS[env]
    # These tests exercise only the Python server extension; skip the JS
    # labextension build (irrelevant here, and it needs node/network). For
    # editable installs hatch-jupyter-builder ignores skip-if-exists, so the
    # SKIP_JUPYTER_BUILDER env var is the reliable lever.
    session.env["SKIP_JUPYTER_BUILDER"] = "1"
    # jupyterlab is installed so any RTC provider that assumes it is present can
    # load cleanly under a real jupyter_server boot.
    session.install("-e", ".[test]", "jupyterlab>=4.0.0,<5", *extra)
    session.run(
        "pytest",
        _TESTS,
        "-vv",
        "-r",
        "ap",
        # RTC_MATRIX gates the environment-specific "expected provider"
        # assertion on; EXPECTED_RTC_PROVIDER is the per-environment provider
        # the test asserts the live server resolved.
        env={"RTC_MATRIX": "1", "EXPECTED_RTC_PROVIDER": expected or ""},
    )
