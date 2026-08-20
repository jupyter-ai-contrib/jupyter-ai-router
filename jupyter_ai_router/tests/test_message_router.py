# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
"""Integration tests for the jupyter_ai_router MessageRouter.

Issue: https://github.com/jupyter-ai-contrib/jupyter-ai-router/issues/48

These tests boot a **real** ``jupyter_server`` (via ``jp_serverapp``) with the
real ``jupyter_ai_router`` and ``jupyterlab_chat`` server extensions, plus
whichever RTC provider is installed in the current matrix environment
(none / ``jupyter_collaboration`` / ``jupyter_server_documents``). The chat
**model is obtained the real way from the live ``ChatManager``** -- which returns
a ``WsChatModel`` (RTC-free) or a ``YChat`` (RTC) depending on the environment.
No chat models are mocked and no models are hand-constructed and injected.

The three router lifecycle observers are exercised end-to-end through the real
event bus and the real model:

    * ``observe_chat_init`` -- a chat is opened the way the server opens it
      (RTC-free: ``ChatManager.ws_open``, exactly what the WS handler calls;
      RTC: the collaboration provider's ``get_document`` materializes the room,
      which emits the real ``initialize`` room event the ``ChatManager`` forwards).
    * ``observe_chat_msg`` -- a real client message flows through the real model
      (RTC: ``YChat.add_message`` writes to the shared doc; RTC-free: the exact
      ``_emit_message_event(CLIENT_MSG_RECEIVED, …)`` the WS handler emits on an
      incoming client frame) and through the real ``observe_messages`` bridge.
    * ``observe_chat_stop`` -- the chat is closed the way the server closes it
      (RTC-free: last client gone; RTC: the provider's ``clean`` room event),
      and the router disconnects and fires the stop observer.

Plus a small set of transport-independent router routing-logic checks that use
real ``Message`` objects and real callbacks (no mocks, no chat model needed).
"""
from __future__ import annotations

import asyncio
import inspect
import os
import uuid
from time import time
from typing import Any, Callable

import pytest

from jupyterlab_chat.events import (
    CHAT_ROOM_EVENT_SCHEMA_ID,
    JUPYTER_COLLABORATION_EVENTS_URI,
)
from jupyterlab_chat.models import (
    ChatMessageAction,
    ChatMessageEvent,
    Message,
    NewMessage,
)

from jupyter_ai_router.router import MessageRouter, matches_pattern
from jupyter_ai_router.utils import get_first_word, is_persona

#: Injected by the nox matrix; empty/unset -> no expected provider.
EXPECTED_RTC_PROVIDER = os.environ.get("EXPECTED_RTC_PROVIDER") or None
#: Set to "1" by the nox matrix; gates the env-specific provider assertion.
RTC_MATRIX = os.environ.get("RTC_MATRIX") == "1"


# ===========================================================================
# Transport-independent routing logic (real Message objects, real callbacks)
# ===========================================================================
class TestUtils:
    def test_get_first_word_normal(self):
        assert get_first_word("hello world") == "hello"
        assert get_first_word("  hello world  ") == "hello"
        assert get_first_word("/refresh-personas") == "/refresh-personas"

    def test_get_first_word_edge_cases(self):
        assert get_first_word("") is None
        assert get_first_word("   ") is None
        assert get_first_word("single") == "single"

    def test_is_persona(self):
        assert is_persona("jupyter-ai-personas::jupyter_ai::JupyternautPersona") is True
        assert is_persona("human_user") is False
        assert is_persona("jupyter-ai-personas::custom::MyPersona") is True


class TestPatternMatching:
    def test_exact(self):
        assert matches_pattern("help", "help") is True
        assert matches_pattern("help", "status") is False

    def test_regex(self):
        assert matches_pattern("ai-generate", "ai-.*") is True
        assert matches_pattern("help", "ai-.*") is False

    def test_regex_groups(self):
        pattern = r"export-(json|csv|xml)"
        assert matches_pattern("export-json", pattern) is True
        assert matches_pattern("export-pdf", pattern) is False

    def test_invalid_regex(self):
        assert matches_pattern("help", "[invalid") is False


class _Recorder:
    """A plain recording callback -- not a mock, just a list-appending callable."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, *args: Any) -> None:
        self.calls.append(args)


class TestRouteMessageLogic:
    """Router parsing/trimming logic. Uses a real ``MessageRouter`` and real
    ``Message`` objects; no chat model is involved in these routing decisions."""

    def setup_method(self):
        self.router = MessageRouter()

    def test_slash_command_trimmed_and_cleaned(self):
        room = "room"
        rec = _Recorder()
        self.router.observe_slash_cmd_msg(room, "help", rec)
        self.router._route_message(
            room, Message(id="1", body="/help topic here", sender="u", time=1)
        )
        assert rec.calls[0][0] == room
        assert rec.calls[0][1] == "help"
        assert rec.calls[0][2].body == "topic here"

    def test_regular_message_routed_to_msg_observer(self):
        room = "room"
        rec = _Recorder()
        self.router.observe_chat_msg(room, rec)
        msg = Message(id="1", body="hello world", sender="u", time=1)
        self.router._route_message(room, msg)
        assert rec.calls[0] == (room, msg)

    def test_unmatched_slash_falls_through_to_msg_observer(self):
        room = "room"
        slash = _Recorder()
        regular = _Recorder()
        self.router.observe_slash_cmd_msg(room, "help", slash)
        self.router.observe_chat_msg(room, regular)
        self.router._route_message(
            room, Message(id="1", body="/unknown x", sender="u", time=1)
        )
        assert not slash.calls
        assert len(regular.calls) == 1

    def test_metadata_preserved_in_trimmed_message(self):
        room = "room"
        rec = _Recorder()
        self.router.observe_slash_cmd_msg(room, "help", rec)
        original = Message(
            id="id",
            body="/help getting-started",
            sender="u",
            time=1.5,
            mentions=["@x"],
            attachments=["f.txt"],
        )
        self.router._route_message(room, original)
        trimmed = rec.calls[0][2]
        assert (trimmed.id, trimmed.sender, trimmed.time) == ("id", "u", 1.5)
        assert trimmed.mentions == ["@x"] and trimmed.attachments == ["f.txt"]
        assert trimmed.body == "getting-started"
        assert original.body == "/help getting-started"  # unchanged

    def test_deleted_message_not_routed(self):
        room = "room"
        rec = _Recorder()
        self.router.observe_chat_msg(room, rec)
        self.router._route_message(
            room, Message(id="1", body="hi", sender="u", time=1, deleted=True)
        )
        assert not rec.calls

    def test_observer_error_is_isolated(self):
        room = "room"
        boom = _Recorder()

        def raiser(*a):
            raise RuntimeError("boom")

        self.router.observe_slash_cmd_msg(room, "help", raiser)
        self.router.observe_slash_cmd_msg(room, "help", boom)
        # A failing observer must not prevent others from being notified.
        self.router._route_message(
            room, Message(id="1", body="/help x", sender="u", time=1)
        )
        assert len(boom.calls) == 1


# ===========================================================================
# Real-server helpers
# ===========================================================================
def _settings(app):
    return app.web_app.settings


def _router(app) -> MessageRouter:
    return _settings(app)["jupyter-ai"]["router"]


def _chat_manager(app):
    return _settings(app)["chat_manager"]


def _event_logger(app):
    return _settings(app).get("event_logger")


def _has_listener(event_logger, schema_id: str) -> bool:
    """Whether any listener is registered for ``schema_id`` (tolerant to
    jupyter_events internal attribute naming across versions)."""
    if event_logger is None:
        return False
    for attr in ("_modified_listeners", "_unmodified_listeners", "_listeners"):
        registry = getattr(event_logger, attr, None)
        if registry and registry.get(schema_id):
            return True
    return False


async def _pump_until(
    condition: Callable[[], bool], timeout: float = 5.0, interval: float = 0.02
) -> bool:
    """Turn the running event loop until ``condition()`` is truthy or times out."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if condition():
            return True
        await asyncio.sleep(interval)
    return condition()


async def _wait_router_subscribed(app) -> None:
    event_logger = _event_logger(app)
    assert await _pump_until(
        lambda: _has_listener(event_logger, CHAT_ROOM_EVENT_SCHEMA_ID)
    ), "RouterExtension never subscribed to the ChatManager lifecycle bus"


async def _open_chat(app, path: str):
    """Open a chat the way the server does and return ``(model, key)`` from the
    live ``ChatManager``.

    ``model`` is a real ``WsChatModel`` (RTC-free) or ``YChat`` (RTC). ``key`` is
    the identifier the router uses in ``active_chats`` (room id under RTC, else
    the path).
    """
    manager = _chat_manager(app)
    router = _router(app)
    if manager._rtc_enabled:
        ydoc_api = _settings(app)["jupyter_server_ydoc"]
        kwargs = dict(path=path, content_type="chat", file_format="text", copy=False)
        # jupyter_collaboration needs create=True; the JSD shim auto-creates and
        # has no such parameter.
        if "create" in inspect.signature(ydoc_api.get_document).parameters:
            kwargs["create"] = True
        # Materialize the room -> the provider emits the real `initialize` room
        # event -> the ChatManager forwards it, caches the YChat, and emits OPENED.
        await ydoc_api.get_document(**kwargs)
        assert await _pump_until(
            lambda: manager.get(path) is not None
        ), "ChatManager never resolved a YChat for the opened room"
        key = next(rid for rid, p in manager._room_to_path.items() if p == path)
    else:
        # Exactly what WSChatHandler.open() calls for the first client; emits OPENED.
        manager.ws_open(path)
        key = path
    # The OPENED event is handled by the router's async listener; wait until it
    # has actually connected the chat (registering observe_messages) so message
    # delivery is not racy.
    assert await _pump_until(
        lambda: key in router.active_chats
    ), "router did not connect the chat after OPENED"
    model = manager.get(path)
    assert model is not None
    assert router.active_chats.get(key) is model
    if manager._rtc_enabled and model.dirty:
        # A collaborative YChat is created ``dirty`` and (under jupyter_collaboration)
        # only clears once the room performs its first save -- which is triggered
        # by a post-ready document change. Nudge it with a metadata write (as a
        # client does on open) so the doc settles to a loaded state; the
        # observe_messages bridge intentionally ignores inserts while dirty.
        model.set_metadata("_router_integration_ready", True)
        assert await _pump_until(
            lambda: not model.dirty, timeout=15.0
        ), "collaborative YChat never settled (stayed dirty after a save nudge)"
    return model, key


def _deliver_client_message(app, model, body: str, sender: str = "human-int") -> None:
    """Deliver a real client message through the real model pipeline."""
    if _chat_manager(app)._rtc_enabled:
        # In RTC mode the frontend writes directly to the shared doc; a non-bot
        # sender is classified CLIENT_MSG_RECEIVED by YChat's real bridge.
        model.add_message(NewMessage(body=body, sender=sender))
    else:
        # Exactly what WSChatHandler._handle_new_message emits on a client frame.
        model._emit_message_event(
            ChatMessageAction.CLIENT_MSG_RECEIVED,
            Message(id=uuid.uuid4().hex, body=body, sender=sender, time=time()),
        )


async def _close_chat(app, path: str, key: str) -> None:
    """Close the chat the way the server closes it (emits the real CLOSED event)."""
    manager = _chat_manager(app)
    if manager._rtc_enabled:
        # Drive the real RTC forwarder with the provider's `clean` room event.
        await manager._on_rtc_room_event(
            None,
            JUPYTER_COLLABORATION_EVENTS_URI,
            {"room": key, "path": path, "action": "clean"},
        )
    else:
        # Last client gone with no active writers -> the manager frees + emits CLOSED.
        manager.ws_client_gone(path)


def _chat_file(jp_root_dir, name: str) -> str:
    (jp_root_dir / name).write_text('{"messages": [], "users": {}, "metadata": {}}')
    return name


# ===========================================================================
# Environment sanity
# ===========================================================================
@pytest.mark.skipif(
    not RTC_MATRIX, reason="provider expectation is only injected by the nox matrix"
)
def test_server_resolved_expected_provider(jp_serverapp):
    from jupyterlab_chat.rtc_lib import get_server_session_rtc_info

    info = get_server_session_rtc_info(jp_serverapp)
    assert info.provider == EXPECTED_RTC_PROVIDER
    assert _chat_manager(jp_serverapp)._rtc_enabled == (EXPECTED_RTC_PROVIDER is not None)


# ===========================================================================
# The three observers, end-to-end, with a real ChatManager-provided model
# ===========================================================================
def test_router_subscribes_at_boot(jp_serverapp, jp_asyncio_loop):
    jp_asyncio_loop.run_until_complete(_wait_router_subscribed(jp_serverapp))


def test_init_message_stop_observers(jp_serverapp, jp_asyncio_loop, jp_root_dir):
    app = jp_serverapp
    router = _router(app)
    init, msg, stop = _Recorder(), _Recorder(), _Recorder()
    router.observe_chat_init(init)
    router.observe_chat_stop(stop)
    path = _chat_file(jp_root_dir, "router-lifecycle.chat")
    body = f"hello from a real client {uuid.uuid4().hex}"

    async def scenario():
        await _wait_router_subscribed(app)

        # INIT
        model, key = await _open_chat(app, path)
        assert await _pump_until(lambda: any(init.calls)), "init observer did not fire"
        assert init.calls[0][0] == key
        assert init.calls[0][1] is model
        assert router.active_chats.get(key) is model

        # MESSAGE
        router.observe_chat_msg(key, msg)
        _deliver_client_message(app, model, body)
        assert await _pump_until(lambda: any(msg.calls)), "msg observer did not fire"
        assert msg.calls[0][0] == key
        assert msg.calls[0][1].body == body

        # STOP
        await _close_chat(app, path, key)
        assert await _pump_until(lambda: any(stop.calls)), "stop observer did not fire"
        assert stop.calls[0][0] == key
        assert key not in router.active_chats

    jp_asyncio_loop.run_until_complete(scenario())


def test_slash_command_observer(jp_serverapp, jp_asyncio_loop, jp_root_dir):
    app = jp_serverapp
    router = _router(app)
    slash = _Recorder()
    path = _chat_file(jp_root_dir, "router-slash.chat")

    async def scenario():
        await _wait_router_subscribed(app)
        model, key = await _open_chat(app, path)
        router.observe_slash_cmd_msg(key, "help", slash)
        _deliver_client_message(app, model, "/help topic")
        assert await _pump_until(lambda: any(slash.calls)), "slash observer did not fire"
        room, command, trimmed = slash.calls[0]
        assert room == key
        assert command == "help"
        assert trimmed.body == "topic"

    jp_asyncio_loop.run_until_complete(scenario())


def test_disconnect_unobserves_real_model(jp_serverapp, jp_asyncio_loop, jp_root_dir):
    app = jp_serverapp
    router = _router(app)
    path = _chat_file(jp_root_dir, "router-disconnect.chat")

    async def scenario():
        await _wait_router_subscribed(app)
        model, key = await _open_chat(app, path)
        assert key in router.active_chats
        assert key in router.message_observers

        router.disconnect_chat(key)
        assert key not in router.active_chats
        assert key not in router.message_observers
        # A message delivered after disconnect must not be routed.
        rec = _Recorder()
        router.observe_chat_msg(key, rec)
        _deliver_client_message(app, model, "after disconnect")
        await asyncio.sleep(0.1)
        assert not rec.calls

    jp_asyncio_loop.run_until_complete(scenario())


def test_pre_existing_messages_not_routed(jp_serverapp, jp_asyncio_loop, jp_root_dir):
    """Messages predating the connection (loaded from disk) must be skipped."""
    app = jp_serverapp
    router = _router(app)
    rec = _Recorder()
    path = _chat_file(jp_root_dir, "router-preexisting.chat")

    async def scenario():
        await _wait_router_subscribed(app)
        _model, key = await _open_chat(app, path)
        router.observe_chat_msg(key, rec)

        connected_at = router._connected_at[key]
        old = Message(id="old", body="old", sender="u", time=connected_at - 60)
        new = Message(id="new", body="new", sender="u", time=connected_at + 1)
        # Drive the router's real message handler with real events on the real
        # connected room.
        router._on_message_event(
            key, ChatMessageEvent(ChatMessageAction.CLIENT_MSG_RECEIVED, old)
        )
        router._on_message_event(
            key, ChatMessageEvent(ChatMessageAction.CLIENT_MSG_RECEIVED, new)
        )
        assert [c[1].body for c in rec.calls] == ["new"]

    jp_asyncio_loop.run_until_complete(scenario())


def test_only_client_messages_routed(jp_serverapp, jp_asyncio_loop, jp_root_dir):
    """Server-sent / edited events must be ignored; only CLIENT_MSG_RECEIVED routes."""
    app = jp_serverapp
    router = _router(app)
    rec = _Recorder()
    path = _chat_file(jp_root_dir, "router-clientonly.chat")

    async def scenario():
        await _wait_router_subscribed(app)
        _model, key = await _open_chat(app, path)
        router.observe_chat_msg(key, rec)
        t = router._connected_at[key] + 1
        msg = Message(id="1", body="x", sender="u", time=t)
        for action in (
            ChatMessageAction.SERVER_MSG_SENT,
            ChatMessageAction.SERVER_MSG_UPDATED,
            ChatMessageAction.CLIENT_MSG_EDITED,
        ):
            router._on_message_event(key, ChatMessageEvent(action, msg))
        assert not rec.calls
        router._on_message_event(
            key, ChatMessageEvent(ChatMessageAction.CLIENT_MSG_RECEIVED, msg)
        )
        assert len(rec.calls) == 1

    jp_asyncio_loop.run_until_complete(scenario())
