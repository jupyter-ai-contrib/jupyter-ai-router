# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
"""End-to-end integration tests for the three MessageRouter lifecycle observers.

Issue: https://github.com/jupyter-ai-contrib/jupyter-ai-router/issues/48

The router registers three lifecycle observers:

    1. ``observe_chat_init``  -- fires when a chat room opens
    2. ``observe_chat_msg``   -- fires when a client message is received
    3. ``observe_chat_stop``  -- fires when a chat room is closed/deleted

These tests exercise all three **end-to-end against a real, booted
jupyter_server**, using:

    * the real ``jupyter_ai_router`` server extension (its ``MessageRouter`` and
      its ``ChatManager`` subscription),
    * the real ``jupyterlab_chat.ChatManager`` and its Jupyter Events bus,
    * a real chat model -- ``WsChatModel`` (RTC-free) or ``YChat`` (RTC).

There are **no mocked chat models**. The same module runs unchanged in every
matrix environment (see ``noxfile.py``); it detects the active transport from
the live ``ChatManager`` and drives the corresponding real model.

In RTC mode the chat model is a real ``YChat`` driven through its real public
API (``add_message``), which writes to the Yjs document and fires the real
``observe_messages`` bridge -- no Yjs websocket client is needed. The only glue
not exercised here is the collaboration provider handing the ``YChat`` to the
``ChatManager`` (``get_document``); that is jupyter-chat's responsibility, so we
construct a real ``YChat`` and register it with the real ``ChatManager`` (the
same approach jupyter-ai-persona-manager's tests use) and emit the real
lifecycle events through the real event bus. The router only ever sees a real
``BaseChatModel``.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any, Callable

import pytest

from jupyterlab_chat.events import (
    CHAT_ROOM_EVENT_SCHEMA_ID,
    ChatEvent,
    ChatEventAction,
)
from jupyterlab_chat.models import (
    ChatMessageAction,
    Message,
    NewMessage,
)

#: Injected by the matrix (noxfile). Empty/unset -> "no expected provider".
EXPECTED_RTC_PROVIDER = os.environ.get("EXPECTED_RTC_PROVIDER") or None
#: Set to "1" by the nox matrix; gates the environment-specific provider
#: assertion (a plain ``pytest`` run has no injected expectation).
RTC_MATRIX = os.environ.get("RTC_MATRIX") == "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _router(jp_serverapp):
    return jp_serverapp.web_app.settings["jupyter-ai"]["router"]


def _chat_manager(jp_serverapp):
    return jp_serverapp.web_app.settings["chat_manager"]


def _event_logger(jp_serverapp):
    return jp_serverapp.web_app.settings.get("event_logger")


def _has_listener(event_logger, schema_id: str) -> bool:
    """Whether *any* listener is registered for ``schema_id``.

    Tolerant to jupyter_events internal attribute naming across versions.
    """
    if event_logger is None:
        return False
    for attr in (
        "_modified_listeners",
        "_unmodified_listeners",
        "_listeners",
    ):
        registry = getattr(event_logger, attr, None)
        if registry and registry.get(schema_id):
            return True
    return False


async def _pump_until(
    loop, condition: Callable[[], bool], timeout: float = 5.0, interval: float = 0.02
) -> bool:
    """Turn the event loop until ``condition()`` is truthy or ``timeout`` s pass."""
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if condition():
            return True
        await asyncio.sleep(interval)
    return condition()


class _Recorder:
    """A plain recording observer -- not a mock, just a list-appending callable."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, *args: Any) -> None:
        self.calls.append(args)


def _make_ychat():
    """Build a real, fully-loaded ``YChat`` (mirrors jupyter-chat's own tests)."""
    from jupyterlab_chat.ychat import YChat

    chat = YChat()
    # Give the document an id first so flipping ``dirty`` does not schedule
    # ``create_id()`` (which needs a running loop), then mark it loaded so the
    # observe_messages bridge stops skipping inserts.
    chat.set_id(f"router-int-{uuid.uuid4().hex}")
    chat.dirty = False
    return chat


# ---------------------------------------------------------------------------
# Environment sanity (provider resolution)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not RTC_MATRIX,
    reason="provider expectation is only injected by the nox RTC matrix",
)
def test_server_resolved_expected_provider(jp_serverapp):
    """The live server must have resolved this environment's RTC provider."""
    from jupyterlab_chat.rtc_lib import get_server_session_rtc_info

    info = get_server_session_rtc_info(jp_serverapp)
    assert info.provider == EXPECTED_RTC_PROVIDER
    # The ChatManager's transport must agree with the resolved provider.
    assert _chat_manager(jp_serverapp)._rtc_enabled == (EXPECTED_RTC_PROVIDER is not None)


# ---------------------------------------------------------------------------
# The three observers, end-to-end
# ---------------------------------------------------------------------------
def test_router_subscribed_to_chat_manager(jp_serverapp, jp_asyncio_loop):
    """The router extension must subscribe to the ChatManager event bus at boot."""
    loop = jp_asyncio_loop
    event_logger = _event_logger(jp_serverapp)

    subscribed = loop.run_until_complete(
        _pump_until(
            loop, lambda: _has_listener(event_logger, CHAT_ROOM_EVENT_SCHEMA_ID)
        )
    )
    assert subscribed, (
        "RouterExtension never subscribed to the ChatManager lifecycle bus "
        f"({CHAT_ROOM_EVENT_SCHEMA_ID})"
    )


def test_three_observers_end_to_end(jp_serverapp, jp_asyncio_loop, tmp_path):
    """init -> message -> stop, driven through the real ChatManager + real model."""
    loop = jp_asyncio_loop
    router = _router(jp_serverapp)
    manager = _chat_manager(jp_serverapp)
    event_logger = _event_logger(jp_serverapp)
    rtc = manager._rtc_enabled

    init_rec = _Recorder()
    msg_rec = _Recorder()
    stop_rec = _Recorder()
    router.observe_chat_init(init_rec)
    router.observe_chat_stop(stop_rec)

    body = f"hello from a real client {uuid.uuid4().hex}"
    path = "router-integration.chat"

    async def scenario():
        # 0) The router subscribes to the bus from a background task; the OPENED
        #    event does not replay, so wait until the subscription is live before
        #    opening the chat.
        assert await _pump_until(
            loop, lambda: _has_listener(event_logger, CHAT_ROOM_EVENT_SCHEMA_ID)
        ), "router did not subscribe to the ChatManager bus"

        # 1) INIT: open a real chat. Both branches emit the *real* OPENED event
        #    through the *real* ChatManager event bus.
        if rtc:
            chat = _make_ychat()
            room_id = f"json:chat:{chat.get_id()}"
            manager._models[path] = chat
            manager._room_to_path[room_id] = path
            manager._emit_event(
                ChatEvent(path=path, action=ChatEventAction.OPENED, room_id=room_id)
            )
            key = room_id
        else:
            # ws_open creates a real WsChatModel and emits the real OPENED event.
            chat = manager.ws_open(path)
            key = path

        assert await _pump_until(loop, lambda: any(init_rec.calls)), (
            "observe_chat_init never fired after OPENED"
        )
        assert init_rec.calls[0][0] == key
        assert init_rec.calls[0][1] is chat
        assert router.active_chats.get(key) is chat

        # 2) MESSAGE: register a message observer, then deliver a real client
        #    message and assert the router routes it. The router only routes
        #    CLIENT_MSG_RECEIVED events.
        router.observe_chat_msg(key, msg_rec)
        now = time.time()
        if rtc:
            # Real public API: add_message writes to the Yjs doc, and a non-bot
            # sender is classified CLIENT_MSG_RECEIVED by YChat's real
            # observe_messages bridge.
            chat.add_message(NewMessage(body=body, sender="human-int"))
        else:
            # WsChatModel emits CLIENT_MSG_RECEIVED via its handler; this is the
            # exact call the real WSChatHandler makes on an incoming client frame
            # (see websocket_handler.WSChatHandler._handle_new_message).
            message = Message(id="int-msg", body=body, sender="human-int", time=now)
            chat._emit_message_event(ChatMessageAction.CLIENT_MSG_RECEIVED, message)

        assert await _pump_until(loop, lambda: any(msg_rec.calls)), (
            "observe_chat_msg never fired for a CLIENT_MSG_RECEIVED message"
        )
        routed_room, routed_msg = msg_rec.calls[0]
        assert routed_room == key
        assert routed_msg.body == body

        # 3) STOP: close the chat. Both branches emit the real CLOSED event; the
        #    router must disconnect the chat and fire the stop observer.
        if rtc:
            manager._free(path, ChatEventAction.CLOSED)
        else:
            # Last client gone with no active writers -> the manager frees the
            # model and emits CLOSED.
            manager.ws_client_gone(path)

        assert await _pump_until(loop, lambda: any(stop_rec.calls)), (
            "observe_chat_stop never fired after CLOSED"
        )
        assert stop_rec.calls[0][0] == key
        assert key not in router.active_chats

    loop.run_until_complete(scenario())


def test_slash_command_observer_end_to_end(jp_serverapp, jp_asyncio_loop, tmp_path):
    """A real client slash-command message reaches a registered slash observer."""
    loop = jp_asyncio_loop
    router = _router(jp_serverapp)
    manager = _chat_manager(jp_serverapp)
    event_logger = _event_logger(jp_serverapp)
    rtc = manager._rtc_enabled

    slash_rec = _Recorder()
    path = "router-integration-slash.chat"

    async def scenario():
        assert await _pump_until(
            loop, lambda: _has_listener(event_logger, CHAT_ROOM_EVENT_SCHEMA_ID)
        )

        if rtc:
            chat = _make_ychat()
            room_id = f"json:chat:{chat.get_id()}"
            manager._models[path] = chat
            manager._room_to_path[room_id] = path
            manager._emit_event(
                ChatEvent(path=path, action=ChatEventAction.OPENED, room_id=room_id)
            )
            key = room_id
        else:
            chat = manager.ws_open(path)
            key = path

        assert await _pump_until(loop, lambda: key in router.active_chats)

        router.observe_slash_cmd_msg(key, "help", slash_rec)
        now = time.time()
        if rtc:
            chat.add_message(NewMessage(body="/help topic", sender="human-int"))
        else:
            message = Message(
                id="int-slash", body="/help topic", sender="human-int", time=now
            )
            chat._emit_message_event(ChatMessageAction.CLIENT_MSG_RECEIVED, message)

        assert await _pump_until(loop, lambda: any(slash_rec.calls)), (
            "slash command observer never fired"
        )
        room, command, trimmed = slash_rec.calls[0]
        assert room == key
        assert command == "help"
        assert trimmed.body == "topic"

    loop.run_until_complete(scenario())
