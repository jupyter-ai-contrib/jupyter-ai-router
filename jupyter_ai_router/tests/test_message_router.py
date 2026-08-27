"""Tests for MessageRouter functionality.

Every test runs against a real, booted jupyter_server (see the top-level
conftest). This file preserves the full router test suite that shipped in
jupyter_ai_router 0.1.0a0 -- with the previously mocked chat model replaced by a
real ``WsChatModel`` -- and adds RTC matrix integration tests (issue #48) that
exercise the three lifecycle observers end-to-end against the model the live
``ChatManager`` resolves for the active transport (``WsChatModel`` or ``YChat``).
"""
from __future__ import annotations

import asyncio
import inspect
import os
import tempfile
import uuid
from pathlib import Path
from time import time
from typing import Any, Callable
from unittest.mock import Mock
from urllib.parse import quote

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
from jupyterlab_chat.websocket_model import WsChatModel

from jupyter_ai_router.router import MessageRouter, matches_pattern
from jupyter_ai_router.utils import get_first_word, is_persona

#: Injected by the nox matrix; empty/unset -> no expected provider.
EXPECTED_RTC_PROVIDER = os.environ.get("EXPECTED_RTC_PROVIDER") or None
#: Set to "1" by the nox matrix; gates the env-specific provider assertion.
RTC_MATRIX = os.environ.get("RTC_MATRIX") == "1"


class TestUtils:
    """Test utility functions."""

    def test_get_first_word_normal(self):
        """Test getting first word from normal string."""
        assert get_first_word("hello world") == "hello"
        assert get_first_word("  hello world  ") == "hello"
        assert get_first_word("/refresh-personas") == "/refresh-personas"

    def test_get_first_word_edge_cases(self):
        """Test edge cases for get_first_word."""
        assert get_first_word("") is None
        assert get_first_word("   ") is None
        assert get_first_word("single") == "single"

    def test_is_persona(self):
        """Test persona username detection."""
        assert is_persona("jupyter-ai-personas::jupyter_ai::JupyternautPersona") is True
        assert is_persona("human_user") is False
        assert is_persona("jupyter-ai-personas::custom::MyPersona") is True


def _make_real_chat(root_dir) -> WsChatModel:
    """Create a *real* RTC-free chat model (no mock).

    ``observe_messages`` / ``unobserve_messages`` / ``add_message`` behave exactly
    as in production; only the observer callbacks stay plain test spies.
    """
    model = WsChatModel(path="test-room.chat", root_dir=Path(root_dir))
    model.load_from_file()
    return model


class TestMessageRouter:
    """Test MessageRouter functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.router = MessageRouter()
        self.mock_chat_init_callback = Mock()
        self.mock_slash_cmd_callback = Mock()
        self.mock_msg_callback = Mock()
        self.mock_specific_cmd_callback = Mock()
        self._tmp = tempfile.TemporaryDirectory()
        self.mock_chat = _make_real_chat(self._tmp.name)

    def teardown_method(self):
        self._tmp.cleanup()

    def test_router_initialization(self):
        """Test router initializes correctly."""
        router = MessageRouter()
        assert len(router.chat_init_observers) == 0
        assert len(router.slash_cmd_observers) == 0
        assert len(router.chat_msg_observers) == 0
        assert len(router.active_chats) == 0

    def test_observe_chat_init(self):
        """Test registering chat init callback."""
        self.router.observe_chat_init(self.mock_chat_init_callback)
        assert self.mock_chat_init_callback in self.router.chat_init_observers

    def test_observe_slash_cmd_msg(self):
        """Test registering slash command callback."""
        room_id = "test-room"
        command_pattern = "help"
        self.router.observe_slash_cmd_msg(room_id, command_pattern, self.mock_slash_cmd_callback)
        assert command_pattern in self.router.slash_cmd_observers[room_id]
        assert self.mock_slash_cmd_callback in self.router.slash_cmd_observers[room_id][command_pattern]

    def test_observe_chat_msg(self):
        """Test registering regular message callback."""
        room_id = "test-room"
        self.router.observe_chat_msg(room_id, self.mock_msg_callback)
        assert self.mock_msg_callback in self.router.chat_msg_observers[room_id]

    def test_connect_chat(self):
        """Test connecting a chat to the router."""
        room_id = "test-room"
        self.router.observe_chat_init(self.mock_chat_init_callback)

        self.router.connect_chat(room_id, self.mock_chat)

        # Should store the chat and call init observers
        assert room_id in self.router.active_chats
        assert self.router.active_chats[room_id] == self.mock_chat
        self.mock_chat_init_callback.assert_called_once_with(room_id, self.mock_chat)
        # Should have registered a real message observer on the real model
        assert room_id in self.router.message_observers
        assert len(self.mock_chat._message_observers) == 1

    def test_disconnect_chat(self):
        """Test disconnecting a chat from the router."""
        room_id = "test-room"
        self.router.connect_chat(room_id, self.mock_chat)

        self.router.disconnect_chat(room_id)

        # Should remove the chat and unobserve the real model
        assert room_id not in self.router.active_chats
        assert room_id not in self.router.message_observers
        assert len(self.mock_chat._message_observers) == 0

    def test_message_routing(self):
        """Test message routing to appropriate callbacks."""
        room_id = "test-room"
        self.router.observe_slash_cmd_msg(room_id, "test", self.mock_slash_cmd_callback)
        self.router.observe_chat_msg(room_id, self.mock_msg_callback)

        # Test slash command routing
        slash_msg = Message(id="1", body="/test command", sender="user", time=123)
        self.router._route_message(room_id, slash_msg)
        
        # Should be called with clean command and trimmed message
        expected_calls = self.mock_slash_cmd_callback.call_args_list
        assert len(expected_calls) == 1
        call_args = expected_calls[0][0]  # Get positional args
        assert call_args[0] == room_id  # room_id
        assert call_args[1] == "test"   # clean command (no slash)
        assert call_args[2].body == "command"  # trimmed message body

        # Test regular message routing
        regular_msg = Message(id="2", body="Hello world", sender="user", time=124)
        self.router._route_message(room_id, regular_msg)
        self.mock_msg_callback.assert_called_once_with(room_id, regular_msg)

    def test_cleanup(self):
        """Test router cleanup."""
        room_id = "test-room"
        self.router.connect_chat(room_id, self.mock_chat)
        self.router.observe_chat_init(self.mock_chat_init_callback)

        self.router.cleanup()

        # Should clear all observers and active chats
        assert len(self.router.active_chats) == 0
        assert len(self.router.chat_init_observers) == 0
        assert len(self.router.slash_cmd_observers) == 0
        assert len(self.router.chat_msg_observers) == 0


    def test_matches_pattern_exact(self):
        """Test exact command matching."""
        assert matches_pattern("help", "help") is True
        assert matches_pattern("help", "status") is False

    def test_matches_pattern_regex(self):
        """Test regex pattern matching."""
        # Pattern with .* (formerly wildcard)
        assert matches_pattern("ai-generate", "ai-.*") is True
        assert matches_pattern("ai-review", "ai-.*") is True
        assert matches_pattern("help", "ai-.*") is False
        assert matches_pattern("export-csv", "export-.*") is True

    def test_matches_pattern_regex_groups(self):
        """Test regex command matching with groups."""
        pattern = r"export-(json|csv|xml)"
        assert matches_pattern("export-json", pattern) is True
        assert matches_pattern("export-csv", pattern) is True
        assert matches_pattern("export-xml", pattern) is True
        assert matches_pattern("export-pdf", pattern) is False

    def test_specific_command_routing_exact(self):
        """Test routing of specific slash commands with exact match."""
        room_id = "test-room"
        self.router.observe_slash_cmd_msg(room_id, "help", self.mock_specific_cmd_callback)
        
        # Test matching command
        help_msg = Message(id="1", body="/help topic", sender="user", time=123)
        self.router._route_message(room_id, help_msg)
        
        # Should be called with clean command and trimmed message
        expected_calls = self.mock_specific_cmd_callback.call_args_list
        assert len(expected_calls) == 1
        call_args = expected_calls[0][0]  # Get positional args
        assert call_args[0] == room_id  # room_id
        assert call_args[1] == "help"   # clean command (no slash)
        assert call_args[2].body == "topic"  # trimmed message body
        
        # Test non-matching command
        self.mock_specific_cmd_callback.reset_mock()
        status_msg = Message(id="2", body="/status", sender="user", time=124)
        self.router._route_message(room_id, status_msg)
        self.mock_specific_cmd_callback.assert_not_called()

    def test_specific_command_routing_regex(self):
        """Test routing of specific slash commands with regex pattern."""
        room_id = "test-room"
        self.router.observe_slash_cmd_msg(room_id, "ai-.*", self.mock_specific_cmd_callback)
        
        # Test matching commands
        generate_msg = Message(id="1", body="/ai-generate code", sender="user", time=123)
        self.router._route_message(room_id, generate_msg)
        
        # Check first call
        call_args = self.mock_specific_cmd_callback.call_args_list[0][0]
        assert call_args[0] == room_id
        assert call_args[1] == "ai-generate"  # clean command
        assert call_args[2].body == "code"    # trimmed body
        
        review_msg = Message(id="2", body="/ai-review file.py", sender="user", time=124)
        self.router._route_message(room_id, review_msg)
        
        # Check second call
        call_args = self.mock_specific_cmd_callback.call_args_list[1][0]
        assert call_args[0] == room_id
        assert call_args[1] == "ai-review"    # clean command
        assert call_args[2].body == "file.py" # trimmed body
        
        # Test non-matching command
        self.mock_specific_cmd_callback.reset_mock()
        help_msg = Message(id="3", body="/help", sender="user", time=125)
        self.router._route_message(room_id, help_msg)
        self.mock_specific_cmd_callback.assert_not_called()

    def test_specific_command_routing_command_passed(self):
        """Test that the actual command is passed to callbacks."""
        room_id = "test-room"
        self.router.observe_slash_cmd_msg(room_id, "export", self.mock_specific_cmd_callback)
        
        # Test command with arguments
        export_msg = Message(id="1", body="/export csv data.json output.csv", sender="user", time=123)
        self.router._route_message(room_id, export_msg)
        
        call_args = self.mock_specific_cmd_callback.call_args_list[0][0]
        assert call_args[0] == room_id
        assert call_args[1] == "export"  # clean command
        assert call_args[2].body == "csv data.json output.csv"  # trimmed body
        
        # Test command with no arguments
        self.mock_specific_cmd_callback.reset_mock()
        export_no_args = Message(id="2", body="/export", sender="user", time=124)
        self.router._route_message(room_id, export_no_args)
        
        call_args = self.mock_specific_cmd_callback.call_args_list[0][0]
        assert call_args[0] == room_id
        assert call_args[1] == "export"  # clean command
        assert call_args[2].body == ""       # empty trimmed body

    def test_multiple_specific_observers_same_pattern(self):
        """Test multiple observers for the same command pattern."""
        room_id = "test-room"
        callback1 = Mock()
        callback2 = Mock()
        
        self.router.observe_slash_cmd_msg(room_id, "help", callback1)
        self.router.observe_slash_cmd_msg(room_id, "help", callback2)
        
        help_msg = Message(id="1", body="/help topic", sender="user", time=123)
        self.router._route_message(room_id, help_msg)
        
        # Both callbacks should be called with clean command and trimmed message
        call_args1 = callback1.call_args_list[0][0]
        assert call_args1[0] == room_id
        assert call_args1[1] == "help"    # clean command
        assert call_args1[2].body == "topic"  # trimmed body
        
        call_args2 = callback2.call_args_list[0][0]
        assert call_args2[0] == room_id
        assert call_args2[1] == "help"    # clean command  
        assert call_args2[2].body == "topic"  # trimmed body

    def test_multiple_patterns_different_commands(self):
        """Test multiple patterns for different commands."""
        room_id = "test-room"
        help_callback = Mock()
        export_callback = Mock()
        
        self.router.observe_slash_cmd_msg(room_id, "help", help_callback)
        self.router.observe_slash_cmd_msg(room_id, "export-.*", export_callback)
        
        help_msg = Message(id="1", body="/help topic", sender="user", time=123)
        self.router._route_message(room_id, help_msg)
        
        call_args = help_callback.call_args_list[0][0]
        assert call_args[0] == room_id
        assert call_args[1] == "help"     # clean command
        assert call_args[2].body == "topic"  # trimmed body
        export_callback.assert_not_called()
        
        export_msg = Message(id="2", body="/export-csv file.csv", sender="user", time=124)
        self.router._route_message(room_id, export_msg)
        
        call_args = export_callback.call_args_list[0][0]
        assert call_args[0] == room_id
        assert call_args[1] == "export-csv"  # clean command
        assert call_args[2].body == "file.csv"  # trimmed body

    def test_specific_command_error_handling(self):
        """Test error handling in specific command observers."""
        room_id = "test-room"
        error_callback = Mock(side_effect=Exception("Test error"))
        self.router.observe_slash_cmd_msg(room_id, "help", error_callback)
        
        help_msg = Message(id="1", body="/help", sender="user", time=123)
        # Should not raise exception even if callback fails
        self.router._route_message(room_id, help_msg)
        
        call_args = error_callback.call_args_list[0][0]
        assert call_args[0] == room_id
        assert call_args[1] == "help"  # clean command
        assert call_args[2].body == ""  # empty trimmed body

    def test_invalid_regex_pattern(self):
        """Test handling of invalid regex patterns."""
        # Invalid regex should not match anything
        assert matches_pattern("help", "[invalid") is False

    def test_message_trimming_and_command_cleaning(self):
        """Test that messages are properly trimmed and commands cleaned."""
        room_id = "test-room"
        callback = Mock()
        self.router.observe_slash_cmd_msg(room_id, "test", callback)
        
        # Test various message formats
        test_cases = [
            ("/test hello world", "test", "hello world"),
            ("/test", "test", ""),
            ("/test    multiple   spaces", "test", "multiple   spaces"),
            ("/test-command with-args", "test-command", "with-args"),
        ]
        
        for original_body, expected_command, expected_trimmed_body in test_cases:
            callback.reset_mock()
            msg = Message(id="1", body=original_body, sender="user", time=123)
            self.router._route_message(room_id, msg)
            
            if callback.called:  # Only check if command matched
                call_args = callback.call_args_list[0][0]
                assert call_args[0] == room_id
                assert call_args[1] == expected_command  # No forward slash
                assert call_args[2].body == expected_trimmed_body  # Trimmed body
                
                # Verify original message wasn't modified
                assert msg.body == original_body

    def test_message_metadata_preserved(self):
        """Test that all message metadata is preserved in trimmed message."""
        room_id = "test-room"
        callback = Mock()
        self.router.observe_slash_cmd_msg(room_id, "help", callback)

        original_msg = Message(
            id="test-id",
            body="/help getting-started",
            sender="test-user",
            time=123.456,
            mentions=["@someone"],
            attachments=["file1.txt"]
        )

        self.router._route_message(room_id, original_msg)

        call_args = callback.call_args_list[0][0]
        trimmed_msg = call_args[2]

        # Check that metadata is preserved
        assert trimmed_msg.id == original_msg.id
        assert trimmed_msg.sender == original_msg.sender
        assert trimmed_msg.time == original_msg.time
        assert trimmed_msg.mentions == original_msg.mentions
        assert trimmed_msg.attachments == original_msg.attachments

        # Only body should be different
        assert trimmed_msg.body == "getting-started"
        assert original_msg.body == "/help getting-started"  # Original unchanged

    def test_deleted_messages_not_routed(self):
        """Test that deleted messages are not routed to any callbacks."""
        room_id = "test-room"
        slash_callback = Mock()
        msg_callback = Mock()

        self.router.observe_slash_cmd_msg(room_id, "help", slash_callback)
        self.router.observe_chat_msg(room_id, msg_callback)

        # Test deleted slash command message
        deleted_slash_msg = Message(
            id="1",
            body="/help topic",
            sender="user",
            time=123,
            deleted=True
        )
        self.router._route_message(room_id, deleted_slash_msg)
        slash_callback.assert_not_called()

        # Test deleted regular message
        deleted_regular_msg = Message(
            id="2",
            body="Hello world",
            sender="user",
            time=124,
            deleted=True
        )
        self.router._route_message(room_id, deleted_regular_msg)
        msg_callback.assert_not_called()

        # Verify non-deleted messages still work
        normal_slash_msg = Message(
            id="3",
            body="/help topic",
            sender="user",
            time=125,
            deleted=False
        )
        self.router._route_message(room_id, normal_slash_msg)
        slash_callback.assert_called_once()

        normal_regular_msg = Message(
            id="4",
            body="Hello world",
            sender="user",
            time=126,
            deleted=False
        )
        self.router._route_message(room_id, normal_regular_msg)
        msg_callback.assert_called_once()



class TestPreExistingMessageFiltering:
    """Test that messages loaded from disk on reconnect are not routed."""

    def setup_method(self):
        self.router = MessageRouter()
        self._tmp = tempfile.TemporaryDirectory()
        self.mock_chat = _make_real_chat(self._tmp.name)
        self.msg_callback = Mock()
        self.slash_callback = Mock()

    def teardown_method(self):
        self._tmp.cleanup()

    def _make_event(self, message: Message, action: ChatMessageAction = ChatMessageAction.CLIENT_MSG_RECEIVED) -> ChatMessageEvent:
        """Create a ChatMessageEvent from a Message."""
        return ChatMessageEvent(action=action, message=message)

    def test_old_messages_skipped_on_reconnect(self):
        """Messages with timestamps before connect_chat should not be routed."""
        room_id = "test-room"
        old_time = time() - 60  # 1 minute ago

        self.router.connect_chat(room_id, self.mock_chat)
        self.router.observe_chat_msg(room_id, self.msg_callback)

        # Simulate old messages arriving
        old_msg1 = Message(id="1", body="old msg 1", sender="user", time=old_time)
        old_msg2 = Message(id="2", body="old msg 2", sender="user", time=old_time - 10)

        self.router._on_message_event(room_id, self._make_event(old_msg1))
        self.router._on_message_event(room_id, self._make_event(old_msg2))

        self.msg_callback.assert_not_called()

    def test_new_messages_routed_after_reconnect(self):
        """Messages with timestamps after connect_chat should be routed."""
        room_id = "test-room"

        self.router.connect_chat(room_id, self.mock_chat)
        self.router.observe_chat_msg(room_id, self.msg_callback)

        new_time = time() + 1
        new_msg = Message(id="1", body="new msg", sender="user", time=new_time)
        self.router._on_message_event(room_id, self._make_event(new_msg))

        self.msg_callback.assert_called_once()
        routed_msg = self.msg_callback.call_args[0][1]
        assert routed_msg.body == "new msg"

    def test_mixed_old_and_new_messages(self):
        """Only new messages should be routed when old and new arrive."""
        room_id = "test-room"
        old_time = time() - 60

        self.router.connect_chat(room_id, self.mock_chat)
        self.router.observe_chat_msg(room_id, self.msg_callback)

        new_time = time() + 1
        old_msg = Message(id="1", body="old", sender="user", time=old_time)
        new_msg = Message(id="2", body="new", sender="user", time=new_time)

        self.router._on_message_event(room_id, self._make_event(old_msg))
        self.router._on_message_event(room_id, self._make_event(new_msg))

        assert self.msg_callback.call_count == 1
        routed_msg = self.msg_callback.call_args[0][1]
        assert routed_msg.body == "new"

    def test_old_slash_commands_skipped(self):
        """Old slash command messages should also be skipped."""
        room_id = "test-room"
        old_time = time() - 60

        self.router.connect_chat(room_id, self.mock_chat)
        self.router.observe_slash_cmd_msg(room_id, "help", self.slash_callback)

        old_msg = Message(id="1", body="/help topic", sender="user", time=old_time)
        self.router._on_message_event(room_id, self._make_event(old_msg))

        self.slash_callback.assert_not_called()

    def test_connected_at_cleaned_on_disconnect(self):
        """Disconnect should clean up the connected_at timestamp."""
        room_id = "test-room"
        self.router.connect_chat(room_id, self.mock_chat)
        assert room_id in self.router._connected_at

        self.router.disconnect_chat(room_id)
        assert room_id not in self.router._connected_at

    def test_non_client_messages_ignored(self):
        """Only CLIENT_MSG_RECEIVED events should be routed."""
        room_id = "test-room"

        self.router.connect_chat(room_id, self.mock_chat)
        self.router.observe_chat_msg(room_id, self.msg_callback)

        new_time = time() + 1
        msg = Message(id="1", body="server msg", sender="persona", time=new_time)

        # Server-sent messages should be ignored
        self.router._on_message_event(
            room_id, ChatMessageEvent(action=ChatMessageAction.SERVER_MSG_SENT, message=msg)
        )
        self.msg_callback.assert_not_called()

        # Server updates should be ignored
        self.router._on_message_event(
            room_id, ChatMessageEvent(action=ChatMessageAction.SERVER_MSG_UPDATED, message=msg)
        )
        self.msg_callback.assert_not_called()

        # Client edits should be ignored
        self.router._on_message_event(
            room_id, ChatMessageEvent(action=ChatMessageAction.CLIENT_MSG_EDITED, message=msg)
        )
        self.msg_callback.assert_not_called()

        # Only client received messages are routed
        self.router._on_message_event(
            room_id, ChatMessageEvent(action=ChatMessageAction.CLIENT_MSG_RECEIVED, message=msg)
        )
        self.msg_callback.assert_called_once()



# ===========================================================================
# RTC matrix integration tests (issue #48): the three observers end-to-end
# against the model the live ChatManager resolves for the active transport.
# ===========================================================================
class _Recorder:
    """A plain recording callback -- not a mock, just a list-appending callable."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, *args: Any) -> None:
        self.calls.append(args)

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
    the identifier the router uses in ``active_chats`` -- the transport-neutral
    chat id (``model.get_id()``), regardless of transport.
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
        # event -> the ChatManager forwards it, caches the YChat (by chat id),
        # and emits OPENED. `get_document(copy=False)` returns that same YChat.
        model = await ydoc_api.get_document(**kwargs)
        assert model is not None
        key = model.get_id()
        assert await _pump_until(
            lambda: manager.get(key) is not None
        ), "ChatManager never cached the YChat for the opened room"
    else:
        # Exactly what WSChatHandler.open() calls for the first client; emits
        # OPENED. ws_open returns the live model.
        model = manager.ws_open(path)
        assert model is not None
        key = model.get_id()
    # The router keys active chats on the transport-neutral chat id (chat.get_id()),
    # not the transport room id / path.
    # The OPENED event is handled by the router's async listener; wait until it
    # has actually connected the chat (registering observe_messages) so message
    # delivery is not racy.
    assert await _pump_until(
        lambda: key in router.active_chats
    ), "router did not connect the chat after OPENED"
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


async def _close_chat(app, path: str, chat_id: str) -> None:
    """Close the chat the way the server closes it (emits the real CLOSED event)."""
    manager = _chat_manager(app)
    if manager._rtc_enabled:
        # Drive the real RTC forwarder with the provider's `clean` room event.
        # The handler resolves the chat by its `path`, so any guard-passing room
        # id (``{format}:chat:{id}``) works here.
        await manager._on_rtc_room_event(
            None,
            JUPYTER_COLLABORATION_EVENTS_URI,
            {"room": "json:chat:integration", "path": path, "action": "clean"},
        )
    else:
        # Last client gone with no active writers -> the manager frees + emits
        # CLOSED. The manager keys on the chat id.
        manager.ws_client_gone(chat_id)


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
        assert key not in router._connected_at
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


def test_stop_observer_fires_on_real_ws_disconnect(
    jp_serverapp, jp_asyncio_loop, jp_ws_fetch, jp_root_dir
):
    """A real client disconnect must fire observe_chat_stop (issue #47).

    Unlike the lifecycle test (which asks the ChatManager to close the chat),
    this drives the *origination* end-to-end: a real WebSocket client connects to
    the jupyterlab_chat handler and then disconnects, with no manual ChatManager
    calls. The handler's ``on_close`` -> ``ws_client_gone`` -> ``_free(CLOSED)``
    must reach the router and fire the stop observer.

    RTC-free only: the plain WS chat endpoint exists only when RTC is off, and
    real RTC room teardown ("clean") is emitted internally by the collaboration
    provider (covered by the reaction test, not driven by a client here).
    """
    app = jp_serverapp
    if _chat_manager(app)._rtc_enabled:
        pytest.skip("WS chat endpoint exists only RTC-free; RTC 'clean' is provider-internal")
    router = _router(app)
    stop = _Recorder()
    router.observe_chat_stop(stop)
    path = _chat_file(jp_root_dir, "router-realstop.chat")

    async def scenario():
        await _wait_router_subscribed(app)
        # Real client connects -> ws_open -> OPENED -> router connects. The
        # router keys on the transport-neutral chat id; grab it once connected.
        # jupyterlab_chat serves the RTC-free chat socket at
        # ``/api/chat/ws/<url-encoded path>`` -- the chat path is a URL segment,
        # not a ``?path=`` query param (jupyterlab_chat >= 0.25.0rc0).
        ws = await jp_ws_fetch("api", "chat", "ws", quote(path, safe=""))
        assert await _pump_until(
            lambda: len(router.active_chats) == 1
        ), "router did not connect the chat on a real ws open"
        chat_id = next(iter(router.active_chats))
        # Real client disconnects -> on_close -> ws_client_gone -> _free(CLOSED).
        ws.close()
        assert await _pump_until(
            lambda: any(stop.calls)
        ), "observe_chat_stop did not fire on a real ws disconnect"
        assert stop.calls[0][0] == chat_id
        assert chat_id not in router.active_chats

    jp_asyncio_loop.run_until_complete(scenario())


def test_cleanup_clears_router(jp_serverapp, jp_asyncio_loop, jp_root_dir):
    """``cleanup`` disconnects every active chat and clears all observers."""
    app = jp_serverapp
    router = _router(app)
    router.observe_chat_init(_Recorder())
    path = _chat_file(jp_root_dir, "router-cleanup.chat")

    async def scenario():
        await _wait_router_subscribed(app)
        _model, key = await _open_chat(app, path)
        router.observe_chat_msg(key, _Recorder())
        router.observe_slash_cmd_msg(key, "help", _Recorder())
        assert key in router.active_chats

        router.cleanup()

        assert not router.active_chats
        assert not router.chat_init_observers
        assert not router.chat_stop_observers
        assert not router.chat_msg_observers
        assert not router.slash_cmd_observers

    jp_asyncio_loop.run_until_complete(scenario())
