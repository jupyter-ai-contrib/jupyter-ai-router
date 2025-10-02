"""
Tests for MessageRouter functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock
from jupyterlab_chat.models import Message
from jupyterlab_chat.ychat import YChat
from jupyter_ai_router.router import MessageRouter
from jupyter_ai_router.utils import get_first_word, is_persona


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


class TestMessageRouter:
    """Test MessageRouter functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.router = MessageRouter()
        self.mock_chat_init_callback = Mock()
        self.mock_slash_cmd_callback = Mock()
        self.mock_msg_callback = Mock()
        self.mock_ychat = Mock(spec=YChat)
        self.mock_ychat.ymessages = Mock()

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
        self.router.observe_slash_cmd_msg(room_id, self.mock_slash_cmd_callback)
        assert self.mock_slash_cmd_callback in self.router.slash_cmd_observers[room_id]

    def test_observe_chat_msg(self):
        """Test registering regular message callback."""
        room_id = "test-room"
        self.router.observe_chat_msg(room_id, self.mock_msg_callback)
        assert self.mock_msg_callback in self.router.chat_msg_observers[room_id]

    def test_connect_chat(self):
        """Test connecting a chat to the router."""
        room_id = "test-room"
        self.router.observe_chat_init(self.mock_chat_init_callback)

        self.router.connect_chat(room_id, self.mock_ychat)

        # Should store the chat and call init observers
        assert room_id in self.router.active_chats
        assert self.router.active_chats[room_id] == self.mock_ychat
        self.mock_chat_init_callback.assert_called_once_with(room_id, self.mock_ychat)

    def test_disconnect_chat(self):
        """Test disconnecting a chat from the router."""
        room_id = "test-room"
        self.router.connect_chat(room_id, self.mock_ychat)

        self.router.disconnect_chat(room_id)

        # Should remove the chat
        assert room_id not in self.router.active_chats

    def test_message_routing(self):
        """Test message routing to appropriate callbacks."""
        room_id = "test-room"
        self.router.observe_slash_cmd_msg(room_id, self.mock_slash_cmd_callback)
        self.router.observe_chat_msg(room_id, self.mock_msg_callback)

        # Test slash command routing
        slash_msg = Message(id="1", body="/test command", sender="user", time=123)
        self.router._route_message(room_id, slash_msg)
        self.mock_slash_cmd_callback.assert_called_once_with(room_id, slash_msg)

        # Test regular message routing
        regular_msg = Message(id="2", body="Hello world", sender="user", time=124)
        self.router._route_message(room_id, regular_msg)
        self.mock_msg_callback.assert_called_once_with(room_id, regular_msg)

    def test_cleanup(self):
        """Test router cleanup."""
        room_id = "test-room"
        self.router.connect_chat(room_id, self.mock_ychat)
        self.router.observe_chat_init(self.mock_chat_init_callback)

        self.router.cleanup()

        # Should clear all observers and active chats
        assert len(self.router.active_chats) == 0
        assert len(self.router.chat_init_observers) == 0
        assert len(self.router.slash_cmd_observers) == 0
        assert len(self.router.chat_msg_observers) == 0
