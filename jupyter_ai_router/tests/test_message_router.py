"""
Tests for MessageRouter functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock
from jupyterlab_chat.models import Message
from jupyterlab_chat.ychat import YChat
from jupyter_ai_router.router import MessageRouter, matches_pattern
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
        self.mock_specific_cmd_callback = Mock()
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
        self.router.connect_chat(room_id, self.mock_ychat)
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

