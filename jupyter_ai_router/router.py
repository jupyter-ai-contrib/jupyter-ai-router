"""
MessageRouter that manages message routing with callbacks.

This module provides a MessageRouter that:
- Handles new chat connections
- Routes slash commands and regular messages via callbacks
- Manages lifecycle and cleanup
"""

import re
from dataclasses import replace
from functools import partial
from time import time
from typing import Any, Callable

from jupyterlab_chat.models import (
    BaseChatModel,
    ChatMessageAction,
    ChatMessageEvent,
    Message,
    MessageObserver,
)
from traitlets.config import LoggingConfigurable

from .utils import get_first_word


def matches_pattern(word: str, pattern: str) -> bool:
    """
    Check if a word matches a regex pattern.

    Args:
        word: The word to match (e.g., "help", "ai-generate")
        pattern: The regex pattern to match against (e.g., "help", "ai-.*", "export-(json|csv)")

    Returns:
        True if the word matches the pattern
    """
    try:
        return bool(re.match(f"^{pattern}$", word))
    except re.error:
        return False


class MessageRouter(LoggingConfigurable):
    """
    Router that manages chat message routing.

    The Router provides three callback points:
    1. When new chats are initialized
    2. When slash commands are received
    3. When regular (non-slash) messages are received
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Callback lists
        self.chat_init_observers: list[Callable[[str, BaseChatModel], Any]] = []
        self.chat_stop_observers: list[Callable[[str], Any]] = []
        self.slash_cmd_observers: dict[str, dict[str, list[Callable[[str, str, Message], Any]]]] = {}
        self.chat_msg_observers: dict[str, list[Callable[[str, Message], Any]]] = {}

        # Active chat rooms
        self.active_chats: dict[str, BaseChatModel] = {}

        # Message observers (opaque handles for unsubscribing)
        self.message_observers: dict[str, MessageObserver] = {}

        # Timestamp recorded when each room is connected. Messages with
        # timestamps older than this are pre-existing (loaded from disk) and
        # should not be routed.
        self._connected_at: dict[str, float] = {}

    def observe_chat_init(self, callback: Callable[[str, BaseChatModel], Any]) -> None:
        """
        Register a callback for when new chats are initialized.

        Args:
            callback: Function called with (chat_id: str, chat: BaseChatModel) when chat connects
        """
        self.chat_init_observers.append(callback)
        self.log.info("Registered new chat initialization callback")

    def observe_chat_stop(self, callback: Callable[[str], Any]) -> None:
        """
        Register a callback for when a chat room is permanently closed
        (freed from memory).

        Args:
            callback: Function called with (chat_id: str) when the room is stopped.
        """
        self.chat_stop_observers.append(callback)
        self.log.info("Registered chat stop callback")

    def observe_slash_cmd_msg(
        self, chat_id: str, command_pattern: str, callback: Callable[[str, str, Message], Any]
    ) -> None:
        """
        Register a callback for when specific slash commands are received.

        Args:
            chat_id: The chat ID (chat.get_id())
            command_pattern: Regex pattern to match commands (without leading slash).
                Examples:
                - Exact match: "help" matches "/help"
                - Pattern match: "ai-.*" matches "/ai-generate", "/ai-review", etc.
                - Multiple options: "export-(json|csv)" matches "/export-json", "/export-csv"
            callback: Function called with (chat_id: str, command: str, message: Message) for matching commands
        """
        if chat_id not in self.slash_cmd_observers:
            self.slash_cmd_observers[chat_id] = {}

        if command_pattern not in self.slash_cmd_observers[chat_id]:
            self.slash_cmd_observers[chat_id][command_pattern] = []

        self.slash_cmd_observers[chat_id][command_pattern].append(callback)
        self.log.info(f"Registered slash command callback for pattern: {command_pattern}")

    def observe_chat_msg(
        self, chat_id: str, callback: Callable[[str, Message], Any]
    ) -> None:
        """
        Register a callback for when regular (non-slash) messages are received.

        Args:
            callback: Function called with (chat_id: str, message: Message) for regular messages
        """
        if chat_id not in self.chat_msg_observers:
            self.chat_msg_observers[chat_id] = []

        self.chat_msg_observers[chat_id].append(callback)
        self.log.info("Registered message callback")

    def connect_chat(self, chat_id: str, chat: BaseChatModel) -> None:
        """
        Connect a new chat session to the router.

        Args:
            chat_id: Unique identifier for the chat (chat.get_id())
            chat: BaseChatModel instance for the room
        """
        if chat_id in self.active_chats:
            self.log.warning(f"Chat {chat_id} already connected to router")
            return

        self.active_chats[chat_id] = chat

        # Record the current time so we can distinguish pre-existing messages
        # (loaded from disk after this point) from genuinely new messages.
        self._connected_at[chat_id] = time()

        # Set up message observer using the transport-neutral API
        callback = partial(self._on_message_event, chat_id)
        observer = chat.observe_messages(callback)
        self.message_observers[chat_id] = observer

        self.log.info(f"Connected chat {chat_id} to router")

        # Notify new chat observers
        self._notify_chat_init_observers(chat_id, chat)

    def disconnect_chat(self, chat_id: str) -> None:
        """
        Disconnect a chat session from the router.

        Args:
            chat_id: Unique identifier for the chat (chat.get_id())
        """
        if chat_id not in self.active_chats:
            return

        # Remove message observer
        if chat_id in self.message_observers:
            chat = self.active_chats[chat_id]
            try:
                chat.unobserve_messages(self.message_observers[chat_id])
            except Exception as e:
                self.log.warning(f"Failed to unobserve chat {chat_id}: {e}")
            del self.message_observers[chat_id]

        del self.active_chats[chat_id]
        self.slash_cmd_observers.pop(chat_id, None)
        self.chat_msg_observers.pop(chat_id, None)
        self._connected_at.pop(chat_id, None)
        self.log.info(f"Disconnected chat {chat_id} from router")

    def _on_message_event(self, chat_id: str, event: ChatMessageEvent) -> None:
        """Handle incoming message events from a chat model.

        Only routes new messages received from clients (human users).
        Ignores edits, server-sent messages, and streaming updates.
        """
        # Only route new messages from clients (human users)
        if event.action != ChatMessageAction.CLIENT_MSG_RECEIVED:
            return

        message = event.message

        # Skip messages that predate this connection (loaded from disk).
        connected_at = self._connected_at.get(chat_id, 0)
        if message.time < connected_at:
            return

        self._route_message(chat_id, message)

    def _route_message(self, chat_id: str, message: Message) -> None:
        """
        Route an incoming message to appropriate observers.

        Args:
            chat_id: The chat ID (chat.get_id())
            message: The message to route
        """

        if message.deleted:
            return

        first_word = get_first_word(message.body)

        # Check if it's a slash command
        if first_word and first_word.startswith("/"):
            # Extract command and create trimmed message
            parts = message.body.split(None, 1)  # Split into max 2 parts
            command = parts[0] if parts else ""
            trimmed_body = parts[1] if len(parts) > 1 else ""

            # Create a copy of the message with trimmed body (command removed)
            trimmed_message = replace(message, body=trimmed_body)

            # Remove forward slash from command for cleaner API
            clean_command = command.removeprefix("/")

            # Route to slash command observers; fall through to regular message
            # observers when no registered pattern matches
            handled = self._notify_slash_cmd_observers(chat_id, trimmed_message, clean_command)
            if not handled:
                self._notify_msg_observers(chat_id, message)
        else:
            self._notify_msg_observers(chat_id, message)

    def _notify_slash_cmd_observers(self, chat_id: str, message: Message, clean_command: str) -> bool:
        """
        Notify observers registered for slash commands.

        Returns:
            True if at least one observer matched.
        """
        room_observers = self.slash_cmd_observers.get(chat_id, {})
        matched = False

        for registered_pattern, callbacks in room_observers.items():
            if matches_pattern(clean_command, registered_pattern):
                matched = True
                for callback in callbacks:
                    try:
                        callback(chat_id, clean_command, message)
                    except Exception as e:
                        self.log.error(f"Slash command observer error for pattern '{registered_pattern}': {e}")

        return matched

    def _notify_chat_init_observers(self, chat_id: str, chat: BaseChatModel) -> None:
        """Notify all new chat observers."""
        for callback in self.chat_init_observers:
            try:
                callback(chat_id, chat)
            except Exception as e:
                self.log.error(f"New chat observer error for {chat_id}: {e}")

    def _notify_chat_stop_observers(self, chat_id: str) -> None:
        """Notify all chat stop observers."""
        for callback in self.chat_stop_observers:
            try:
                callback(chat_id)
            except Exception as e:
                self.log.error(f"Chat stop observer error for {chat_id}: {e}")

    def _notify_msg_observers(self, chat_id: str, message: Message) -> None:
        """Notify all message observers."""
        callbacks = self.chat_msg_observers.get(chat_id, [])
        for callback in callbacks:
            try:
                callback(chat_id, message)
            except Exception as e:
                self.log.error(f"Message observer error for {chat_id}: {e}")

    def cleanup(self) -> None:
        """Clean up router resources."""
        self.log.info("Cleaning up MessageRouter...")

        # Disconnect all chats
        chat_ids = list(self.active_chats.keys())
        for chat_id in chat_ids:
            self.disconnect_chat(chat_id)

        # Clear callbacks
        self.chat_init_observers.clear()
        self.chat_stop_observers.clear()
        self.slash_cmd_observers.clear()
        self.chat_msg_observers.clear()

        self.log.info("MessageRouter cleanup complete")
