"""
MessageRouter that manages message routing with callbacks.

This module provides a MessageRouter that:
- Handles new chat connections
- Routes slash commands and regular messages via callbacks
- Manages lifecycle and cleanup
"""

from typing import Any, Callable, Dict, List, TYPE_CHECKING
from functools import partial
import re
from dataclasses import replace
from jupyterlab_chat.models import Message
from pycrdt import ArrayEvent
from traitlets.config import LoggingConfigurable

if TYPE_CHECKING:
    from jupyterlab_chat.ychat import YChat

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
    Router that manages ychat message routing.

    The Router provides three callback points:
    1. When new chats are initialized
    2. When slash commands are received
    3. When regular (non-slash) messages are received
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Callback lists
        self.chat_init_observers: List[Callable[[str, "YChat"], Any]] = []
        self.slash_cmd_observers: Dict[str, Dict[str, List[Callable[[str, str, Message], Any]]]] = {}
        self.chat_msg_observers: Dict[str, List[Callable[[str, Message], Any]]] = {}
        self.chat_reset_observers: List[Callable[[str, "YChat"], Any]] = []

        # Active chat rooms
        self.active_chats: Dict[str, "YChat"] = {}

        # Root observers for keeping track of incoming messages
        self.message_observers: Dict[str, Callable] = {}

    def observe_chat_init(self, callback: Callable[[str, "YChat"], Any]) -> None:
        """
        Register a callback for when new chats are initialized.

        Args:
            callback: Function called with (room_id: str, ychat: YChat) when chat connects
        """
        self.chat_init_observers.append(callback)
        self.log.info("Registered new chat initialization callback")
    
    def observe_chat_reset(self, callback: Callable[[str, "YChat"], Any]) -> None:
        """
        Register a callback for when a `YChat` document is reset. This will only
        occur if `jupyter_server_documents` is installed.

        Args:
            callback: Function called with (room_id: str, new_ychat: YChat) when chat resets
        """
        self.chat_reset_observers.append(callback)
        self.log.info("Registered new chat reset callback")
    
    def observe_slash_cmd_msg(
        self, room_id: str, command_pattern: str, callback: Callable[[str, str, Message], Any]
    ) -> None:
        """
        Register a callback for when specific slash commands are received.

        Args:
            room_id: The chat room ID
            command_pattern: Regex pattern to match commands (without leading slash).
                Examples:
                - Exact match: "help" matches "/help"
                - Pattern match: "ai-.*" matches "/ai-generate", "/ai-review", etc.
                - Multiple options: "export-(json|csv)" matches "/export-json", "/export-csv"
            callback: Function called with (room_id: str, command: str, message: Message) for matching commands
        """
        if room_id not in self.slash_cmd_observers:
            self.slash_cmd_observers[room_id] = {}
        
        if command_pattern not in self.slash_cmd_observers[room_id]:
            self.slash_cmd_observers[room_id][command_pattern] = []

        self.slash_cmd_observers[room_id][command_pattern].append(callback)
        self.log.info(f"Registered slash command callback for pattern: {command_pattern}")

    def observe_chat_msg(
        self, room_id: str, callback: Callable[[str, Message], Any]
    ) -> None:
        """
        Register a callback for when regular (non-slash) messages are received.

        Args:
            callback: Function called with (room_id: str, message: Message) for regular messages
        """
        if room_id not in self.chat_msg_observers:
            self.chat_msg_observers[room_id] = []

        self.chat_msg_observers[room_id].append(callback)
        self.log.info("Registered message callback")

    def connect_chat(self, room_id: str, ychat: "YChat") -> None:
        """
        Connect a new chat session to the router.

        Args:
            room_id: Unique identifier for the chat room
            ychat: YChat instance for the room
        """
        if room_id in self.active_chats:
            self.log.warning(f"Chat {room_id} already connected to router")
            return

        self.active_chats[room_id] = ychat

        # Set up message observer
        callback = partial(self._on_message_change, room_id, ychat)
        ychat.ymessages.observe(callback)
        self.message_observers[room_id] = callback

        self.log.info(f"Connected chat {room_id} to router")

        # Notify new chat observers
        self._notify_chat_init_observers(room_id, ychat)

    def disconnect_chat(self, room_id: str) -> None:
        """
        Disconnect a chat session from the router.

        Args:
            room_id: Unique identifier for the chat room
        """
        if room_id not in self.active_chats:
            return

        # Remove message observer
        if room_id in self.message_observers:
            ychat = self.active_chats[room_id]
            try:
                ychat.ymessages.unobserve(self.message_observers[room_id])
            except Exception as e:
                self.log.warning(f"Failed to unobserve chat {room_id}: {e}")
            del self.message_observers[room_id]

        del self.active_chats[room_id]
        self.log.info(f"Disconnected chat {room_id} from router")

    def _on_message_change(
        self, room_id: str, ychat: "YChat", events: ArrayEvent
    ) -> None:
        """Handle incoming messages from YChat."""
        for change in events.delta:  # type: ignore[attr-defined]
            if "insert" not in change.keys():
                continue

            # Process new messages (filter out raw_time duplicates)
            new_messages = [
                Message(**m) for m in change["insert"] if not m.get("raw_time", False)
            ]

            for message in new_messages:
                self._route_message(room_id, message)

    def _route_message(self, room_id: str, message: Message) -> None:
        """
        Route an incoming message to appropriate observers.

        Args:
            room_id: The chat room ID
            message: The message to route
        """
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
            clean_command = command[1:] if command.startswith("/") else command

            # Route to slash command observers
            self._notify_slash_cmd_observers(room_id, trimmed_message, clean_command)
        else:
            self._notify_msg_observers(room_id, message)


    def _notify_slash_cmd_observers(self, room_id: str, message: Message, clean_command: str) -> None:
        """Notify observers registered for slash commands."""
        room_observers = self.slash_cmd_observers.get(room_id, {})

        for registered_pattern, callbacks in room_observers.items():
            if matches_pattern(clean_command, registered_pattern):
                for callback in callbacks:
                    try:
                        callback(room_id, clean_command, message)
                    except Exception as e:
                        self.log.error(f"Slash command observer error for pattern '{registered_pattern}': {e}")

    def _notify_chat_init_observers(self, room_id: str, ychat: "YChat") -> None:
        """Notify all new chat observers."""
        for callback in self.chat_init_observers:
            try:
                callback(room_id, ychat)
            except Exception as e:
                self.log.error(f"New chat observer error for {room_id}: {e}")

    def _notify_msg_observers(self, room_id: str, message: Message) -> None:
        """Notify all message observers."""
        callbacks = self.chat_msg_observers.get(room_id, [])
        for callback in callbacks:
            try:
                callback(room_id, message)
            except Exception as e:
                self.log.error(f"Message observer error for {room_id}: {e}")
    
    def _on_chat_reset(self, room_id, ychat: "YChat") -> None:
        """
        Method to call when the YChat undergoes a document reset, e.g. when the
        `.chat` file is modified directly on disk.
        
        NOTE: Document resets will only occur when `jupyter_server_documents` is
        installed.
        """
        self.log.warning(f"Detected `YChat` document reset in room '{room_id}'.")
        for callback in self.chat_reset_observers:
            try:
                callback(room_id, ychat)
            except Exception as e:
                self.log.error(f"Reset chat observer error for {room_id}: {e}")

    def cleanup(self) -> None:
        """Clean up router resources."""
        self.log.info("Cleaning up MessageRouter...")

        # Disconnect all chats
        room_ids = list(self.active_chats.keys())
        for room_id in room_ids:
            self.disconnect_chat(room_id)

        # Clear callbacks
        self.chat_init_observers.clear()
        self.slash_cmd_observers.clear()
        self.chat_msg_observers.clear()

        self.log.info("MessageRouter cleanup complete")
