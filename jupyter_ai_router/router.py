"""
MessageRouter that manages message routing with callbacks.

This module provides a MessageRouter that:
- Handles new chat connections
- Routes slash commands and regular messages via callbacks
- Manages lifecycle and cleanup
"""

import time
from typing import Any, Callable, Dict, List, TYPE_CHECKING
from functools import partial
import re
from dataclasses import replace
from jupyterlab_chat.models import Message
from pycrdt import ArrayEvent
from traitlets.config import LoggingConfigurable
from jupyter_ydoc.ybasedoc import YBaseDoc

if TYPE_CHECKING:
    from jupyterlab_chat.ychat import YChat

from .utils import get_first_word


class UserTracker:
    """Tracks a user's observers and current document."""
    def __init__(self, username: str):
        self.username = username
        self.current_document: str | None = None
        self.observer_ids: List[int] = []
        self.room_states: Dict[str, Dict] = {}  # room_id -> awareness_state


class RoomTracker:
    """Tracks a room's observation state and metadata."""
    def __init__(self, room_id: str, ydoc: "YBaseDoc"):
        self.room_id = room_id
        self.ydoc = ydoc
        self.last_edit_time = 0.0
        self.last_trigger_time = 0.0
        self.active_users: set[str] = set()
        self.subscribers = {
            "awareness": None,
            "notebook": None
        }

    def start_observing(self, awareness_cb, notebook_cb):
        """Start observing this room's awareness and notebook changes."""
        self.subscribers["awareness"] = self.ydoc.awareness.observe(awareness_cb)
        self.subscribers["notebook"] = self.ydoc._ycells.observe_deep(notebook_cb)

    def stop_observing(self):
        """Stop observing this room and clean up subscriptions."""
        if self.subscribers["awareness"]:
            self.ydoc.awareness.unobserve(self.subscribers["awareness"])
        if self.subscribers["notebook"]:
            self.ydoc._ycells.unobserve(self.subscribers["notebook"])


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
        
        # Notebook observers
        self.notebook_reset_observers: List[Callable[[str, YBaseDoc], Any]] = []

        # Simplified tracking with just 2 main structures
        self.users: Dict[str, UserTracker] = {}  # username -> user_info
        self.rooms: Dict[str, RoomTracker] = {}  # room_id -> room_info

        # Global observer management
        self.observer_counter = 0
        self.trigger_cooldown = 1  # seconds between triggers
        self._observer_callbacks: Dict[int, Dict] = {}  # observer_id -> callback_info

        # Active chat rooms
        self.active_chats: Dict[str, "YChat"] = {}

        # Root observers for keeping track of incoming messages
        self.message_observers: Dict[str, Callable] = {}

        self.event_loop.create_task(self._start_observing_global_awareness())

    async def _room_id_from_path(self, path: str) -> str | None:
        room_id = await self.parent._room_id_from_path(path)
        return room_id
    
    async def _get_doc(self, room_id: str) -> YBaseDoc | None:
        doc = await self.parent._get_doc(room_id)
        return doc
    
    def _get_global_awareness(self):
        awareness = self.parent._get_global_awareness()
        return awareness

    @property
    def event_loop(self):
        """Return the event loop from parent."""
        return self.parent.event_loop
    
    async def _start_observing_global_awareness(self):
        awareness = self._get_global_awareness()
        awareness.observe(self._on_global_awareness_change)

    async def _start_observing_room(self, room_id: str, username: str):
        """Start observing a room's document and awareness changes."""
        if room_id in self.rooms:
            # Already observing, just add user
            self.rooms[room_id].active_users.add(username)
            self.log.info(f"Added user {username} to existing room observation {room_id}")
            return

        # Get the document for this room
        ydoc = await self._get_doc(room_id)
        if not ydoc:
            self.log.error(f"Could not get document for room {room_id}")
            return

        # Create room tracker
        room = RoomTracker(room_id, ydoc)
        room.active_users.add(username)

        # Set up observers
        awareness_cb = partial(self._on_awareness_change, room_id, ydoc)
        notebook_change_cb = partial(self._on_notebook_change, room_id)

        # Start observing using the RoomTracker
        room.start_observing(awareness_cb, notebook_change_cb)

        # Track this room
        self.rooms[room_id] = room

        self.log.info(f"Started observing room {room_id} for user {username}")

    def _stop_observing_room(self, room_id: str):
        """Stop observing a room when no users need it."""
        if room_id not in self.rooms:
            return

        room = self.rooms[room_id]

        # Remove observers using the RoomTracker
        try:
            room.stop_observing()
            self.log.info(f"Stopped observing awareness and notebook changes for room_id: {room_id}")
        except Exception as e:
            self.log.warning(f"Error stopping observation for room {room_id}: {e}")

        del self.rooms[room_id]
        self.log.info(f"Stopped observing room {room_id}")

    def _maybe_stop_observing_room(self, room_id: str, username: str):
        """Stop observing room if this was the last user using it."""
        if room_id not in self.rooms:
            return

        room = self.rooms[room_id]
        room.active_users.discard(username)

        if not room.active_users:
            self._stop_observing_room(room_id)
            self.log.info(f"Stopped observing room {room_id} - no more users")

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

    
    def observe_notebook_activity(
        self, username: str, callback: Callable[[Dict], Any]
    ) -> int:
        """
        Register a callback for notebook activity for a specific user.
        Returns observer_id for unregistering.
        """
        observer_id = self.observer_counter
        self.observer_counter += 1

        # Create or get user tracker
        if username not in self.users:
            self.users[username] = UserTracker(username)

        user = self.users[username]
        user.observer_ids.append(observer_id)

        # Store observer callback (we still need this for the actual callback)
        self._observer_callbacks[observer_id] = {
            "username": username,
            "callback": callback
        }

        # Check user's current notebook and start observing it
        self.event_loop.create_task(self._observe_active_notebook(username))

        self.log.info(f"Registered notebook activity observer {observer_id} for user {username}")
        return observer_id

    def unobserve_notebook_activity(self, observer_id: int) -> bool:
        """Remove a notebook activity observer by ID."""
        if observer_id not in self._observer_callbacks:
            return False

        observer_info = self._observer_callbacks[observer_id]
        username = observer_info["username"]

        # Remove from tracking
        del self._observer_callbacks[observer_id]

        if username in self.users:
            user = self.users[username]
            if observer_id in user.observer_ids:
                user.observer_ids.remove(observer_id)

            # If user has no more observers, clean up
            if not user.observer_ids:
                del self.users[username]

        # Check if we can stop observing any rooms
        self._cleanup_unused_room_observers()

        self.log.info(f"Unregistered observer {observer_id}")
        return True

    async def _observe_active_notebook(self, username: str):
        """Finds user's active notebook and starts observing it."""

        try:
            awareness = self._get_global_awareness()
            for _, state in awareness.states.items():
                state_username = state.get("user", {}).get("username", "")
                if state_username != username:
                    continue

                active_doc = state.get("current")
                if not (active_doc and active_doc.startswith("notebook:")):
                    continue

                # Update user's current document
                if username in self.users:
                    self.users[username].current_document = active_doc

                path = active_doc.split(":", maxsplit=1)[1]
                room_id = await self._room_id_from_path(path)

                self.log.info(f"Got room_id {room_id} in _observe_active_notebook")

                if room_id:
                    await self._start_observing_room(room_id, username)
                    self.log.info(f"Started observing {room_id} for newly registered user {username}")
                    break
        except Exception as e:
            self.log.error(f"Error checking {username}'s current notebook: {e}")

    def _cleanup_unused_room_observers(self):
        """Remove room observers that have no active users."""
        rooms_to_remove = []

        for room_id, room in self.rooms.items():
            # Check if any of this room's users still have active observers
            active_users = set()
            for username in room.active_users:
                if username in self.users and self.users[username].observer_ids:
                    active_users.add(username)

            if not active_users:
                rooms_to_remove.append(room_id)
            else:
                # Update the room's user list to only active users
                room.active_users = active_users

        # Remove unused rooms
        for room_id in rooms_to_remove:
            self._stop_observing_room(room_id)

    def _on_global_awareness_change(self, topic, updates):
        """
        Handle global awareness changes to track client document switching.
        """

        _, room = updates

        if isinstance(room, str):
            return

        awareness = room.get_awareness()
        for _, state in awareness.states.items():
            username = state.get("user", {}).get("username", None)
            if not (username and username in self.users):
                continue

            active_doc = state.get("current")

            if not (active_doc and active_doc.startswith("notebook:")):
                continue

            user = self.users[username]
            prev_doc = user.current_document

            # Check if user switched to a different document
            if prev_doc != active_doc:
                self.log.info(
                    f"User {username} switched from {prev_doc} to {active_doc}"
                )

                # Update stored current document
                user.current_document = active_doc

                self.event_loop.create_task(
                    self._handle_user_document_switch(username, active_doc, prev_doc)
                )

    async def _handle_user_document_switch(
        self, username: str, current_doc: str, prev_doc: str | None
    ):
        """Handle user switching documents - async version for room operations."""
        try:
            # Only handle users we have observers for
            if username not in self.users:
                return

            path = current_doc.split(":", maxsplit=1)[1]
            # Convert document path to room_id (async)
            room_id = await self._room_id_from_path(path)

            if room_id:

                # Start observing new room
                await self._start_observing_room(room_id, username)

                # Initialize user awareness state for new room if needed
                user = self.users[username]
                if room_id not in user.room_states:
                    user.room_states[room_id] = {
                        "active_cell": None,
                        "notebook_path": current_doc,
                        "last_check": 0
                    }
                    self.log.info(f"Initialized tracking for user {username} in room {room_id}")

                # Stop observing old room if needed
                if prev_doc:
                    old_path = prev_doc.split(":", maxsplit=1)[1]
                    old_room_id = await self._room_id_from_path(old_path)
                    if old_room_id:
                        self._maybe_stop_observing_room(old_room_id, username)

        except Exception as e:
            self.log.error(f"Error handling document switch for user {username}: {e}")

    def _on_notebook_change(self, room_id: str):
        """Handle notebook document changes and log event details."""

        # Save the timestamp that a change was made indicating notebook has changed
        current_time = time.time()
        if room_id in self.rooms:
            self.rooms[room_id].last_edit_time = current_time
        self.log.info(f"Notebook cells changed in {room_id} at {current_time}")
        

    def _on_awareness_change(self, room_id: str, ydoc: YBaseDoc, topic, updates):
        """Handle awareness changes for notebook activity tracking."""

        awareness_states = ydoc.awareness.states
        current_time = time.time()

        # Get room tracker
        if room_id not in self.rooms:
            return

        room = self.rooms[room_id]

        # Extract username from each client's state
        for _, state in awareness_states.items():
            username = state.get("user", {}).get("username", None)

            if not (username and username in self.users):
                continue

            user = self.users[username]
            active_cell = state.get("activeCellId")
            notebook_path = state.get("notebookPath")

            if not active_cell:
                continue

            # Get previous state for this user in this room
            prev_state = user.room_states.get(room_id, {})
            prev_active_cell = prev_state.get("active_cell")
            prev_check = prev_state.get("last_check", 0)

            # Skip if this was the first change
            if not prev_active_cell:
                user.room_states[room_id] = {
                    "active_cell": active_cell,
                    "notebook_path": notebook_path,
                    "last_check": current_time
                }
                continue

            if prev_active_cell != active_cell:
                # Check if there was a notebook change since last check
                if room.last_edit_time > prev_check:
                    # Check if enough time has passed since last trigger (debouncing)
                    if current_time - room.last_trigger_time >= self.trigger_cooldown:
                        room.last_trigger_time = current_time
                        self._notify_notebook_activity_observers(
                            username=username,
                            prev_active_cell=prev_active_cell,
                            notebook_path=notebook_path
                        )

                # Update stored state for this user
                user.room_states[room_id] = {
                    "active_cell": active_cell,
                    "notebook_path": notebook_path,
                    "last_check": current_time
                }

    def _notify_notebook_activity_observers(
        self, username: str, prev_active_cell: str, notebook_path: str
    ) -> None:
        """Notify all notebook activity observers."""

        if username not in self.users:
            return

        user = self.users[username]
        observer_ids = user.observer_ids

        for observer_id in observer_ids:
            if observer_id in self._observer_callbacks:
                callback = self._observer_callbacks[observer_id]["callback"]
                try:
                    callback(username, prev_active_cell, notebook_path)
                except Exception as e:
                    self.log.error(f"Notebook activity observer error for {username}: {e}")


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
        self.active_chats[room_id] = ychat
        for callback in self.chat_reset_observers:
            try:
                callback(room_id, ychat)
            except Exception as e:
                self.log.error(f"Reset chat observer error for {room_id}: {e}")

    def _on_notebook_reset(self, room_id, ydoc: YBaseDoc) -> None:
        """
        Method to call when the YDoc undergoes a document reset, e.g. when the
        `.ipynb` file is modified directly on disk.
        
        NOTE: Document resets will only occur when `jupyter_server_documents` is
        installed.
        """
        self.log.warning(f"Detected `YDoc` document reset in room '{room_id}'.")
        for callback in self.notebook_reset_observers:
            try:
                callback(room_id, ydoc)
            except Exception as e:
                self.log.error(f"Reset notebook observer error for {room_id}: {e}")

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
        self.chat_reset_observers.clear()

        self.log.info("MessageRouter cleanup complete")
