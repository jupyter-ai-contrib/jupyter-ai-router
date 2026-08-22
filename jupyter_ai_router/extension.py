from __future__ import annotations

import time
from typing import TYPE_CHECKING

from jupyter_server.extension.application import ExtensionApp
from jupyterlab_chat.events import ChatEventAction, ChatManager

from jupyter_ai_router.handlers import RouteHandler

from .router import MessageRouter

if TYPE_CHECKING:
    from jupyter_events import EventLogger
    from jupyterlab_chat.models import BaseChatModel


class RouterExtension(ExtensionApp):
    """
    Jupyter AI Router Extension
    """

    name = "jupyter_ai_router"
    handlers = [
        (r"jupyter-ai-router/health/?", RouteHandler),
    ]

    router: MessageRouter

    def initialize_settings(self):
        """Initialize router settings and event listeners."""
        start = time.time()

        # Create MessageRouter instance
        self.router = MessageRouter(parent=self)

        # Maps a ChatManager room_id (the transport-level identifier carried on
        # lifecycle events) to the transport-neutral chat id (chat.get_id()).
        # CLOSED/DELETED events carry no chat model, so we record the mapping at
        # OPENED time to translate them back to a chat id.
        self._chat_ids_by_room: dict[str, str] = {}

        # Make router available to other extensions
        if "jupyter-ai" not in self.settings:
            self.settings["jupyter-ai"] = {}
        self.settings["jupyter-ai"]["router"] = self.router

        # Subscribe to ChatManager lifecycle events for chat room discovery
        if self.serverapp is not None:
            import asyncio
            asyncio.get_event_loop().create_task(self._setup_chat_manager())

        elapsed = time.time() - start
        self.log.info(f"Initialized RouterExtension in {elapsed:.2f}s")

    async def _setup_chat_manager(self) -> None:
        """Wait for ChatManager to appear in settings, then subscribe.

        jupyterlab_chat may initialize after this extension, so the ChatManager
        instance may not be in settings yet. This mirrors the pattern used by
        PersonaManagerExtension to wait for the router.
        """
        import asyncio
        while True:
            chat_manager: ChatManager | None = self.serverapp.web_app.settings.get("chat_manager")
            if chat_manager is not None:
                chat_manager.observe_chats(self._on_chat_event)
                self.log.info("Subscribed to ChatManager lifecycle events")
                break
            await asyncio.sleep(0.1)

    async def _on_chat_event(
        self, logger: EventLogger, schema_id: str, data: dict
    ) -> None:
        """Handle chat lifecycle events from the ChatManager."""
        action = data.get("action")
        path = data.get("path", "")
        room_id = data.get("room_id") or path

        if action == ChatEventAction.OPENED.value:
            self.log.info(f"New chat detected: {room_id}")

            # Retrieve the chat model from ChatManager
            chat = self._get_chat(room_id)
            if chat is None:
                self.log.error(f"Failed to get chat model for {room_id}")
                return

            # Key the router on the transport-neutral chat id, not the room id.
            chat_id = chat.get_id()
            self._chat_ids_by_room[room_id] = chat_id

            # Connect chat to router
            self.router.connect_chat(chat_id, chat)

        elif action == ChatEventAction.CLOSED.value:
            self.log.info(f"Chat closed: {room_id}")
            chat_id = self._chat_ids_by_room.pop(room_id, None)
            if chat_id is None:
                return
            self.router.disconnect_chat(chat_id)
            self.router._notify_chat_stop_observers(chat_id)

        elif action == ChatEventAction.DELETED.value:
            self.log.info(f"Chat deleted: {room_id}")
            chat_id = self._chat_ids_by_room.pop(room_id, None)
            if chat_id is None:
                return
            self.router.disconnect_chat(chat_id)
            self.router._notify_chat_stop_observers(chat_id)

    def _get_chat(self, room_id: str) -> BaseChatModel | None:
        """
        Get the chat model for a room/path using the ChatManager.

        The ChatManager handles the transport difference internally:
        - RTC mode: resolves the YChat from the collaboration provider
        - WebSocket mode: returns the WsChatModel from its registry
        """
        chat_manager: ChatManager | None = self.serverapp.web_app.settings.get("chat_manager")
        if chat_manager is None:
            self.log.error("ChatManager not available in settings")
            return None

        return chat_manager.get(room_id)

    async def stop_extension(self):
        """Clean up router when extension stops."""
        try:
            if hasattr(self, "router"):
                self.router.cleanup()
        except Exception as e:
            self.log.error(f"Error during router cleanup: {e}")
