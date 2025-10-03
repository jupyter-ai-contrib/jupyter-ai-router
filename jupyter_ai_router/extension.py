from __future__ import annotations
import time
from jupyter_events import EventLogger
from jupyter_server.extension.application import ExtensionApp

from jupyter_ai_router.handlers import RouteHandler

from .router import MessageRouter

# Check jupyter-collaboration version for compatibility
try:
    from jupyter_collaboration import __version__ as jupyter_collaboration_version

    JCOLLAB_VERSION = int(jupyter_collaboration_version[0])
    if JCOLLAB_VERSION >= 3:
        from jupyter_server_ydoc.utils import JUPYTER_COLLABORATION_EVENTS_URI
    else:
        from jupyter_collaboration.utils import JUPYTER_COLLABORATION_EVENTS_URI
except ImportError:
    # Fallback if jupyter-collaboration is not available
    JUPYTER_COLLABORATION_EVENTS_URI = (
        "https://events.jupyter.org/jupyter_collaboration"
    )


class RouterExtension(ExtensionApp):
    """
    Jupyter AI Router Extension
    """

    name = "jupyter_ai_router"
    handlers = [
        (r"jupyter-ai-router/health/?", RouteHandler),
    ]

    def initialize_settings(self):
        """Initialize router settings and event listeners."""
        start = time.time()

        # Create MessageRouter instance
        self.router = MessageRouter(parent=self)

        # Make router available to other extensions
        if "jupyter-ai" not in self.settings:
            self.settings["jupyter-ai"] = {}
        self.settings["jupyter-ai"]["router"] = self.router

        # Listen for new chat room events
        if self.serverapp is not None:
            self.event_logger = self.serverapp.web_app.settings["event_logger"]
            self.event_logger.add_listener(
                schema_id=JUPYTER_COLLABORATION_EVENTS_URI, listener=self._on_chat_event
            )

        elapsed = time.time() - start
        self.log.info(f"Initialized RouterExtension in {elapsed:.2f}s")

    async def _on_chat_event(
        self, logger: EventLogger, schema_id: str, data: dict
    ) -> None:
        """Handle chat room events and connect new chats to router."""
        # Only handle chat room initialization events
        if not (
            data["room"].startswith("text:chat:")
            and data["action"] == "initialize"
            and data["msg"] == "Room initialized"
        ):
            return

        room_id = data["room"]
        self.log.info(f"New chat room detected: {room_id}")

        # Get YChat document for the room
        ychat = await self._get_chat(room_id)
        if ychat is None:
            self.log.error(f"Failed to get YChat for room {room_id}")
            return

        # Connect chat to router
        self.router.connect_chat(room_id, ychat)

    async def _get_chat(self, room_id: str):
        """Get YChat instance for a room ID."""
        if not self.serverapp:
            return None

        try:
            if JCOLLAB_VERSION >= 3:
                collaboration = self.serverapp.web_app.settings["jupyter_server_ydoc"]
                document = await collaboration.get_document(room_id=room_id, copy=False)
            else:
                collaboration = self.serverapp.web_app.settings["jupyter_collaboration"]
                server = collaboration.ywebsocket_server
                room = await server.get_room(room_id)
                document = room._document

            return document
        except Exception as e:
            self.log.error(f"Error getting chat document for {room_id}: {e}")
            return None

    async def stop_extension(self):
        """Clean up router when extension stops."""
        try:
            if hasattr(self, "router"):
                self.router.cleanup()
        except Exception as e:
            self.log.error(f"Error during router cleanup: {e}")
