from __future__ import annotations
from typing import TYPE_CHECKING
import time
from jupyter_events import EventLogger
from jupyter_server.extension.application import ExtensionApp

from jupyter_ai_router.handlers import RouteHandler

from .router import MessageRouter

try:
    from jupyter_server_ydoc.utils import JUPYTER_COLLABORATION_EVENTS_URI
except ImportError:
    # Fallback if jupyter-collaboration is not available
    JUPYTER_COLLABORATION_EVENTS_URI = (
        "https://events.jupyter.org/jupyter_collaboration"
    )

# Define `JSD_PRESENT` to indicate whether `jupyter_server_documents` is
# installed in the current environment.
JSD_PRESENT = False
try:
    import jupyter_server_documents
    JSD_PRESENT = True
except ImportError:
    pass

if TYPE_CHECKING:
    from jupyterlab_chat.ychat import YChat


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

    async def _get_chat(self, room_id: str) -> YChat | None:
        """
        Get YChat instance for a room ID.

        Dispatches to either `_get_chat_jcollab()` or `_get_chat_jsd()` based on
        whether `jupyter_server_documents` is installed.
        """

        if JSD_PRESENT:
            return await self._get_chat_jsd(room_id)
        else:
            return await self._get_chat_jcollab(room_id)
    
    async def _get_chat_jcollab(self, room_id: str) -> YChat | None:
        """
        Method used to retrieve the `YChat` instance for a given room when
        `jupyter_server_documents` **is not** installed.
        """
        if not self.serverapp:
            return None

        try:
            collaboration = self.serverapp.web_app.settings["jupyter_server_ydoc"]
            document = await collaboration.get_document(room_id=room_id, copy=False)
            return document
        except Exception as e:
            self.log.error(f"Error getting chat document for {room_id}: {e}")
            return None
    
    async def _get_chat_jsd(self, room_id: str) -> YChat | None:
        """
        Method used to retrieve the `YChat` instance for a given room when
        `jupyter_server_documents` **is** installed.

        This method uniquely attaches a callback which is fired whenever the
        `YChat` is reset.
        """
        if not self.serverapp:
            return None

        try:
            jcollab_api = self.serverapp.web_app.settings["jupyter_server_ydoc"]
            yroom_manager = jcollab_api.yroom_manager
            yroom = yroom_manager.get_room(room_id)
            
            def _on_ychat_reset(new_ychat: YChat):
                self.router._on_chat_reset(room_id, new_ychat)

            ychat = await yroom.get_jupyter_ydoc(on_reset=_on_ychat_reset)
            return ychat
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
