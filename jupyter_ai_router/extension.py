from __future__ import annotations
import asyncio
import json
from typing import TYPE_CHECKING
import time
from jupyter_events import EventLogger
from jupyter_server.extension.application import ExtensionApp
from jupyter_ydoc.ybasedoc import YBaseDoc

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

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop:
        """
        Returns a reference to the asyncio event loop.
        """
        return asyncio.get_event_loop_policy().get_event_loop()
    
    @property
    def fileid_manager(self):
        return self.serverapp.web_app.settings["file_id_manager"]


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
            self.event_loop.create_task(self._check_notebook_observer())
            
        elapsed = time.time() - start
        self.log.info(f"Initialized RouterExtension in {elapsed:.2f}s")

    
    async def _check_notebook_observer(self):
        await asyncio.sleep(20)
        def callback(username, prev_active_cell, notebook_path):
            self.log.info(
                f"notebook observer callback : {username=}, {prev_active_cell=}, {notebook_path=}"
            )

        jcollab_api = self.serverapp.web_app.settings["jupyter_server_ydoc"]
        yroom_manager = jcollab_api.yroom_manager
        yroom = yroom_manager.get_room("JupyterLab:globalAwareness")
        awareness = yroom.get_awareness()
        for _, state in awareness.states.items():
            if username := state.get("user", {}).get("username", None):
                self.router.observe_notebook_activity(
                    username=username, callback=callback
                )
                break
        

    def _get_global_awareness(self):
        # TODO: make this compatible with jcollab
        jcollab_api = self.serverapp.web_app.settings["jupyter_server_ydoc"]
        yroom_manager = jcollab_api.yroom_manager
        yroom = yroom_manager.get_room("JupyterLab:globalAwareness")
        return yroom.get_awareness()
    
    async def _room_id_from_path(self, path: str) -> str | None:
        """Returns room_id from document path"""
        # TODO: Make this compatible with jcollab
        yroom_manager = self.serverapp.web_app.settings["yroom_manager"]
        for room_id in yroom_manager._rooms_by_id:
            if room_id == "JupyterLab:globalAwareness":
                continue
            ydoc = await self._get_doc(room_id)
            state = ydoc.awareness.get_local_state()
            file_id = state["file_id"]
            ydoc_path = self.fileid_manager.get_path(file_id)
            if ydoc_path == path:
                print(f"Found match in path {path}")
                return room_id

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

    async def _on_notebook_event(
        self, logger: EventLogger, schema_id: str, data: dict
    ) -> None:
        """Handle notebook room events and connect new chats to router."""
        # Only handle notebook room initialization events
        if not (
            data["room"].startswith("json:notebook:")
            and data["action"] == "initialize"
            and data["msg"] == "Room initialized"
        ):
            return

        room_id = data["room"]
        self.log.info(f"New notebook room detected: {room_id}")

        # Get YDoc document for the room
        ydoc = await self._get_doc(room_id)
        if ydoc is None:
            self.log.error(f"Failed to get YDoc for room {room_id}")
            return

        # Connect notebook to router
        self.router.connect_notebook(room_id, ydoc)

    async def _get_doc(self, room_id: str) -> YBaseDoc | None:
        """
        Get YDoc instance for a room ID.

        Dispatches to either `_get_doc_jcollab()` or `_get_doc_jsd()` based on
        whether `jupyter_server_documents` is installed.
        """

        if JSD_PRESENT:
            return await self._get_doc_jsd(room_id)
        else:
            return await self._get_doc_jcollab(room_id)
        
    async def _get_doc_jcollab(self, room_id: str) -> YBaseDoc | None:
        """
        Method used to retrieve the `YDoc` instance for a given room when
        `jupyter_server_documents` **is not** installed.
        """
        if not self.serverapp:
            return None

        try:
            collaboration = self.serverapp.web_app.settings["jupyter_server_ydoc"]
            document = await collaboration.get_document(room_id=room_id, copy=False)
            return document
        except Exception as e:
            self.log.error(f"Error getting ydoc for {room_id}: {e}")
            return None
        
    async def _get_doc_jsd(self, room_id: str) -> YBaseDoc | None:
        """
        Method used to retrieve the `YDoc` instance for a given room when
        `jupyter_server_documents` **is** installed.

        This method uniquely attaches a callback which is fired whenever the
        `YDoc` is reset.
        """
        if not self.serverapp:
            return None

        try:
            jcollab_api = self.serverapp.web_app.settings["jupyter_server_ydoc"]
            yroom_manager = jcollab_api.yroom_manager
            yroom = yroom_manager.get_room(room_id)
            
            def _on_ydoc_reset(new_ydoc: YBaseDoc):
                self.router._on_notebook_reset(room_id, new_ydoc)

            ydoc = await yroom.get_jupyter_ydoc(on_reset=_on_ydoc_reset)
            return ydoc
        except Exception as e:
            self.log.error(f"Error getting ydoc for {room_id}: {e}")
            return None

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
