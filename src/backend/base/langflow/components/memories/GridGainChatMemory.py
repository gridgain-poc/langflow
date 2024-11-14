
from langflow.base.memory.model import LCChatMemoryComponent
from langflow.field_typing import BaseChatMessageHistory
from langflow.inputs import MessageTextInput, StrInput
import pyignite
import pygridgain


class GridGainChatMemory(LCChatMemoryComponent):
    """Chat memory component that stores history in GridGain/Apache Ignite."""
    
    display_name = "GridGain Chat Memory"
    description = "Retrieves and stores chat messages using GridGain/Apache Ignite."
    name = "GridGainChatMemory"
    icon: str = "GridGain"

    inputs = [
        StrInput(
            name="host",
            display_name="Host",
            info="GridGain/Ignite server host address.",
            required=True,
            value="localhost",
        ),
        StrInput(
            name="port",
            display_name="Port",
            info="GridGain/Ignite server port number.",
            required=True,
            value="10800",
        ),
        StrInput(
            name="cache_name",
            display_name="Cache Name",
            info="The name of the cache within GridGain where messages will be stored.",
            required=True,
            value="langchain_message_store",
        ),
        MessageTextInput(
            name="session_id",
            display_name="Session ID",
            info="The session ID of the chat. If empty, the current session ID parameter will be used.",
            advanced=True,
        ),
        StrInput(
            name="client_type",
            display_name="Client Type",
            info="Type of client to use (pyignite or pygridgain).",
            required=True,
            value="pyignite",
        ),
    ]

    def build_message_history(self) -> BaseChatMessageHistory:
        """Build and return a GridGain chat message history instance."""
        try:
            from langchain_community.chat_message_histories.ignite import GridGainChatMessageHistory
        except ImportError as e:
            msg = (
                "Could not import GridGain chat message history implementation. "
                "Please ensure the implementation file is in the correct location."
            )
            raise ImportError(msg) from e

        # Create the appropriate client based on the specified type
        if self.client_type.lower() == "pyignite":
            client = pyignite.Client()
        elif self.client_type.lower() == "pygridgain":
            client = pygridgain.Client()
        else:
            raise ValueError(
                "Invalid client_type. Must be either 'pyignite' or 'pygridgain'."
            )

        # Connect to the GridGain server
        try:
            client.connect(self.host, int(self.port))
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to GridGain server at {self.host}:{self.port}: {str(e)}"
            )

        return GridGainChatMessageHistory(
            session_id=self.session_id,
            cache_name=self.cache_name,
            client=client,
        )