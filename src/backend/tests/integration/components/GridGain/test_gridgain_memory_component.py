import pytest
import uuid
from langchain_core.messages import HumanMessage, AIMessage

class TestGridGainChatMemory:
    @pytest.fixture(params=['pyignite', 'pygridgain'])
    def gridgain_memory(self, request):
        """
        Fixture to create GridGainChatMemory instances for both client types
        """
        # Import the component dynamically to match the original implementation
        from base.langflow.components.memories.GridGainChatMemory import GridGainChatMemory

        # Generate a unique session ID for each test
        session_id = str(uuid.uuid4())
        
        # Create an instance using the input parameters
        memory = GridGainChatMemory()
        memory.host = 'localhost'
        memory.port = '10800'
        memory.cache_name = 'test_langchain_message_store'
        memory.session_id = session_id
        memory.client_type = request.param
        
        yield memory
        
        # Cleanup: Clear the cache after each test
        try:
            import pyignite
            import pygridgain
            
            if request.param == 'pyignite':
                client = pyignite.Client()
            else:
                client = pygridgain.Client()
            
            client.connect('localhost', 10800)
            client.cache_clear('test_langchain_message_store')
        except Exception as e:
            print(f"Cleanup failed: {e}")

    def test_build_message_history_connection(self, gridgain_memory):
        """
        Test that a message history can be built successfully
        """
        try:
            message_history = gridgain_memory.build_message_history()
            assert message_history is not None, "Message history should not be None"
        except Exception as e:
            pytest.fail(f"Failed to build message history: {e}")

    def test_message_storage_and_retrieval(self, gridgain_memory):
        """
        Test storing and retrieving messages
        """
        # Build message history
        message_history = gridgain_memory.build_message_history()
        
        # Add some messages
        test_messages = [
            HumanMessage(content="Hello, how are you?"),
            AIMessage(content="I'm doing well, thank you!"),
            HumanMessage(content="What's the weather like?")
        ]
        
        # Store messages
        for msg in test_messages:
            message_history.add_message(msg)
        
        # Retrieve messages
        stored_messages = message_history.messages
        
        # Assertions
        assert len(stored_messages) == len(test_messages), "Number of stored messages should match"
        
        for original, stored in zip(test_messages, stored_messages):
            assert original.content == stored.content, "Message content should match"
            assert type(original) == type(stored), "Message type should match"

    def test_different_session_ids(self):
        """
        Test that different session IDs create separate message histories
        """
        from base.langflow.components.memories.GridGainChatMemory import GridGainChatMemory

        # Create two instances with different session IDs
        session_id1 = str(uuid.uuid4())
        session_id2 = str(uuid.uuid4())

        memory1 = GridGainChatMemory()
        memory1.host = 'localhost'
        memory1.port = '10800'
        memory1.cache_name = 'test_langchain_message_store'
        memory1.session_id = session_id1
        memory1.client_type = 'pyignite'

        memory2 = GridGainChatMemory()
        memory2.host = 'localhost'
        memory2.port = '10800'
        memory2.cache_name = 'test_langchain_message_store'
        memory2.session_id = session_id2
        memory2.client_type = 'pyignite'

        # Add messages to first session
        history1 = memory1.build_message_history()
        history1.add_message(HumanMessage(content="Session 1 message"))

        # Add messages to second session
        history2 = memory2.build_message_history()
        history2.add_message(HumanMessage(content="Session 2 message"))

        # Verify messages are separate
        assert len(history1.messages) == 1
        assert len(history2.messages) == 1
        assert history1.messages[0].content != history2.messages[0].content

    def test_invalid_client_type(self):
        """
        Test that an invalid client type raises a ValueError
        """
        from base.langflow.components.memories.GridGainChatMemory import GridGainChatMemory

        memory = GridGainChatMemory()
        memory.host = 'localhost'
        memory.port = '10800'
        memory.cache_name = 'test_langchain_message_store'
        memory.client_type = 'invalid_client'

        with pytest.raises(ValueError, match="Invalid client_type. Must be either 'pyignite' or 'pygridgain'."):
            memory.build_message_history()

    def test_connection_failure(self):
        """
        Test connection failure to GridGain server
        """
        from base.langflow.components.memories.GridGainChatMemory import GridGainChatMemory

        memory = GridGainChatMemory()
        memory.host = 'nonexistent_host'
        memory.port = '99999'
        memory.cache_name = 'test_langchain_message_store'
        memory.client_type = 'pyignite'

        with pytest.raises(ConnectionError, match="Failed to connect to GridGain server"):
            memory.build_message_history()

