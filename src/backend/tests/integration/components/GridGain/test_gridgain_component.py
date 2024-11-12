import unittest
from unittest.mock import MagicMock
import tempfile
import pandas as pd
from base.langflow.components.vectorstores.GridGain import GridGainVectorStoreComponent
from langchain.schema import Document

class TestGridGainVectorStoreIntegration(unittest.TestCase):
    def setUp(self):
        # Initialize the component with test configuration
        self.gridgain_component = GridGainVectorStoreComponent(
            cache_name="vector_cache",
            host="localhost",
            port=10800
        )
        # Mock embedding object with an embed_documents method
        self.gridgain_component.embedding = MagicMock()
        self.gridgain_component.embedding.embed_documents = lambda docs: [[0.1] * 128] * len(docs)

    def test_connect_to_ignite(self):
        """Test connecting to the GridGain server."""
        try:
            client = self.gridgain_component.connect_to_ignite("localhost", 10800)
            self.assertIsNotNone(client, "Failed to connect to GridGain server.")
        except Exception as e:
            self.fail(f"Connection to GridGain failed with exception: {e}")

    def test_build_vector_store(self):
        """Test building the vector store with a sample embedding."""
        try:
            vector_store = self.gridgain_component.build_vector_store()
            self.assertIsNotNone(vector_store, "Vector store build failed.")
        except Exception as e:
            self.fail(f"Vector store build failed with exception: {e}")

    def test_process_csv(self):
        """Test processing a CSV file to create Document objects."""
        # Create a temporary CSV file for testing
        csv_data = pd.DataFrame({
            'title': ['Title1', 'Title2'],
            'text': ['Text1', 'Text2'],
            'id': [1, 2],
            'url': ['http://example1.com', 'http://example2.com'],
            'vector_id': ['v1', 'v2']
        })
        
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
            csv_data.to_csv(tmp_csv.name, index=False)
            try:
                documents = self.gridgain_component.process_csv(tmp_csv.name)
                self.assertEqual(len(documents), 2, "CSV processing did not produce expected number of documents.")
                self.assertTrue(all(isinstance(doc, Document) for doc in documents), "CSV rows did not convert to Document instances.")
            finally:
                tmp_csv.close()

    def test_search_documents(self):
        """Test searching documents within the vector store."""
        # Set search query and number of results
        self.gridgain_component.search_query = "test search"
        self.gridgain_component.number_of_results = 2

        try:
            results = self.gridgain_component.search_documents()
            self.assertIsInstance(results, list, "Search results should be a list.")
            print(f"Found {len(results)} documents matching query.")
        except Exception as e:
            self.fail(f"Search failed with exception: {e}")

    def test_search_with_empty_query(self):
        """Test searching with an empty query, expecting no results."""
        self.gridgain_component.search_query = ""  # Empty query
        self.gridgain_component.number_of_results = 2

        try:
            results = self.gridgain_component.search_documents()
            self.assertEqual(len(results), 0, "Expected no results for empty search query.")
        except Exception as e:
            self.fail(f"Search with empty query failed with exception: {e}")

    def test_handle_invalid_host_port(self):
        """Test connection handling with an invalid host and port."""
        invalid_component = GridGainVectorStoreComponent(
            cache_name="vector_cache",
            host="invalid_host",
            port=12345
        )
        try:
            invalid_component.connect_to_ignite("invalid_host", 12345)
            self.fail("Expected connection failure with invalid host/port but connected successfully.")
        except Exception as e:
            self.assertIn("connect", str(e).lower(), "Expected connection failure with invalid host/port.")

if __name__ == '__main__':
    unittest.main()
