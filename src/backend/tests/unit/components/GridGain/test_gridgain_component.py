import pytest
import pandas as pd
from unittest.mock import Mock, patch, mock_open
from langchain.schema import Document
from langchain_community.vectorstores.ignite import GridGainVectorStore

from typing import List
import tempfile
import os

# Import the class to test
from base.langflow.components.vectorstores.GridGain import GridGainVectorStoreComponent


@pytest.fixture
def component():
    """Create a basic component instance for testing."""
    component = GridGainVectorStoreComponent()
    component.cache_name = "test_cache"
    component.api_endpoint = "http://test-endpoint"
    component.embedding = Mock()
    return component

@pytest.fixture
def sample_csv_content():
    """Create sample CSV content for testing."""
    return """id,title,text,url,vector_id
1,Test Title 1,Test Content 1,http://test1.com,vec1
2,Test Title 2,Test Content 2,http://test2.com,vec2
3,,Just Content 3,http://test3.com,vec3"""

@pytest.fixture
def sample_csv_file(sample_csv_content):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(sample_csv_content)
        return f.name

class TestGridGainVectorStoreComponent:
    def test_initialization(self, component):
        """Test basic component initialization."""
        assert component.cache_name == "test_cache"
        assert component.api_endpoint == "http://test-endpoint"
        assert component.embedding is not None

    def test_process_csv_valid_file(self, component: GridGainVectorStoreComponent, sample_csv_file):
        """Test processing a valid CSV file."""
        documents = component.process_csv(sample_csv_file)
        
        assert isinstance(documents, list)
        assert len(documents) == 3
        assert all(isinstance(doc, Document) for doc in documents)
        
        # Check first document content and metadata
        assert "Test Title 1\nTest Content 1" in documents[0].page_content
        assert documents[0].metadata == {
            "id": "1",
            "url": "http://test1.com",
            "vector_id": "vec1"
        }

    def test_process_csv_missing_columns(self, component):
        """Test processing CSV with missing columns."""
        minimal_csv = """text
Some content
Another content"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(minimal_csv)
            csv_path = f.name

        documents = component.process_csv(csv_path)
        assert len(documents) == 2
        assert documents[0].metadata == {"id": "", "url": "", "vector_id": ""}
        
        os.unlink(csv_path)

    def test_process_csv_invalid_file(self, component: GridGainVectorStoreComponent):
        """Test processing an invalid CSV file."""
        with pytest.raises(Exception):
            component.process_csv("nonexistent_file.csv")

    @patch('langchain_community.vectorstores.ignite.GridGainVectorStore')
    def test_build_vector_store_without_csv(self, mock_gridgain, component: GridGainVectorStoreComponent):
        """Test building vector store without CSV input."""
        mock_instance = Mock()
        mock_gridgain.return_value = mock_instance
        
        vector_store = component.build_vector_store()
        
        mock_gridgain.assert_called_once_with(
            cache_name="test_cache",
            embedding=component.embedding,
            api_endpoint="http://test-endpoint"
        )
        assert vector_store == mock_instance

    @patch('langchain_community.vectorstores.ignite.GridGainVectorStore')
    def test_build_vector_store_with_csv(self, mock_gridgain, component: GridGainVectorStoreComponent, sample_csv_file):
        """Test building vector store with CSV input."""
        mock_instance = Mock()
        mock_gridgain.return_value = mock_instance
        
        component.csv_file = sample_csv_file
        vector_store = component.build_vector_store()
        
        mock_gridgain.assert_called_once()
        mock_instance.add_documents.assert_called_once()
        assert "Added 3 documents" in component.status

    def test_search_documents_empty_query(self, component):
        """Test search with empty query."""
        component.search_query = ""
        results = component.search_documents()
        
        assert results == []
        assert "No search query provided" in component.status

    @patch('langchain_community.vectorstores.ignite.GridGainVectorStore')
    def test_search_documents_valid_query(self, mock_gridgain, component):
        """Test search with valid query."""
        mock_instance = Mock()
        mock_docs = [
            Document(page_content="Test content 1", metadata={"source": "test1"}),
            Document(page_content="Test content 2", metadata={"source": "test2"})
        ]
        mock_instance.similarity_search.return_value = mock_docs
        mock_gridgain.return_value = mock_instance
        
        component.search_query = "test query"
        component.number_of_results = 2
        results = component.search_documents()
        
        assert len(results) == 2
        mock_instance.similarity_search.assert_called_once_with(
            query="test query",
            k=2
        )
        assert "Found 2 results" in component.status

    @patch('langchain_community.vectorstores.ignite.GridGainVectorStore')
    def test_search_documents_error_handling(self, mock_gridgain, component):
        """Test search error handling."""
        mock_gridgain.side_effect = Exception("Test error")
        
        component.search_query = "test query"
        results = component.search_documents()
        
        assert results == []
        assert "Search error: Test error" in component.status

    def teardown_method(self, method):
        """Cleanup after tests."""
        # Clean up any temporary files created during testing
        if hasattr(self, 'temp_csv_path'):
            try:
                os.unlink(self.temp_csv_path)
            except:
                pass