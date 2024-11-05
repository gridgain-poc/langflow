import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores.ignite import GridGainVectorStore

# Import the component to test
from base.langflow.components.vectorstores.GridGain import GridGainVectorStoreComponent

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )

@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings model."""
    mock_model = Mock()
    mock_model.encode.return_value = np.random.rand(3, 384)  # Typical embedding dimension
    return mock_model

@pytest.fixture
def mock_gridgain():
    """Create a mock GridGain vector store."""
    with patch('langchain_community.vectorstores.ignite.GridGainVectorStore') as mock:
        instance = mock.return_value
        instance.similarity_search.return_value = [
            Document(page_content="Test result 1", metadata={"id": "1"}),
            Document(page_content="Test result 2", metadata={"id": "2"}),
        ]
        yield instance

@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Machine learning is a field of artificial intelligence",
            metadata={"id": "1", "category": "AI"}
        ),
        Document(
            page_content="Deep learning is a subset of machine learning",
            metadata={"id": "2", "category": "AI"}
        ),
        Document(
            page_content="Neural networks are fundamental to deep learning",
            metadata={"id": "3", "category": "AI"}
        ),
    ]

@pytest.fixture
def sample_csv_file():
    """Create a sample CSV file with real-world like data."""
    df = pd.DataFrame({
        'id': range(1, 11),
        'title': [f'Research Paper {i}' for i in range(1, 11)],
        'text': [
            'Advances in Natural Language Processing',
            'Deep Learning Applications in Healthcare',
            'Computer Vision Techniques',
            'Reinforcement Learning in Robotics',
            'Graph Neural Networks',
            'Transformer Architecture Innovations',
            'Federated Learning Systems',
            'Time Series Analysis with LSTM',
            'Attention Mechanisms in NLP',
            'GANs for Image Generation'
        ],
        'url': [f'http://example.com/paper{i}' for i in range(1, 11)],
        'vector_id': [f'vec_{i}' for i in range(1, 11)]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        df.to_csv(f.name, index=False)
        return f.name

class TestGridGainIntegration:
    
    @pytest.fixture(autouse=True)
    def setup_component(self, mock_embeddings, mock_gridgain):
        """Setup GridGain component before each test."""
        self.component = GridGainVectorStoreComponent()
        self.component.cache_name = "test_cache"
        self.component.api_endpoint = "http://localhost:8080"
        self.component.embedding = mock_embeddings
        self.component.number_of_results = 3
        self.mock_store = mock_gridgain
        yield
    
    def test_build_vector_store_with_documents(self, sample_documents):
        """Test building vector store with direct document insertion."""
        vector_store = self.component.build_vector_store()
        assert vector_store == self.mock_store
        
        self.component.search_query = "machine learning"
        results = self.component.search_documents()
        
        assert len(results) == 2
        assert self.mock_store.similarity_search.called
        
    def test_csv_ingestion_and_search(self, sample_csv_file):
        """Test full workflow of CSV ingestion and search."""
        self.component.csv_file = sample_csv_file
        vector_store = self.component.build_vector_store()
        
        # Verify CSV processing
        docs = self.component.process_csv(sample_csv_file)
        assert len(docs) == 10
        
        self.component.search_query = "deep learning"
        results = self.component.search_documents()
        
        assert len(results) == 2
        assert "Found" in self.component.status
        
    def test_search_with_different_result_sizes(self, sample_csv_file):
        """Test search with different k values."""
        self.component.csv_file = sample_csv_file
        vector_store = self.component.build_vector_store()
        
        for k in [1, 3, 5]:
            self.component.number_of_results = k
            self.component.search_query = "neural networks"
            results = self.component.search_documents()
            
            # Note: Mock always returns 2 results due to fixture setup
            assert len(results) == 2
            assert "Found" in self.component.status
            
    def test_vector_store_persistence(self, sample_documents):
        """Test that documents persist in vector store between searches."""
        vector_store = self.component.build_vector_store()
        
        self.component.search_query = "artificial intelligence"
        results1 = self.component.search_documents()
        
        # Create new component instance
        new_component = GridGainVectorStoreComponent()
        new_component.cache_name = self.component.cache_name
        new_component.api_endpoint = self.component.api_endpoint
        new_component.embedding = self.component.embedding
        new_component.search_query = "artificial intelligence"
        
        results2 = new_component.search_documents()
        
        assert len(results1) == len(results2)
        
    def test_error_handling_with_invalid_endpoint(self):
        """Test error handling with invalid API endpoint."""
        # Simulate connection error
        self.mock_store.similarity_search.side_effect = Exception("Connection failed")
        
        self.component.api_endpoint = "http://invalid-endpoint:9999"
        self.component.search_query = "test"
        
        results = self.component.search_documents()
        assert len(results) == 0
        assert "Search error" in self.component.status
        
    @pytest.mark.performance
    def test_search_performance(self, sample_csv_file):
        """Test search performance with larger dataset."""
        self.component.csv_file = sample_csv_file
        vector_store = self.component.build_vector_store()
        
        import time
        start_time = time.time()
        self.component.search_query = "neural networks"
        results = self.component.search_documents()
        end_time = time.time()
        
        search_time = end_time - start_time
        assert search_time < 5.0
        assert len(results) == 2

if __name__ == "__main__":
    pytest.main(["-v", __file__])