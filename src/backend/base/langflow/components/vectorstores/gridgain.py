from loguru import logger
import uuid

from langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from langflow.helpers.data import docs_to_data
from langflow.io import HandleInput, IntInput, MessageTextInput, StrInput, FloatInput
from langflow.schema import Data
from langchain.schema import Document

from pygridgain import Client
from langchain_gridgain.vectorstores import GridGainVectorStore


class GridGainVectorStoreComponent(LCVectorStoreComponent):
    """GridGain Vector Store with data ingestion capabilities."""

    display_name: str = "GridGain"
    description: str = "GridGain Vector Store with data ingestion and search capabilities"
    documentation = "https://www.gridgain.com/docs/latest/index"
    name = "GridGain"
    icon = "GridGain"

    inputs = [
        StrInput(name="cache_name", display_name="Cache Name", required=True),
        StrInput(name="host", display_name="Host", required=True),
        IntInput(name="port", display_name="Port", required=True),
        FloatInput(name="score_threshold", display_name="Score Threshold", required=True, value=0.6),
        HandleInput(
            name="embedding",
            display_name="Embedding",
            input_types=["Embeddings"],
            required=True,
        ),
        MessageTextInput(
            name="search_query",
            display_name="Search Query",
        ),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="Number of results to return.",
            value=4,
        ),
        *LCVectorStoreComponent.inputs,
    ]

    def _process_data_input(self, data_input: Data) -> Document:
        """Process a single Data input into a Document with proper metadata."""
        try:
            # Convert Data to LangChain Document
            doc = data_input.to_lc_document()
            
            # Ensure document has metadata
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            
            # Ensure required metadata fields with proper formatting
            doc_id = str(doc.metadata.get('id', uuid.uuid4()))
            doc.metadata.update({
                'id': doc_id,
                'vector_id': str(doc.metadata.get('vector_id', doc_id)),
                'url': str(doc.metadata.get('url', '')),
                'title': str(doc.metadata.get('title', ''))
            })
            
            return doc
        except Exception as e:
            logger.error(f"Error processing data input: {e}")
            raise

    def _add_documents_to_vector_store(self, vector_store: GridGainVectorStore) -> None:
        """Add documents from ingest_data to the vector store using add_texts."""
        try:
            documents = []
            texts = []
            metadatas = []
            
            for _input in self.ingest_data or []:
                if isinstance(_input, Data):
                    doc = self._process_data_input(_input)
                    documents.append(doc)
                    texts.append(doc.page_content)
                    metadatas.append(doc.metadata)
                else:
                    msg = "Vector Store Inputs must be Data objects."
                    raise TypeError(msg)

            if documents:
                logger.info(f"Adding {len(documents)} documents to the Vector Store")
                vector_store.add_texts(texts=texts, metadatas=metadatas)
                self.status = f"Successfully added {len(documents)} documents to GridGain"
            else:
                logger.info("No documents to add to the Vector Store")

        except Exception as e:
            msg = f"Error adding documents to GridGainVectorStore: {e}"
            logger.error(msg)
            raise ValueError(msg) from e

    @check_cached_vector_store
    def build_vector_store(self) -> GridGainVectorStore:
        """Build and return a configured GridGain vector store."""
        try:
            # Connect to GridGain
            client = Client()
            client.connect(self.host, self.port)
            logger.info(f"Connected to GridGain at {self.host}:{self.port}")

            # Initialize vector store
            vector_store = GridGainVectorStore(
                cache_name=self.cache_name,
                embedding=self.embedding,
                client=client
            )

            # Add documents from ingest_data
            self._add_documents_to_vector_store(vector_store)

            return vector_store

        except Exception as e:
            logger.error(f"Error building vector store: {e}")
            raise

    def search_documents(self, vector_store=None) -> list[Data]:
        """Search documents with similarity search."""
        try:
            vector_store = vector_store or self.build_vector_store()

            if not self.search_query or not isinstance(self.search_query, str) or not self.search_query.strip():
                self.status = "No search query provided"
                return []

            docs = vector_store.similarity_search(
                query=self.search_query,
                k=self.number_of_results,
                score_threshold=self.score_threshold
            )

            data = docs_to_data(docs)
            self.status = f"Found {len(data)} results for the query: {self.search_query}"
            return data
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            self.status = f"Search error: {str(e)}"
            return []