import os
from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
import chromadb
from dotenv import load_dotenv
from . import data_loader
from llama_index.embeddings.google import GooglePaLMEmbedding

load_dotenv()

class Index:
    def __init__(self, directory: str = os.environ.get("DATA_STORAGE_DIRECTORY"),
                 storage_directory: str = os.environ.get("INDEX_STORAGE_DIRECTORY")) -> None:
        self.directory = directory
        self.storage_directory = storage_directory
        self.index = None
        print(f"Index class constructor: Data storage directory: {self.directory}")
        print(f"Index class constructor: Index storage directory: {self.storage_directory}")

        # Initialize Chroma client and collection
        self.chroma_client = chromadb.EphemeralClient()
   
            # Check if the collection exists
        collection_name = "quickstart"
        
        # Check if the collection already exists
        self.chroma_collection=self.chroma_client.get_or_create_collection(collection_name)

    def create_index(self, documents: List[str]) -> VectorStoreIndex:
        """Creates an index from the provided documents."""
        try:
            print("Creating index...")
            self.embedding_name =  "models/embedding-gecko-001"
            self.api_key="AIzaSyAchKo1r7xttfjHPCpaVpGzd7RyfqbvRdU"
            self.embedding=GooglePaLMEmbedding(model_name=self.embedding_name, api_key=self.api_key)
           
            Settings.embed_model = self.embedding
            vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, embed_model=Settings.embed_model 
            )
            print("Index created.")
            return index
        except Exception as e:
            print(f"Error creating index: {e}")
            raise

    def persist_index(self, index: VectorStoreIndex) -> None:
        """Persists the provided index to local storage."""
        try:
            print("Persisting Index ... ")
            if not os.path.exists(self.storage_directory):
                os.makedirs(self.storage_directory)

            # Persist index to local storage
            index.storage_context.persist(persist_dir=self.storage_directory)
            print("Index persisted.")
        except Exception as e:
            print(f"Error creating index: {e}")
            raise

    def load_index(self, urls: list = None) -> VectorStoreIndex:
        """Loads the index from Chroma or creates a new one if necessary."""
        try:
            print("Loading index ...")
            documents = data_loader.DataLoader(self.directory).load_documents(urls)
            self.index = self.create_index(documents)
            self.persist_index(self.index)
            print("Loaded Index.")
            return self.index
        except Exception as e:
            print(f"Error loading index: {e}")
            raise
