import os
import shutil
from langchain_openai import OpenAIEmbeddings  # <--- Changed from Google
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document 
import config
from utils.csv_utils import LogAnalyzer

class LogRetriever:
    def __init__(self):
        # 1. Setup OpenAI Embedding Model
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME,
            openai_api_key=config.OPENAI_API_KEY  # <--- Using the key from config.py
        )
        
        # 2. Initialize Vector DB Path
        self.db_path = config.VECTOR_DB_PATH
        
        self.log_analyzer = LogAnalyzer()
        
        # 3. Connection placeholder
        self.vector_store = None

    def ingest_logs(self):
        """
        Reads the latest CSV logs and rebuilds the Vector Database.
        """
        print("Loading logs for ingestion...")
        log_sentences = self.log_analyzer.get_all_logs_formatted()
        
        if not log_sentences:
            print("No logs found to ingest.")
            return

        # Convert strings to LangChain Documents
        docs = [Document(page_content=text) for text in log_sentences]

        print(f"Ingesting {len(docs)} events into Knowledge Base using OpenAI...")
        
        if os.path.exists(self.db_path):
            try:
                shutil.rmtree(self.db_path)
            except Exception as e:
                print(f"Warning: Could not delete old DB: {e}")
            
        # Create new DB with OpenAI Embeddings
        self.vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print("✅ Knowledge Base Updated Successfully with OpenAI.")

    def _ensure_vector_store(self):
        if self.vector_store is None:
            if os.path.exists(self.db_path):
                self.vector_store = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embeddings
                )

    def query_relevant_logs(self, query, k=5):
        self._ensure_vector_store()
        
        if self.vector_store is None:
            print("Vector Store is empty or not initialized.")
            return []

        print(f"Searching Knowledge Base for: '{query}'...")
        results = self.vector_store.similarity_search(query, k=k)
        return results