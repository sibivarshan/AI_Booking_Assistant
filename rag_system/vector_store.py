import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import uuid
from pathlib import Path
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

from .config import CHROMA_DB_PATH, COLLECTION_NAME

# Configure Gemini
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


class VectorStore:
    def __init__(self):
        # Use Gemini for embeddings
        self.embedding_model_name = "models/embedding-001"
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection without embedding function
        # We will always pass embeddings explicitly
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "booking RAG Collection"}
        )
        print(f"Initialized collection: {COLLECTION_NAME}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding using Google's embedding API."""
        try:
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 768
    
    def _get_query_embedding(self, text: str) -> List[float]:
        """Get embedding for query using Google's embedding API."""
        try:
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error getting query embedding: {e}")
            return [0.0] * 768
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store"""
        if not documents:
            return
        
        ids = []
        embeddings = []
        metadatas = []
        documents_text = []
        
        for doc in documents:
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            
            # Create embedding using Gemini
            content = doc['content']
            embedding = self._get_embedding(content)
            embeddings.append(embedding)
            
            # Prepare metadata - filter out non-string values for ChromaDB
            metadata = {}
            for key, value in doc['metadata'].items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
            metadata['title'] = doc['title']
            metadatas.append(metadata)
            
            # Store document text
            documents_text.append(content)
        
        # Add to collection with explicit embeddings
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_text
        )
        
        print(f"Added {len(documents)} documents to vector store")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        # Generate query embedding using Gemini
        query_embedding = self._get_query_embedding(query)
        
        # Search in collection with explicit embedding
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i]
                result = {
                    'content': results['documents'][0][i],
                    'metadata': metadata,
                    'similarity_score': 1 - results['distances'][0][i],
                    'title': metadata.get('title', 'Unknown'),
                    'source_file': metadata.get('source_file', 'Unknown'),
                    'page_number': metadata.get('page_number'),
                    'chunk_id': metadata.get('chunk_id'),
                    'original_document_id': metadata.get('original_document_id'),
                    'has_full_context': 'full_content' in metadata
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def get_full_context(self, original_document_id: str) -> Dict[str, Any]:
        """Get full context for a document"""
        try:
            results = self.collection.get(
                where={"original_document_id": original_document_id},
                limit=1,
                include=['metadatas']
            )
            
            if results['metadatas'] and len(results['metadatas']) > 0:
                metadata = results['metadatas'][0]
                return {
                    'full_content': metadata.get('full_content', ''),
                    'title': metadata.get('title', 'Unknown'),
                    'source_file': metadata.get('source_file', 'Unknown'),
                    'page_number': metadata.get('page_number'),
                    'total_pages': metadata.get('total_pages'),
                    'file_type': metadata.get('file_type')
                }
        except Exception as e:
            print(f"Error getting full context: {e}")
        
        return {'error': 'Document not found'}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": COLLECTION_NAME,
            "embedding_model": self.embedding_model_name
        }
    
    def reset_collection(self) -> None:
        """Reset the collection (delete all data)"""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "booking RAG Collection"}
        )
        print("Collection reset successfully")