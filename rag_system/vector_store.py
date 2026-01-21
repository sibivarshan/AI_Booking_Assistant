import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import uuid
from pathlib import Path
from .config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

class VectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
            print(f"Loaded existing collection: {COLLECTION_NAME}")
        except:
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "booking RAG Collection"}
            )
            print(f"Created new collection: {COLLECTION_NAME}")
    
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
            
            # Create embedding
            content = doc['content']
            embedding = self.embedding_model.encode(content).tolist()
            embeddings.append(embedding)
            
            # Prepare metadata
            metadata = doc['metadata'].copy()
            metadata['title'] = doc['title']
            metadatas.append(metadata)
            
            # Store document text
            documents_text.append(content)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_text
        )
        
        print(f"Added {len(documents)} documents to vector store")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
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
        results = self.collection.query(
            query_embeddings=None,
            where={"original_document_id": original_document_id},
            n_results=1,
            include=['metadatas']
        )
        
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            metadata = results['metadatas'][0][0]
            return {
                'full_content': metadata.get('full_content', ''),
                'title': metadata.get('title', 'Unknown'),
                'source_file': metadata.get('source_file', 'Unknown'),
                'page_number': metadata.get('page_number'),
                'total_pages': metadata.get('total_pages'),
                'file_type': metadata.get('file_type')
            }
        
        return {'error': 'Document not found'}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL
        }
    
    def reset_collection(self) -> None:
        """Reset the collection (delete all data)"""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "booking RAG Collection"}
        )
        print("Collection reset successfully")