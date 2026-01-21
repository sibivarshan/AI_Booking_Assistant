import glob
from pathlib import Path
from typing import List, Dict, Any
from .data_processor import DataProcessor
from .vector_store import VectorStore
from .config import BASE_DIR, DATA_DIR

class DataIngestion:
    def __init__(self):
        self.processor = DataProcessor()
        self.vector_store = VectorStore()
    
    def ingest_all_data(self, reset_db: bool = False) -> Dict[str, Any]:
        """Ingest all available data"""
        if reset_db:
            self.vector_store.reset_collection()
        
        stats = {
            "total_files_processed": 0,
            "total_documents_added": 0,
            "files_by_type": {},
            "errors": []
        }
        
        # Process PDFs
        pdf_stats = self._process_pdfs()
        stats["files_by_type"]["pdf"] = pdf_stats
        stats["total_files_processed"] += pdf_stats["files_processed"]
        stats["total_documents_added"] += pdf_stats["documents_added"]
        stats["errors"].extend(pdf_stats["errors"])
        
        # Process JSON files
        json_stats = self._process_json_files()
        stats["files_by_type"]["json"] = json_stats
        stats["total_files_processed"] += json_stats["files_processed"]
        stats["total_documents_added"] += json_stats["documents_added"]
        stats["errors"].extend(json_stats["errors"])
        
        return stats
    
    def _process_pdfs(self) -> Dict[str, Any]:
        """Process all PDF files from data_for_rag directory"""
        stats = {
            "files_processed": 0,
            "documents_added": 0,
            "errors": []
        }
        
        if not DATA_DIR.exists():
            stats["errors"].append(f"Data directory not found: {DATA_DIR}")
            return stats
        
        pdf_files = list(DATA_DIR.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            try:
                print(f"Processing PDF: {pdf_file.name}")
                documents = self.processor.process_pdf_file(pdf_file)
                
                if documents:
                    self.vector_store.add_documents(documents)
                    stats["documents_added"] += len(documents)
                    print(f"Added {len(documents)} chunks from {pdf_file.name}")
                
                stats["files_processed"] += 1
                
            except Exception as e:
                error_msg = f"Error processing {pdf_file.name}: {str(e)}"
                stats["errors"].append(error_msg)
                print(error_msg)
        
        return stats
    
    def _process_json_files(self) -> Dict[str, Any]:
        """Process all JSON files from data_for_rag directory"""
        stats = {
            "files_processed": 0,
            "documents_added": 0,
            "errors": []
        }
        
        if not DATA_DIR.exists():
            stats["errors"].append(f"Data directory not found: {DATA_DIR}")
            return stats
        
        json_files = list(DATA_DIR.glob("*.json"))
        
        for json_file in json_files:
            try:
                print(f"Processing JSON: {json_file.name}")
                documents = self.processor.process_json_file(json_file)
                
                if documents:
                    # Chunk documents
                    all_chunks = []
                    for doc in documents:
                        chunks = self.processor.chunk_document(doc)
                        all_chunks.extend(chunks)
                    
                    if all_chunks:
                        self.vector_store.add_documents(all_chunks)
                        stats["documents_added"] += len(all_chunks)
                        print(f"Added {len(all_chunks)} chunks from {json_file.name}")
                
                stats["files_processed"] += 1
                
            except Exception as e:
                error_msg = f"Error processing {json_file.name}: {str(e)}"
                stats["errors"].append(error_msg)
                print(error_msg)
        
        return stats
    
    def add_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Add a single file to the vector store"""
        result = {
            "success": False,
            "documents_added": 0,
            "error": None
        }
        
        try:
            if file_path.suffix.lower() == '.pdf':
                documents = self.processor.process_pdf_file(file_path)
            elif file_path.suffix.lower() == '.json':
                raw_documents = self.processor.process_json_file(file_path)
                documents = []
                for doc in raw_documents:
                    chunks = self.processor.chunk_document(doc)
                    documents.extend(chunks)
            else:
                result["error"] = f"Unsupported file type: {file_path.suffix}"
                return result
            
            if documents:
                self.vector_store.add_documents(documents)
                result["documents_added"] = len(documents)
                result["success"] = True
                print(f"Successfully added {len(documents)} documents from {file_path.name}")
            else:
                result["error"] = "No valid documents found in file"
            
        except Exception as e:
            result["error"] = str(e)
            print(f"Error adding file {file_path}: {str(e)}")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return self.vector_store.get_collection_stats()