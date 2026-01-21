from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from pathlib import Path
import tempfile
import shutil

from .data_ingestion import DataIngestion
from .vector_store import VectorStore
from .config import API_HOST, API_PORT, TOP_K_RESULTS

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = TOP_K_RESULTS

class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time_ms: float

class StatsResponse(BaseModel):
    total_documents: int
    collection_name: str
    embedding_model: str

class UploadResponse(BaseModel):
    success: bool
    message: str
    documents_added: int

# Initialize FastAPI app
app = FastAPI(
    title="booking RAG API",
    description="RAG system for booking-related content",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_ingestion = DataIngestion()
vector_store = VectorStore()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "booking RAG API is running!",
        "version": "1.0.0",
        "endpoints": ["/search", "/stats", "/upload", "/ingest", "/health"]
    }

@app.post("/search", response_model=QueryResponse)
async def search(request: QueryRequest):
    """Search for documents similar to the query"""
    import time
    start_time = time.time()
    
    try:
        results = vector_store.search(request.query, request.top_k)
        processing_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            processing_time_ms=round(processing_time, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get vector store statistics"""
    try:
        stats = data_ingestion.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a new file"""
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ['.pdf', '.json']:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Only PDF and JSON files are supported."
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = Path(tmp_file.name)
        
        try:
            # Process the file
            result = data_ingestion.add_single_file(tmp_path)
            
            if result["success"]:
                return UploadResponse(
                    success=True,
                    message=f"Successfully processed {file.filename}",
                    documents_added=result["documents_added"]
                )
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Failed to process file: {result['error']}"
                )
        
        finally:
            # Clean up temporary file
            tmp_path.unlink(missing_ok=True)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/ingest")
async def ingest_all_data(reset_db: bool = False):
    """Ingest all available data files"""
    try:
        stats = data_ingestion.ingest_all_data(reset_db=reset_db)
        return {
            "message": "Data ingestion completed",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        stats = data_ingestion.get_stats()
        return {
            "status": "healthy",
            "total_documents": stats["total_documents"],
            "timestamp": "2025-08-03T02:00:00Z"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2025-08-03T02:00:00Z"
        }

if __name__ == "__main__":
    print("Starting booking RAG API...")
    print(f"API will be available at: http://{API_HOST}:{API_PORT}")
    print("Endpoints:")
    print("  - GET  /          : API information")
    print("  - POST /search    : Search documents")
    print("  - GET  /stats     : Get statistics")
    print("  - POST /upload    : Upload new files")
    print("  - POST /ingest    : Ingest all data")
    print("  - GET  /health    : Health check")
    
    uvicorn.run(
        "rag_system.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )