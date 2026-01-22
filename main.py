#!/usr/bin/env python3
"""
booking RAG System - FastAPI Server
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import time
from pathlib import Path
import tempfile
import shutil
import sqlite3
import os
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Load environment variables
load_dotenv()

from rag_system.data_ingestion import DataIngestion
from rag_system.vector_store import VectorStore

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

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


# Simple SQLite booking DB
DB_PATH = Path(__file__).parent / "booking.db"


class BookingRequest(BaseModel):
    name: str
    email: str
    phone: str
    booking_type: str
    date: str  # ISO date string, e.g. 2025-01-21
    time: str  # time string, e.g. 14:30
    status: Optional[str] = "pending"


class BookingResponse(BaseModel):
    booking_id: int
    customer_id: int
    message: str


class EmailRequest(BaseModel):
    booking_id: int
    recipient_email: Optional[str] = None  # If not provided, uses customer's email from DB


class EmailResponse(BaseModel):
    success: bool
    message: str
    booking_id: int


def init_booking_db() -> None:
    """Create minimal customers and bookings tables if they do not exist."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                booking_type TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
            """
        )

        conn.commit()

# Initialize FastAPI app
app = FastAPI(
    title="booking RAG API",
    description="RAG system for booking-related content with page/index tracking",
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

# Initialize booking database
init_booking_db()

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint"""
    return {
        "message": "booking RAG API is running!",
        "version": "1.0.0",
        "endpoints": ["/search", "/context/{document_id}", "/stats", "/upload", "/ingest", "/health"]
    }

@app.post("/search", response_model=QueryResponse)
async def search(request: QueryRequest):
    """Search for documents similar to the query"""
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

@app.get("/context/{document_id}", response_model=Dict[str, Any])
async def get_context(document_id: str):
    """Get full context for a document using its original_document_id"""
    try:
        context = vector_store.get_full_context(document_id)
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get context: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get vector store statistics"""
    try:
        stats = data_ingestion.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.post("/booking", response_model=BookingResponse)
async def create_booking(booking: BookingRequest):
    """Create a booking and store customer + booking details in SQLite."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            # Create or update customer
            cursor.execute("SELECT customer_id FROM customers WHERE email = ?", (booking.email,))
            row = cursor.fetchone()

            if row:
                customer_id = row[0]
                cursor.execute(
                    "UPDATE customers SET name = ?, phone = ? WHERE customer_id = ?",
                    (booking.name, booking.phone, customer_id),
                )
            else:
                cursor.execute(
                    "INSERT INTO customers (name, email, phone) VALUES (?, ?, ?)",
                    (booking.name, booking.email, booking.phone),
                )
                customer_id = cursor.lastrowid

            # Insert booking row
            cursor.execute(
                """
                INSERT INTO bookings (customer_id, booking_type, date, time, status, created_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    customer_id,
                    booking.booking_type,
                    booking.date,
                    booking.time,
                    booking.status,
                ),
            )

            booking_id = cursor.lastrowid
            conn.commit()

        return BookingResponse(
            booking_id=booking_id,
            customer_id=customer_id,
            message="Booking stored successfully",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create booking: {str(e)}")


@app.post("/booking/send-email", response_model=EmailResponse)
async def send_booking_email(email_request: EmailRequest):
    """Send booking confirmation email using SendGrid."""
    try:
        # Get booking and customer details from database
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT b.id, b.booking_type, b.date, b.time, b.status,
                       c.name, c.email, c.phone
                FROM bookings b
                JOIN customers c ON b.customer_id = c.customer_id
                WHERE b.id = ?
                """,
                (email_request.booking_id,),
            )
            row = cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Booking not found")

            booking_id, booking_type, date, time, status, name, customer_email, phone = row

        # Use provided email or customer's email from DB
        to_email = email_request.recipient_email or customer_email

        # Get SendGrid API key from environment
        sendgrid_api_key = os.environ.get('SENDGRID_API_KEY')
        from_email = os.environ.get('FROM_EMAIL', 'bookings@example.com')

        if not sendgrid_api_key:
            raise HTTPException(
                status_code=500,
                detail="SENDGRID_API_KEY not configured in .env file"
            )

        # Create email content
        subject = f"Booking Confirmation - {booking_type}"
        html_content = f"""
        <html>
            <body>
                <h2>Booking Confirmation</h2>
                <p>Dear {name},</p>
                <p>Your booking has been confirmed with the following details:</p>
                <ul>
                    <li><strong>Booking ID:</strong> {booking_id}</li>
                    <li><strong>Type:</strong> {booking_type}</li>
                    <li><strong>Date:</strong> {date}</li>
                    <li><strong>Time:</strong> {time}</li>
                    <li><strong>Status:</strong> {status}</li>
                </ul>
                <p>If you have any questions, please contact us.</p>
                <p>Thank you!</p>
            </body>
        </html>
        """

        # Send email via SendGrid
        message = Mail(
            from_email=from_email,
            to_emails=to_email,
            subject=subject,
            html_content=html_content
        )

        sg = SendGridAPIClient(sendgrid_api_key)
        
        # Uncomment if using EU regional subuser
        # if os.environ.get('SENDGRID_EU_RESIDENCY') == 'true':
        #     sg.set_sendgrid_data_residency("eu")
        
        response = sg.send(message)

        return EmailResponse(
            success=True,
            message=f"Email sent successfully to {to_email}",
            booking_id=booking_id
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send email: {str(e)}"
        )


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
    """Ingest all available data files from data_for_rag directory"""
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
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Booking RAG FastAPI Server")
    parser.add_argument("--list-tools", action="store_true", help="List all available agent tools")
    args = parser.parse_args()
    
    if args.list_tools:
        from agent.tools import TOOL_MAP
        print("\n🔧 Available Agent Tools:")
        print("=" * 50)
        for name, func in TOOL_MAP.items():
            doc = func.__doc__ or "No description"
            # Get first line of docstring
            first_line = doc.strip().split('\n')[0]
            print(f"\n📌 {name}")
            print(f"   {first_line}")
        print("\n" + "=" * 50)
        print(f"Total: {len(TOOL_MAP)} tools available")
    else:
        print("🏏 Starting booking RAG FastAPI Server...")
        print("📡 API will be available at: http://localhost:8000")
        print("📖 Interactive docs at: http://localhost:8000/docs")
        print("🔍 Endpoints:")
        print("  - POST /search           : Search documents")
        print("  - GET  /context/{doc_id} : Get full context")
        print("  - GET  /stats            : Get statistics")
        print("  - POST /upload           : Upload new files")
        print("  - POST /ingest           : Ingest all data")
        print("  - GET  /health           : Health check")
        
        uvicorn.run(
            "__main__:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )