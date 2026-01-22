"""
Tools for the AI Booking Assistant
Simple Python functions without LangChain dependency.
"""

import os
import sqlite3
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Database path
DB_PATH = Path(__file__).parent.parent / "booking.db"

# Lazy loaded components
_vector_store = None
_data_ingestion = None


def get_vector_store():
    """Lazy load vector store to avoid import issues."""
    global _vector_store
    if _vector_store is None:
        from rag_system.vector_store import VectorStore
        _vector_store = VectorStore()
    return _vector_store


def get_data_ingestion():
    """Lazy load data ingestion to avoid import issues."""
    global _data_ingestion
    if _data_ingestion is None:
        from rag_system.data_ingestion import DataIngestion
        _data_ingestion = DataIngestion()
    return _data_ingestion


def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for information related to the user's query.
    
    Args:
        query: The user's question or search query
        
    Returns:
        Relevant information from the knowledge base
    """
    try:
        vector_store = get_vector_store()
        results = vector_store.search(query, top_k=3)
        
        if not results:
            return "No relevant information found in the knowledge base. The user may need to upload documents first."
        
        # Format results
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.get('source_file', 'Unknown')
            page = result.get('page_number', '')
            content = result.get('content', '')
            page_info = f" (Page {page})" if page else ""
            context_parts.append(f"[Source: {source}{page_info}]\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


def create_booking(
    customer_name: str,
    customer_email: str,
    customer_phone: str,
    booking_type: str,
    booking_date: str,
    booking_time: str
) -> str:
    """
    Create a new booking in the database.
    
    Args:
        customer_name: Full name of the customer
        customer_email: Email address of the customer
        customer_phone: Phone number of the customer
        booking_type: Type of booking
        booking_date: Date for the booking in YYYY-MM-DD format
        booking_time: Time for the booking in HH:MM format
        
    Returns:
        Confirmation message with booking ID or error message
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Check if customer exists
            cursor.execute("SELECT customer_id FROM customers WHERE email = ?", (customer_email,))
            row = cursor.fetchone()
            
            if row:
                customer_id = row[0]
                cursor.execute(
                    "UPDATE customers SET name = ?, phone = ? WHERE customer_id = ?",
                    (customer_name, customer_phone, customer_id)
                )
            else:
                cursor.execute(
                    "INSERT INTO customers (name, email, phone) VALUES (?, ?, ?)",
                    (customer_name, customer_email, customer_phone)
                )
                customer_id = cursor.lastrowid
            
            cursor.execute(
                """
                INSERT INTO bookings (customer_id, booking_type, date, time, status, created_at)
                VALUES (?, ?, ?, ?, 'confirmed', datetime('now'))
                """,
                (customer_id, booking_type, booking_date, booking_time)
            )
            
            booking_id = cursor.lastrowid
            conn.commit()
            
        return f"✅ Booking created successfully!\n\n**Booking ID:** {booking_id}\n**Customer:** {customer_name}\n**Type:** {booking_type}\n**Date:** {booking_date}\n**Time:** {booking_time}"
    
    except Exception as e:
        return f"❌ Error creating booking: {str(e)}"


def send_confirmation_email(booking_id: int) -> str:
    """
    Send a confirmation email for an existing booking.
    
    Args:
        booking_id: The ID of the booking
        
    Returns:
        Success or failure message
    """
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        
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
                (booking_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return f"❌ Booking with ID {booking_id} not found."
            
            bid, booking_type, date, time, status, name, email, phone = row
        
        sendgrid_api_key = os.environ.get('SENDGRID_API_KEY')
        from_email = os.environ.get('FROM_EMAIL', 'bookings@example.com')
        
        if not sendgrid_api_key:
            return "❌ Email service not configured."
        
        subject = f"Booking Confirmation - {booking_type}"
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif;">
                <h2 style="color: #2563eb;">Booking Confirmation</h2>
                <p>Dear {name},</p>
                <p>Your booking has been confirmed:</p>
                <div style="background: #f3f4f6; padding: 20px; border-radius: 8px;">
                    <p><strong>Booking ID:</strong> {bid}</p>
                    <p><strong>Type:</strong> {booking_type}</p>
                    <p><strong>Date:</strong> {date}</p>
                    <p><strong>Time:</strong> {time}</p>
                </div>
                <p>Thank you!</p>
            </body>
        </html>
        """
        
        message = Mail(from_email=from_email, to_emails=email, subject=subject, html_content=html_content)
        sg = SendGridAPIClient(sendgrid_api_key)
        sg.send(message)
        
        return f"✅ Confirmation email sent to {email}!"
    
    except Exception as e:
        return f"❌ Failed to send email: {str(e)}"


def get_booking_info(booking_id: int) -> str:
    """
    Retrieve information about an existing booking.
    
    Args:
        booking_id: The ID of the booking
        
    Returns:
        Booking details or error message
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT b.id, b.booking_type, b.date, b.time, b.status, b.created_at,
                       c.name, c.email, c.phone
                FROM bookings b
                JOIN customers c ON b.customer_id = c.customer_id
                WHERE b.id = ?
                """,
                (booking_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return f"No booking found with ID {booking_id}."
            
            bid, booking_type, date, time, status, created_at, name, email, phone = row
            
        return f"""**Booking Details:**
- **Booking ID:** {bid}
- **Customer:** {name}
- **Email:** {email}
- **Phone:** {phone}
- **Type:** {booking_type}
- **Date:** {date}
- **Time:** {time}
- **Status:** {status}"""
    
    except Exception as e:
        return f"Error retrieving booking: {str(e)}"


def web_search_movies(query: str) -> str:
    """
    Search the web for currently running movies using Tavily.
    
    Args:
        query: Search query about movies (e.g., "movies running in India now", "latest Bollywood movies in theaters")
        
    Returns:
        Search results with movie information
    """
    try:
        from tavily import TavilyClient
        
        tavily_api_key = os.environ.get('TAVILY_API_KEY')
        if not tavily_api_key:
            return "❌ Tavily API key not configured. Please add TAVILY_API_KEY to your .env file."
        
        client = TavilyClient(api_key=tavily_api_key)
        
        # Enhance query for better movie results
        enhanced_query = f"{query} movies currently running in theaters 2026"
        
        response = client.search(
            query=enhanced_query,
            search_depth="advanced",
            max_results=5
        )
        
        if not response.get('results'):
            return "No movie information found. Please try a different search query."
        
        # Format results
        results_text = "🎬 **Currently Running Movies:**\n\n"
        
        for i, result in enumerate(response['results'], 1):
            title = result.get('title', 'Unknown')
            content = result.get('content', '')[:200]  # Truncate for brevity
            url = result.get('url', '')
            
            results_text += f"**{i}. {title}**\n"
            results_text += f"{content}...\n"
            if url:
                results_text += f"🔗 [Source]({url})\n"
            results_text += "\n"
        
        return results_text
    
    except Exception as e:
        return f"❌ Error searching for movies: {str(e)}"


# Tool mapping for the agent
TOOL_MAP = {
    "search_knowledge_base": search_knowledge_base,
    "create_booking": create_booking,
    "send_confirmation_email": send_confirmation_email,
    "get_booking_info": get_booking_info,
    "web_search_movies": web_search_movies
}

