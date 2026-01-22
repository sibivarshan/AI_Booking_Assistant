# AI Booking Assistant

An AI-powered booking assistant with RAG (Retrieval Augmented Generation) capabilities, built with Streamlit and LangChain.

## Features

- **AI Chat Interface** - Conversational booking assistant powered by Google Gemini
- **RAG Support** - Upload PDFs to enhance the assistant's knowledge
-  **Smart Booking** - Multi-turn dialogue for collecting booking details
-  **Email Confirmations** - Automatic confirmation emails via SendGrid
-  **Admin Dashboard** - View, filter, and manage all bookings
- **Persistent Storage** - SQLite database for customers and bookings
## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Copy `.env.example` to `.env` and add your API keys:

```env
GOOGLE_API_KEY=your_google_api_key_here
SENDGRID_API_KEY=your_sendgrid_api_key_here
FROM_EMAIL=your_email@example.com
```
kk
### 3. Run the Application

```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

## Project Structure

```
AI_Booking_Assistant/
├── streamlit_app.py        # Main Streamlit chat interface
├── pages/
│   └── admin.py           # Admin dashboard
├── agent/
│   ├── agent.py           # LangChain agent with Gemini LLM
│   └── tools.py           # RAG, Booking, Email tools
├── rag_system/
│   ├── config.py          # Configuration settings
│   ├── data_ingestion.py  # PDF/JSON ingestion
│   ├── data_processor.py  # Text extraction & chunking
│   └── vector_store.py    # ChromaDB vector store
├── main.py                # FastAPI backend (optional)
├── booking.db             # SQLite database
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables (create from .env.example)
```

## Usage

### Chat Interface
1. Open the app at `http://localhost:8501`
2. Upload PDFs using the sidebar to enhance the assistant's knowledge
3. Ask questions or say "I want to make a booking"
4. Follow the assistant's prompts to complete your booking

### Admin Dashboard
1. Click "Admin Dashboard" in the sidebar
2. View all bookings with filtering options
3. Update booking status as needed
4. Export data to CSV or JSON

## API Endpoints (Optional Backend)

If you need a REST API, run the FastAPI backend:

```bash
python main.py
```

Available at `http://localhost:8000`:
- `POST /search` - Search documents
- `POST /booking` - Create booking
- `POST /booking/send-email` - Send confirmation email
- `GET /stats` - Vector store statistics
- `GET /health` - Health check

## Technologies Used

- **Frontend**: Streamlit
- **LLM**: Google Gemini (via LangChain)
- **Vector Store**: ChromaDB
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Database**: SQLite
- **Email**: SendGrid
- **Backend** (optional): FastAPI

## License

MIT License
