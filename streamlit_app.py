"""
AI Booking Assistant - Streamlit Frontend
Main chat interface with PDF upload and conversation management.
"""

import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agent.agent import BookingAssistant
from rag_system.data_ingestion import DataIngestion

# Page configuration
st.set_page_config(
    page_title="AI Booking Assistant",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better chat UI
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    
    .chat-message.assistant {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ed 100%);
        color: #1a1a2e;
        margin-right: 2rem;
    }
    
    .chat-message .avatar {
        font-size: 1.5rem;
        min-width: 2rem;
    }
    
    .chat-message .content {
        flex: 1;
        line-height: 1.5;
    }
    
    /* Sidebar styling */
    .sidebar .stButton button {
        width: 100%;
        margin-bottom: 0.5rem;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Status message styling */
    .status-box {
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .status-success {
        background-color: #d1fae5;
        color: #065f46;
        border-left: 4px solid #10b981;
    }
    
    .status-error {
        background-color: #fee2e2;
        color: #991b1b;
        border-left: 4px solid #ef4444;
    }
    
    .status-info {
        background-color: #dbeafe;
        color: #1e40af;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "assistant" not in st.session_state:
        st.session_state.assistant = BookingAssistant(max_history=20)
    
    if "data_ingestion" not in st.session_state:
        st.session_state.data_ingestion = DataIngestion()


def display_chat_message(role: str, content: str):
    """Display a chat message with styling."""
    if role == "user":
        avatar = "👤"
        css_class = "user"
    else:
        avatar = "🤖"
        css_class = "assistant"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div class="avatar">{avatar}</div>
        <div class="content">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with upload and controls."""
    with st.sidebar:
        st.markdown("## 📁 Document Upload")
        st.markdown("Upload PDFs to enhance the assistant's knowledge.")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF documents for RAG"
        )
        
        if uploaded_files:
            if st.button("📤 Process Uploaded Files", type="primary"):
                with st.spinner("Processing files..."):
                    import requests
                    
                    # API endpoint - change to your deployed URL
                    API_URL = "https://ai-booking-assistant-svqo.onrender.com/upload"
                    
                    success_count = 0
                    for file in uploaded_files:
                        try:
                            # Reset file pointer to beginning
                            file.seek(0)
                            
                            # Send file to API endpoint
                            files = {"file": (file.name, file.read(), "application/pdf")}
                            response = requests.post(API_URL, files=files, timeout=120)
                            
                            if response.status_code == 200:
                                result = response.json()
                                success_count += 1
                                st.success(f"✅ {file.name}: Added {result['documents_added']} chunks")
                            else:
                                error_detail = response.json().get("detail", response.text)
                                st.error(f"❌ {file.name}: {error_detail}")
                        
                        except requests.exceptions.Timeout:
                            st.error(f"❌ {file.name}: Request timed out. Try again.")
                        except requests.exceptions.RequestException as e:
                            st.error(f"❌ {file.name}: Connection error - {str(e)}")
                        except Exception as e:
                            st.error(f"❌ {file.name}: {str(e)}")
                    
                    if success_count > 0:
                        st.success(f"Successfully processed {success_count} file(s)!")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("## ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.session_state.assistant.clear_history()
                st.rerun()
        
        with col2:
            if st.button("📊 View Stats"):
                try:
                    stats = st.session_state.data_ingestion.get_stats()
                    st.info(f"📚 Documents: {stats['total_documents']}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        
        # Navigation
        st.markdown("## 🧭 Navigation")
        st.page_link("streamlit_app.py", label="💬 Chat", icon="💬")
        st.page_link("pages/admin.py", label="📊 Admin Dashboard", icon="📊")
        
        st.markdown("---")
        
        # Info
        st.markdown("## ℹ️ About")
        st.markdown("""
        **AI Booking Assistant** helps you:
        - 📚 Answer questions from uploaded PDFs
        - 📅 Create and manage bookings
        - 📧 Send confirmation emails
        
        Just type your message below!
        """)
        
        # Show conversation length
        history_len = st.session_state.assistant.get_history_length()
        st.caption(f"💭 Conversation memory: {history_len // 2} exchanges")


def render_chat():
    """Render the main chat interface."""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📅 AI Booking Assistant</h1>
        <p style="color: #6b7280;">Your intelligent assistant for bookings and information</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display existing messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])
        
        # Welcome message if no messages
        if not st.session_state.messages:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("""
                👋 **Hello! I'm your AI Booking Assistant.**
                
                I can help you with:
                - 📚 **Answering questions** about uploaded documents
                - 📅 **Making bookings** - just say "I want to make a booking"
                - 📧 **Sending confirmations** via email
                - 🔍 **Checking booking details**
                
                How can I assist you today?
                """)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.assistant.chat(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"I encountered an error: {str(e)}. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
