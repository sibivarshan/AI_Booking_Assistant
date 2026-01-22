"""
LangChain Agent for the AI Booking Assistant
Uses Google Generative AI with native function calling.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

from .tools import TOOL_MAP, search_knowledge_base, create_booking, send_confirmation_email, get_booking_info, web_search_movies

# Configure Gemini
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


# Define tools for Gemini function calling
GEMINI_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "search_knowledge_base",
                "description": "Search the knowledge base for information related to the user's query. Use this to answer questions about services, policies, or uploaded documents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's question or search query"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "create_booking",
                "description": "Create a new booking in the database. ONLY use this after the user has confirmed all their booking details.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_name": {
                            "type": "string",
                            "description": "Full name of the customer"
                        },
                        "customer_email": {
                            "type": "string",
                            "description": "Email address of the customer"
                        },
                        "customer_phone": {
                            "type": "string",
                            "description": "Phone number of the customer"
                        },
                        "booking_type": {
                            "type": "string",
                            "description": "Type of booking (e.g., Movie, Appointment, etc.)"
                        },
                        "booking_date": {
                            "type": "string",
                            "description": "Date for the booking in YYYY-MM-DD format"
                        },
                        "booking_time": {
                            "type": "string",
                            "description": "Time for the booking in HH:MM format (24-hour)"
                        }
                    },
                    "required": ["customer_name", "customer_email", "customer_phone", "booking_type", "booking_date", "booking_time"]
                }
            },
            {
                "name": "send_confirmation_email",
                "description": "Send a confirmation email for an existing booking.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "booking_id": {
                            "type": "integer",
                            "description": "The ID of the booking to send confirmation for"
                        }
                    },
                    "required": ["booking_id"]
                }
            },
            {
                "name": "get_booking_info",
                "description": "Retrieve information about an existing booking.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "booking_id": {
                            "type": "integer",
                            "description": "The ID of the booking to retrieve"
                        }
                    },
                    "required": ["booking_id"]
                }
            },
            {
                "name": "web_search_movies",
                "description": "Search the web for currently running Indian movies in theaters. Use this when users ask about what movies are playing, new releases, or current Bollywood/Indian films.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query about movies (e.g., 'movies running in India now', 'latest Bollywood movies in theaters')"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    }
]


# System prompt for the booking assistant
SYSTEM_PROMPT = """You are an AI Booking Assistant - a friendly and professional virtual assistant that helps users with bookings and answers their questions.

## Your Capabilities:
1. **Answer Questions**: Use the search_knowledge_base tool to answer questions about services, policies, or any uploaded documents.
2. **Search Movies**: Use the web_search_movies tool to find currently running movies in theaters when users ask about movies.
3. **Create Bookings**: Help users make bookings by collecting their information step by step, then use the create_booking tool.
4. **Send Confirmations**: Use send_confirmation_email tool after successful bookings.
5. **Retrieve Booking Info**: Use get_booking_info tool to check existing booking details.

## Booking Flow:
When a user wants to make a booking, follow this process:

1. **Collect Information One Step at a Time:**
   - First ask for their **full name**
   - Then ask for their **email address**  
   - Then ask for their **phone number**
   - Then ask for the **type of booking** they want
   - Then ask for their **preferred date** (YYYY-MM-DD format)
   - Then ask for their **preferred time** (HH:MM format, 24-hour)

2. **Confirm Before Creating:**
   - Show a summary of ALL collected information
   - Ask for explicit confirmation (e.g., "Is this correct? Should I proceed?")
   - ONLY call create_booking tool AFTER the user confirms with "yes" or similar

3. **After Booking:**
   - Offer to send a confirmation email
   - If they agree, use send_confirmation_email with the booking ID

## Important Rules:
- Be conversational and friendly, but professional
- If the user asks a question, use search_knowledge_base tool first
- Never create a booking without explicit user confirmation
- When user confirms with "yes", you MUST call the create_booking tool
- Keep responses concise but helpful
- Use emojis sparingly for friendliness (📅, ✅, 📧)
"""


class BookingAssistant:
    """
    Booking Assistant class that manages conversation history and LLM interactions.
    Uses Gemini native function calling for reliable tool execution.
    """
    
    def __init__(self, max_history: int = 20):
        """
        Initialize the booking assistant.
        
        Args:
            max_history: Maximum number of messages to keep in conversation history (default 20)
        """
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT,
            tools=GEMINI_TOOLS
        )
        self.chat_history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.chat_session = None
    
    def _trim_history(self):
        """Keep only the last max_history messages."""
        if len(self.chat_history) > self.max_history * 2:
            self.chat_history = self.chat_history[-(self.max_history * 2):]
    
    def _execute_tool(self, function_call) -> str:
        """Execute a tool call and return the result."""
        tool_name = function_call.name
        args = dict(function_call.args)
        
        print(f"[DEBUG] Executing tool: {tool_name} with args: {args}")
        
        if tool_name not in TOOL_MAP:
            return f"Error: Unknown tool '{tool_name}'"
        
        try:
            tool_func = TOOL_MAP[tool_name]
            result = tool_func(**args)
            print(f"[DEBUG] Tool result: {result}")
            return result
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            print(f"[DEBUG] {error_msg}")
            return error_msg
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the assistant's response.
        
        Args:
            user_message: The user's input message
            
        Returns:
            The assistant's response
        """
        try:
            # Start or continue chat session
            if self.chat_session is None:
                self.chat_session = self.model.start_chat(history=[])
            
            # Send message and get response
            response = self.chat_session.send_message(user_message)
            
            # Check for function calls
            while response.candidates[0].content.parts:
                # Look for function call in response parts
                function_call = None
                text_parts = []
                
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call.name:
                        function_call = part.function_call
                        break
                    elif hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                
                if function_call:
                    # Execute the tool
                    tool_result = self._execute_tool(function_call)
                    
                    # Send tool result back to model
                    response = self.chat_session.send_message(
                        genai.protos.Content(
                            parts=[
                                genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name=function_call.name,
                                        response={"result": tool_result}
                                    )
                                )
                            ]
                        )
                    )
                else:
                    # No more function calls, break
                    break
            
            # Extract final text response
            assistant_message = self._extract_response_text(response)
            
            # Update chat history for display purposes
            self.chat_history.append({"role": "user", "content": user_message})
            self.chat_history.append({"role": "assistant", "content": assistant_message})
            
            # Trim history
            self._trim_history()
            
            return assistant_message
            
        except Exception as e:
            error_str = str(e)
            print(f"[DEBUG] Error in chat: {error_str}")
            if "finish_reason" in error_str:
                return "I apologize, but I couldn't process that request. Please try rephrasing your message or provide the information in a different way."
            error_msg = f"I encountered an error: {error_str}. Please try again or rephrase your request."
            return error_msg
    
    def _extract_response_text(self, response) -> str:
        """Safely extract text from Gemini response."""
        try:
            # Try to get text directly
            if hasattr(response, 'text') and response.text:
                return response.text
            
            # Check candidates
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        text_parts = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)
                        if text_parts:
                            return "\n".join(text_parts)
                
                # Check for blocked response
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    if finish_reason == 3:  # SAFETY
                        return "I apologize, but I can't respond to that request due to content restrictions."
                    elif finish_reason == 12:  # BLOCKLIST or other
                        return "I apologize, but I couldn't generate a response. Please try rephrasing your request."
            
            return "I apologize, but I couldn't generate a response. Please try again."
        except Exception as e:
            return f"Error processing response: {str(e)}"
    
    def clear_history(self):
        """Clear the conversation history."""
        self.chat_history = []
        self.chat_session = None
    
    def get_history_length(self) -> int:
        """Get the current number of messages in history."""
        return len(self.chat_history)


# Singleton instance for the Streamlit app
_assistant_instance: Optional[BookingAssistant] = None


def get_assistant() -> BookingAssistant:
    """Get or create the singleton assistant instance."""
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = BookingAssistant()
    return _assistant_instance
