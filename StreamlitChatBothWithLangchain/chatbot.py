import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
# --- Configuration and Constants ---

# Load environment variables (useful for future API keys/secrets)
load_dotenv(override=True)
# Configuration constants
session_id = "JSS001"
DEFAULT_SESSION_ID = "JSS001"
OLLAMA_BASE_URL = "http://localhost:11434" 
OLLAMA_MODEL = "qwen2.5:latest"
DATABASE_CONNECTION_STRING = "sqlite:///chatDatabase.db"

# --- LangChain Component Definitions ---
def get_local_llm():
    """Initializes and returns the ChatOllama model."""
    return ChatOllama(base_url=OLLAMA_BASE_URL,
                      model=OLLAMA_MODEL,
                      temperature=0.5,
                      max_tokens=250)

def get_prompt_template() -> ChatPromptTemplate:
    """Creates and returns the standardized chat prompt template."""
    return ChatPromptTemplate.from_messages(
        [   # History comes first to maintain context
            MessagesPlaceholder(variable_name="history"),
            # System instruction with dynamic role injection
            ("system", "You are an {role} level to user to answer this query"),
            # The current user query
            ("human", "{prompt}")
        ]
    )

def get_pipeline_chain():
    """Constructs the core LangChain processing pipeline."""
    return get_prompt_template() | get_local_llm() | StrOutputParser()

def get_session_info(session_id):
    """
    Initializes and returns the SQLChatMessageHistory for a given session.
    This function is used as the `get_session_history` callback in LangChain.
    """
    return SQLChatMessageHistory(session_id=session_id, 
                                 connection_string="sqlite:///chatDatabase.db")

def invoke_history(prompt, session_id, role):
    """
    Creates and streams the response from the history-aware pipeline.

    Args:
        prompt: The user's current query.
        session_id: The ID for fetching/storing chat history.
        role: The persona/expertise level for the system prompt.

    Yields:
        Chunks of the LLM's response.
    """
    history = RunnableWithMessageHistory(get_pipeline_chain(),
                                      get_session_history=get_session_info,
                                      input_messages_key="prompt",
                                      history_messages_key="history")
    # The config dictionary is crucial for RunnableWithMessageHistory
    config = {"configurable": {"session_id": session_id}}
    # The input dictionary contains the keys used in the prompt template
    inputs = {"prompt": prompt, "role": role}
    # Stream the response
    for chuck in history.stream(inputs,config=config):
        yield chuck
        
def clear_session_history(session_id):
    """Clears the chat history for a given session."""
    st.session_state.chat_history = []
    get_session_info(session_id).clear()

def render_message_history():
    """Renders all messages stored in st.session_state.chat_history."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with st.sidebar:
    user_id = st.text_input("Enter your name", session_id)
    role = st.radio("How detailed should the answer be?",
                    [
                        "Beginner",
                        "Expert",
                        "PhD"
                    ])
    if st.button("Start all new conversation"):
        clear_session_history(user_id)

# Main chat interface
st.title("🤖 LangChain-Streamlit Chatbot")
# Render existing messages
render_message_history()
# Get the user's input
prompt = st.chat_input("Enter your query")

if prompt:
    # 1. Append user message to state and display it
    st.session_state.chat_history.append({"role": "user", 
                                          "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Stream the assistant's response
    with st.chat_message("assistant"):
        stream_response = st.write_stream(invoke_history(prompt, session_id, role))

    # 3. Append the complete assistant response to state
    st.session_state.chat_history.append({"role": "assistant",
                                          "content": stream_response})
    
