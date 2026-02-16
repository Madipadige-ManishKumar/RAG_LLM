# from RAG.single_rag import query_chatbot

# response = query_chatbot("DBMS_bot", "What is a database schema?")
# print(response)



import streamlit as st
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import backend logic
# We use try-except to handle cases where the folder structure might slightly differ
try:
    from RAG.single_rag import query_chatbot, chatbot_configs
except ImportError as e:
    st.error(f"Error importing RAG modules: {e}")
    st.stop()

# --- Page Config ---
st.set_page_config(
    page_title="Academic RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- Custom CSS for better UI ---
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
    }
    .stMarkdown h3 {
        color: #2e86c1;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # 1. Subject/Model Selection
    st.subheader("Select Knowledge Base")
    
    # Get available bots from the config dictionary imported from single_rag
    # Mapping friendly names to bot IDs
    bot_options = list(chatbot_configs.keys())
    
    selected_bot = st.selectbox(
        "Choose a Subject:",
        options=bot_options,
        index=0,
        help="Select the specific academic domain you want to query."
    )
    
    st.divider()
    
    # API Key Status
    api_key = os.getenv("apikey")
    if api_key:
        st.success("API Key detected ✅")
    else:
        st.error("API Key missing ❌")
        st.warning("Please check your .env file")

    st.markdown("---")
    st.caption("Powered by Gemma-3 & ChromaDB")

# --- Main Interface ---
st.title("📚 Academic Context-Aware Chatbot")
st.markdown(f"**Current Context:** `{selected_bot}`")

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat history if the user switches bots
if "last_bot" not in st.session_state:
    st.session_state.last_bot = selected_bot

if st.session_state.last_bot != selected_bot:
    st.session_state.messages = []
    st.session_state.last_bot = selected_bot
    st.rerun()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & Processing ---
if prompt := st.chat_input("Ask a question about " + selected_bot.replace("_bot", "").replace("-", " ")):
    
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Process Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Retrieving context and generating answer..."):
            try:
                # Call the backend function
                # Note: query_chatbot in single_rag.py handles the retrieval and generation
                response_text = query_chatbot(selected_bot, prompt)
                
                # Simple typing effect for better UX
                full_response = response_text
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                full_response = f"⚠️ **Error:** {str(e)}"
                message_placeholder.error(full_response)
    
    # 3. Save Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": full_response})