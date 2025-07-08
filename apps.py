# File: app.py

import streamlit as st
import os
import sys
import time # For simulating typing/streaming effect

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="CrediTrust Financial Complaint Analyst Assistant",
    page_icon="🤖",
    layout="centered", # Can be "wide" for more horizontal space
    initial_sidebar_state="auto"
)

# --- Path Configuration ---
# This app.py script is assumed to be in the project root.
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the 'src' directory to Python's path so we can import rag_pipeline
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import the rag_pipeline module
import rag_pipeline

# --- Global RAG Components (loaded once when the app starts) ---
# Use st.cache_resource to cache these heavy objects across reruns
@st.cache_resource
def initialize_rag_components():
    """
    Initializes and caches the RAG components (FAISS DB, embeddings, LLM pipeline).
    This function will only run once per Streamlit app session.
    """
    st.info("Initializing RAG components. This may take a moment...", icon="⏳")
    db, embeddings, llm_pipeline = rag_pipeline.load_components()
    if not (db and embeddings and llm_pipeline):
        st.error("Failed to load all RAG components. The application cannot process queries.")
        st.warning("Please ensure `src/embed_and_index.py` ran successfully and all required libraries are installed.")
        return None, None, None
    st.success("RAG components loaded successfully!")
    return db, embeddings, llm_pipeline

# Load components only once
db, embeddings, llm_pipeline = initialize_rag_components()

# --- Streamlit UI Elements ---
st.title("CrediTrust Financial Complaint Analyst Assistant")
st.markdown(
    """
    Ask me questions about customer complaints related to credit cards, personal loans, 
    Buy Now, Pay Later (BNPL), savings accounts, and money transfers. 
    (Powered by t5-small LLM)
    """
)

# Developer Attribution
st.sidebar.markdown("### About This App")
st.sidebar.markdown(
    "This is a demo of a Retrieval Augmented Generation (RAG) system "
    "designed to answer questions based on a dataset of consumer complaints."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Developed By **Desta Getaw**")

# --- Chat History Management (using Streamlit's session_state) ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores conversation history as a list of {"role": ..., "content": ...} dicts
if "last_llm_response" not in st.session_state:
    st.session_state.last_llm_response = "" # Stores the last full answer for source display
if "last_retrieved_sources" not in st.session_state:
    st.session_state.last_retrieved_sources = [] # Stores the last retrieved sources for display

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Clear Chat Button ---
def clear_chat_history():
    """Clears the chat history and last response/sources from session state."""
    st.session_state.messages = []
    st.session_state.last_llm_response = ""
    st.session_state.last_retrieved_sources = []
    st.rerun() # Rerun the app to clear the display

st.button("Clear Chat", on_click=clear_chat_history)


# --- Text Input and Response Generation ---
# Use st.chat_input for the modern chat UI
if prompt := st.chat_input("Ask a question about complaints..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process and display AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Create an empty container to fill with streaming text
        full_response_content = ""
        
        # Check if RAG components are loaded
        if not (db and embeddings and llm_pipeline):
            full_response_content = "Sorry, the RAG system is not fully loaded or encountered an error. Please try refreshing the page or check the server logs."
            st.error(full_response_content) # Display error in the chat
        else:
            # Run the RAG pipeline to get the answer and sources
            answer, retrieved_docs = rag_pipeline.run_rag_pipeline(prompt, db, llm_pipeline, k=5)
            
            # --- Simulate Streaming Effect (Optional but recommended) ---
            # t5-small pipeline doesn't inherently stream token by token.
            # We simulate it by splitting the full answer into words and yielding them.
            for word in answer.split():
                full_response_content += word + " "
                time.sleep(0.02) # Small delay for typing effect
                message_placeholder.markdown(full_response_content + "▌") # Cursor effect
            message_placeholder.markdown(full_response_content) # Display final complete message

            # Store the full response and sources in session state for display outside the chat history
            st.session_state.last_llm_response = full_response_content
            st.session_state.last_retrieved_sources = retrieved_docs

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response_content})
    # This `st.rerun()` is important to trigger a refresh after the prompt is processed.
    # It ensures the "Sources" section updates immediately.
    st.rerun()


# --- Display Sources Area (below the main chat history) ---
if st.session_state.last_llm_response: # Only show if there's a recent LLM response
    st.markdown("---") # Separator
    st.subheader("Last Answer's Sources")

    if st.session_state.last_retrieved_sources:
        for i, doc in enumerate(st.session_state.last_retrieved_sources[:3]): # Display top 3 sources
            st.markdown(f"**Source {i+1}:**")
            st.markdown(f"- **Complaint ID:** `{doc.metadata.get('complaint_id', 'N/A')}`")
            st.markdown(f"- **Product:** `{doc.metadata.get('product', 'N/A')}`")
            st.expander(f"**Content Preview (Click to expand)**").markdown(f"```\n{doc.page_content}\n```")
    else:
        st.info("No relevant sources were retrieved for the last question.")

# Final message for user
st.caption("Enter your question above to start the conversation.")