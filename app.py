# File: app.py

import gradio as gr
import os
import sys

# --- Path Configuration ---
# This app.py script is assumed to be in the project root.
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the 'src' directory to Python's path so we can import rag_pipeline
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import the rag_pipeline module
import rag_pipeline

# --- Global RAG Components (loaded once when the app starts) ---
print("Initializing RAG components for the Gradio app...")
# This will load the FAISS DB, embedding model, and LLM (t5-small)
db, embeddings, llm_pipeline = rag_pipeline.load_components()

# Check if RAG components loaded successfully. If not, the app will still launch
# but will inform the user of the failure and won't process queries.
if not (db and embeddings and llm_pipeline):
    print("\nERROR: Failed to load all RAG components. The Gradio app cannot process queries.")
    print("Please ensure:")
    print("1. All required libraries are installed (`pip install transformers accelerate`).")
    print("2. `src/embed_and_index.py` ran successfully and created the vector store (`vector_store/complaint_faiss_index/`).")
    print("3. You are running `app.py` from the project root directory.")
    # Set them explicitly to None to prevent calls if loading failed
    db, embeddings, llm_pipeline = None, None, None

# --- Gradio Chat Interface Function ---

def respond(message, history):
    """
    This function processes user messages, retrieves context, and generates an answer.
    It handles the Gradio ChatInterface interaction.
    """
    # Check if RAG components were successfully initialized at startup
    if not (db and embeddings and llm_pipeline):
        # If not initialized, return an error message to the user
        return history + [{"role": "assistant", "content": "Sorry, the RAG system is not initialized. Please check the server logs for errors."}]

    # Provide immediate feedback to the user (simulated thinking/searching)
    yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Searching for relevant complaints and generating response..."}]
    
    try:
        # Run the core RAG pipeline (retrieval + generation)
        answer, retrieved_docs = rag_pipeline.run_rag_pipeline(message, db, llm_pipeline, k=5)

        # Format the generated answer
        final_message_content = f"{answer}"
        
        # Append sources for display below the answer for transparency and trust
        sources_text = ""
        if retrieved_docs:
            sources_text += "\n\n**Sources:**\n"
            # Display top 3 sources (or fewer if less are retrieved)
            for i, doc in enumerate(retrieved_docs[:3]): 
                comp_id = doc.metadata.get('complaint_id', 'N/A')
                product = doc.metadata.get('product', 'N/A')
                # Limit content preview to 200 characters for readability in UI
                # Escape markdown sensitive characters like '*' and '_' in content preview
                content_preview = doc.page_content[:200].replace('*', '\\*').replace('_', '\\_')
                sources_text += f"- **Complaint ID:** {comp_id}, **Product:** {product}\n  Content: \"{content_preview}...\"\n"
        else:
            sources_text += "\n\n**Sources:** No relevant sources found in the knowledge base."

        # Combine the answer and sources for the final message in chat history
        final_message_content += sources_text

        # Yield the final formatted response. This replaces the previous "Searching..." message.
        yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": final_message_content}]

    except Exception as e:
        # Catch any unexpected errors during RAG execution and display a user-friendly message
        error_message = "An internal error occurred while generating the response. Please try again later."
        print(f"Error during RAG response generation: {e}") # Log full error to console for debugging
        yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_message}]
        
# --- Gradio Interface Setup ---

# Construct the description string with developer attribution
app_description = (
    "Ask me questions about customer complaints related to credit cards, personal loans, BNPL, "
    "savings accounts, and money transfers. (Powered by t5-small LLM)"
    "\n\n**Developed By Desta Getaw**" # Attribution added here
)

gr.ChatInterface(
    fn=respond, # The function that processes user input
    chatbot=gr.Chatbot(height=500, type='messages'), 
    textbox=gr.Textbox(placeholder="Ask a question about customer complaints...", container=False, scale=7),
    title="CrediTrust Financial Complaint Analyst Assistant (Desta RAG App)", # App title for the UI
    description=app_description, # Use the combined description
    theme="soft", # Visual theme for the app
    examples=[ # Predefined questions users can click to quickly test the app
        "What common issues are reported with credit card billing?",
        "Can you find complaints about unauthorized transactions in savings accounts?",
        "Summarize recent issues with Buy Now, Pay Later services."
    ],
    # `clear_btn` was removed in previous steps due to TypeError.
    # `footer` is not consistently visible in your Gradio setup.
    # Gradio's ChatInterface provides a default "Reset" button (an 'x' icon)
    # in the chat history, which is intuitive for clearing the conversation.
).launch(share=False)

print("\nGradio app launched. Check your terminal for the local URL.")