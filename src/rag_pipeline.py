# File: src/rag_pipeline.py

import os
import pandas as pd
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from transformers import pipeline, set_seed # set_seed for deterministic LLM output

# --- Path Configuration (Standard for local scripts) ---
# Assuming this script is located in 'src/'
project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(project_root, os.pardir))

# Paths to data and vector store relative to project_root
VECTOR_STORE_PATH = os.path.join(project_root, 'vector_store', 'complaint_faiss_index')

# --- RAG Core Logic ---

def load_components():
    """
    Loads the embedding model, the persisted FAISS vector store, and the LLM pipeline.
    Returns:
        tuple: (FAISS vector store object, embedding model object, LLM pipeline object)
        Returns (None, None, None) if any component fails to load.
    """
    print("--- Loading RAG Components ---")

    embeddings = None
    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        print(f"Embedding model '{embeddings.model_name}' loaded successfully.")
    except ImportError:
        print("Error: 'sentence-transformers' library not found. Please install it: pip install sentence-transformers")
        return None, None, None
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None, None, None

    db = None
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            # allow_dangerous_deserialization=True is needed for security when loading local FAISS indices
            db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            print("FAISS vector store loaded successfully!")
        except ImportError:
            print("Error: 'faiss-cpu' library not found. Please install it: pip install faiss-cpu")
            return None, None, None
        except Exception as e:
            print(f"Error loading FAISS vector store from '{VECTOR_STORE_PATH}': {e}")
            print("Ensure the index was saved correctly by 'src/embed_and_index.py'.")
            return None, None, None
    else:
        print(f"Error: Vector store directory '{VECTOR_STORE_PATH}' not found.")
        print("Please run 'src/embed_and_index.py' first to generate and persist the vector store.")
        return None, None, None

    llm_pipeline = None
    try:
        # Using a text2text generation model like T5-small for demonstration.
        # For better answers, consider larger models, but they require more resources.
        # device=0 for GPU, -1 for CPU. Automatically detects if CUDA is available.
        llm_pipeline = pipeline("text2text-generation", model="t5-small", device=0 if os.environ.get('CUDA_VISIBLE_DEVICES', '0')=='0' else -1)
        print("Text generation LLM (t5-small) loaded successfully.")
        set_seed(42) # For reproducible LLM outputs during evaluation
    except Exception as e:
        print(f"Error loading text generation LLM: {e}")
        print("Please ensure 'transformers' and 'accelerate' libraries are installed and model can be downloaded.")
        return None, None, None

    print("--- All RAG Components Loaded ---")
    return db, embeddings, llm_pipeline

def retrieve_chunks(question: str, db: FAISS, k: int = 5) -> list[Document]:
    """
    Retrieves top-k most relevant text chunks from the vector store based on the question embedding.
    Args:
        question (str): The user's question.
        db (FAISS): The FAISS vector store.
        k (int): Number of top relevant chunks to retrieve.
    Returns:
        list[Document]: A list of LangChain Document objects.
    """
    if not db:
        print("Vector store not initialized. Cannot retrieve chunks.")
        return []
    print(f"Retrieving top {k} chunks for query: '{question}'")
    try:
        # Using similarity_search to get just the documents.
        # Use similarity_search_with_score if you need the scores for analysis.
        docs = db.similarity_search(query=question, k=k)
        print(f"Retrieved {len(docs)} chunks.")
        return docs
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []

def generate_answer(question: str, context_chunks: list[Document], llm_pipeline) -> str:
    """
    Combines the question and retrieved context into a prompt and generates an answer using the LLM.
    Args:
        question (str): The user's question.
        context_chunks (list[Document]): List of retrieved LangChain Document objects.
        llm_pipeline: The initialized Hugging Face transformers pipeline for text generation.
    Returns:
        str: The generated answer from the LLM.
    """
    if not llm_pipeline:
        return "LLM generator not initialized."
    if not context_chunks:
        return "I don't have enough information from the retrieved context to answer this question."

    context_text = "\n\n".join([doc.page_content for doc in context_chunks])

    # Prompt Template (Critical for guiding the LLM)
    prompt_template = """You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use ONLY the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}
Answer:"""

    formatted_prompt = prompt_template.format(context=context_text, question=question)

    print("\nGenerating answer with LLM...")
    try:
        # max_new_tokens limits the response length to avoid excessively long generations.
        # do_sample=False for deterministic outputs; True for more creative/varied.
        # truncation=True handles cases where the combined prompt exceeds model's max input length.
        generated_response = llm_pipeline(formatted_prompt, max_new_tokens=200, do_sample=False, truncation=True)[0]['generated_text']
        return generated_response
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return "An error occurred during answer generation."

def run_rag_pipeline(question: str, db: FAISS, llm_pipeline, k: int = 5):
    """
    Executes the full RAG pipeline: retrieves chunks and generates an answer.
    """
    retrieved_docs = retrieve_chunks(question, db, k=k)
    answer = generate_answer(question, retrieved_docs, llm_pipeline)
    return answer, retrieved_docs

if __name__ == "__main__":
    # This block will run if you execute src/rag_pipeline.py directly (for quick testing)
    print("--- Running RAG Pipeline Script Directly (for testing/demonstration) ---")
    db, embeddings, llm_pipeline = load_components()

    if db and embeddings and llm_pipeline:
        test_question = "What was the issue with the credit card statement?"
        print(f"\nTest Question: '{test_question}'")
        answer, retrieved_docs = run_rag_pipeline(test_question, db, llm_pipeline)

        print("\n--- RAG Output ---")
        print(f"Question: {test_question}")
        print(f"Generated Answer: {answer}")
        print("\nRetrieved Sources (Top 3 Content Preview):")
        for i, doc in enumerate(retrieved_docs[:3]):
            print(f"  Source {i+1} (ID: {doc.metadata.get('complaint_id', 'N/A')}, Product: {doc.metadata.get('product', 'N/A')}): {doc.page_content[:150]}...")
    else:
        print("Failed to initialize RAG components.")
    print("\n--- RAG Pipeline Script Finished ---")