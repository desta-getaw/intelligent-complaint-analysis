# File: src/embed_and_index.py

import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document # For clarity on Document object structure

def run_embedding_and_indexing():
    """
    Loads cleaned data, chunks narratives, generates embeddings,
    and creates/persists a FAISS vector store.
    """
    print("--- Starting Text Chunking, Embedding, and Vector Store Indexing (src/embed_and_index.py) ---")

    # --- 0. Set up Paths ---
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(project_root, os.pardir))

    input_csv_path = os.path.join(project_root, 'data', 'filtered', 'filtered_complaints.csv')

    vector_store_dir = os.path.join(project_root, 'vector_store')
    try:
        if not os.path.exists(vector_store_dir):
            os.makedirs(vector_store_dir)
            print(f"Created directory: {vector_store_dir}")
    except OSError as e:
        print(f"Error creating vector_store directory {vector_store_dir}: {e}")
        print("Please check directory permissions.")
        return

    faiss_index_path = os.path.join(vector_store_dir, "complaint_faiss_index")


    # --- 1. Load the Cleaned and Filtered Dataset ---
    df = None # Initialize df to None
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Cleaned and filtered dataset loaded successfully from '{input_csv_path}'!")
        print(f"Total rows in loaded DataFrame: {df.shape[0]}, Total columns: {df.shape[1]}")
    except FileNotFoundError:
        print(f"Error: Cleaned and filtered data '{input_csv_path}' not found.")
        print("Please ensure 'src/preprocess.py' has been run successfully to generate this file.")
        print("Exiting...")
        return # Exit if the file is not found
    except pd.errors.EmptyDataError:
        print(f"Error: '{input_csv_path}' is empty. No data to process.")
        print("Please check your preprocessing steps.")
        return # Exit if file is empty
    except Exception as e:
        print(f"An unexpected error occurred while loading '{input_csv_path}': {e}")
        return

    # Check if DataFrame is empty after loading
    if df.empty:
        print("Loaded DataFrame is empty. No narratives to process.")
        print("Exiting...")
        return

    # Ensure 'Cleaned_Narrative' and 'Complaint ID' columns exist
    required_cols = ['Cleaned_Narrative', 'Complaint ID', 'Product']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        found_cols = df.columns.tolist() # Get the actual columns found
        print(f"Error: Missing required columns: {missing_cols} in the dataset.")
        print(f"Columns actually found: {found_cols}") # <<< This will tell us what's really there
        print("Please check your preprocessing script and input data.")
        print("Exiting...")
        return

    # --- 2. Text Chunking Strategy ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # characters (approx 250-300 words)
        chunk_overlap=150, # characters (10% overlap for context continuity)
        length_function=len,
        add_start_index=True, # Adds 'start_index' to metadata for debugging
    )
    # THIS LINE IS THE FIX FOR THE ATTRIBUTEERROR:
    print(f"\nInitialized Text Splitter with chunk_size={text_splitter._chunk_size} and chunk_overlap={text_splitter._chunk_overlap}")

    documents = []
    processed_narratives_count = 0
    for index, row in df.iterrows():
        narrative = str(row['Cleaned_Narrative']).strip()
        if pd.notna(narrative) and narrative != '': # Check if narrative is valid after stripping
            processed_narratives_count += 1
            doc = Document(
                page_content=narrative,
                metadata={
                    "complaint_id": row['Complaint ID'],
                    "product": row['Product'],
                    "original_text_length": len(narrative)
                }
            )
            chunks = text_splitter.split_documents([doc])
            documents.extend(chunks)

    print(f"\nProcessed {processed_narratives_count} non-empty narratives.")
    print(f"Original narratives split into {len(documents)} chunks.")

    if len(documents) == 0:
        print("No chunks generated. This could mean all narratives were empty after cleaning.")
        print("Exiting...")
        return


    if len(documents) > 0:
        print(f"First chunk example (first 200 chars): '{documents[0].page_content[:200]}...'")
        print(f"First chunk metadata: {documents[0].metadata}")


    # --- 3. Choose an Embedding Model ---
    embeddings = None # Initialize embeddings to None
    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        print(f"\nEmbedding model '{embeddings.model_name}' loaded successfully.")
    except ImportError:
        print("Error: 'sentence-transformers' library not found.")
        print("Please install it using: pip install sentence-transformers")
        print("Exiting...")
        return
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        print("Exiting...")
        return

    # --- 4. Embedding and Indexing (FAISS) ---
    print("\nAttempting to create FAISS vector store and generate embeddings...")
    db = None
    try:
        db = FAISS.from_documents(documents, embeddings)
        print("FAISS vector store created successfully.")

        # --- 5. Persist the vector store ---
        db.save_local(faiss_index_path)
        print(f"FAISS vector store persisted to: {faiss_index_path}")
    except ImportError:
        print("Error: 'faiss-cpu' library not found.")
        print("Please install it using: pip install faiss-cpu")
        print("Exiting...")
        return
    except Exception as e:
        print(f"An unexpected error occurred during FAISS indexing or saving: {e}")
        print("This might indicate an issue with the FAISS library or the data format.")
        print("Exiting...")
        return

    print("\n--- Text Chunking, Embedding, and Vector Store Indexing Complete ---")

if __name__ == "__main__":
    run_embedding_and_indexing()