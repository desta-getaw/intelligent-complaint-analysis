# 📊 Intelligent Complaint Analysis for Financial Services

## 📝 Introduction
CrediTrust Financial is a fast-growing digital finance company serving over 500,000 users across East Africa with products such as Credit Cards, Personal Loans, Buy Now Pay Later (BNPL), Savings Accounts, and Money Transfers. The company receives thousands of unstructured customer complaints every month.

This project aims to transform this raw complaint data into actionable insights by building an **AI-powered chatbot** using Retrieval-Augmented Generation (RAG). This tool will help product managers, support teams, and compliance officers quickly identify customer pain points and trends.

---

## 🎯 Business Objectives & Understanding
The key business objectives are:
- Reduce the time it takes to identify major complaint trends from days to minutes.
- Empower non-technical teams to get answers without needing a data analyst.
- Shift from reactive problem-solving to proactively identifying issues based on real-time customer feedback.

We’re building this tool specifically for internal stakeholders, like product managers who spend significant time manually reading complaints, and support & compliance teams who currently face serious bottlenecks due to scattered and high-volume data.

---

## 📂 Selected Dataset Types
We are using the **Consumer Financial Protection Bureau (CFPB) complaint dataset**, which includes:
- Short issue labels (e.g., “Billing dispute”)
- Free-text consumer narratives
- Product and company information
- Submission date and metadata

The core input to our AI chatbot is the **free-text complaint narrative**, filtered to include only five product categories:
- Credit Cards
- Personal Loans
- Buy Now Pay Later (BNPL)
- Savings Accounts
- Money Transfers

---

## ⚙️ Methodology

### ✅ GitHub Setup & Branches
- The project uses GitHub for version control.
- We created three branches:
  - `main`: the production-ready branch
  - `task-1`: EDA and data preprocessing
  - `task-2`: Embedding, indexing, and RAG pipeline

These branches help isolate and merge work cleanly.

---

### 📌 Detailed Tasks & Activities

#### **Task 1: Exploratory Data Analysis & Data Preprocessing**
- Load CFPB complaint dataset.
- Analyze complaint distribution by product.
- Visualize narrative lengths (word count) and identify short/long narratives.
- Filter dataset to the five target products and remove empty narratives.
- Clean text by lowercasing, removing special characters, and normalizing text.
- **Deliverables:**
  - Notebook/script in `notebooks/` or `src/`
  - Filtered dataset saved as `data/filtered_complaints.csv`
  - Short summary of key findings

---

#### **Task 2: Text Chunking, Embedding & Vector Store Indexing**
- Implement text chunking (e.g., using LangChain’s `RecursiveCharacterTextSplitter`).
- Choose and justify embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`).
- Generate embeddings and store in vector database (e.g., FAISS or ChromaDB).
- Save vector store and metadata in `vector_store/`.
- **Deliverables:**
  - Python script performing chunking, embedding, and indexing
  - Report section explaining chunking strategy and model choice

---

#### **Task 3: Build RAG Core Logic & Evaluate**
- Retrieve top-k (e.g., k=5) most relevant complaint chunks for a user’s question.
- Design prompt template to guide LLM (e.g., instruct it to use only retrieved context).
- Integrate LLM (e.g., via Hugging Face or LangChain).
- Evaluate with a set of representative questions and create an evaluation table:
  - Question | Generated Answer | Retrieved Sources | Quality Score (1–5) | Comments
- **Deliverables:**
  - Python module with RAG logic
  - Evaluation table and analysis in report

---

#### **Task 4: Interactive Chat Interface**
- Build a user-friendly web interface using **Streamlit** or **Gradio**.
- Features:
  - Text input box
  - “Ask” button
  - Display AI-generated answer and retrieved source text
  - (Optional) Streaming answers
  - “Clear” button to reset
- **Deliverables:**
  - `app.py` script running the app
  - Screenshots or GIF of the working application

---

## 🛠️ Tools Used
- Python (Data processing, embedding, pipeline development)
- Git & GitHub (version control, collaboration, CI/CD)
- Jupyter Notebook / VS Code (development environment)
- FAISS / ChromaDB (vector database)
- Hugging Face Transformers (embedding & LLM integration)
- LangChain (retrieval pipeline and orchestration)
- Streamlit or Gradio (interactive UI)
- CI/CD tools for GitHub Actions

---

## ✅ Conclusion
By following this structured approach, we aim to build a production-ready AI tool that empowers teams to turn thousands of customer complaints into actionable business insights quickly and reliably.

For detailed technical code, task notebooks, and final deployment, see our GitHub repository.