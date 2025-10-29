# 🧠 File Q&A with RAG (LangChain + Streamlit + Chroma)

This project implements a **Retrieval-Augmented Generation (RAG)** application using **LangChain**, **Chroma**, and **Streamlit**.  
It allows users to upload `.txt` and `.pdf` files, which are then converted into embeddings for retrieval-based question answering with **GPT-4o**.

---

## 🚀 Overview

This application combines **document retrieval** and **conversational AI** to enable users to query their uploaded documents interactively.

**Main Features:**

- ✅ **Multi-file upload**: Supports `.txt` and `.pdf` formats.
- ✅ **Automatic text chunking**: Splits large files into manageable text chunks.
- ✅ **Embedding generation**: Uses `OpenAIEmbeddings` with the `text-embedding-3-large` model.
- ✅ **Chroma vector database**: Stores embeddings for similarity-based retrieval.
- ✅ **Question-answering chat**: Powered by `GPT-4o` through LangChain’s `ChatOpenAI` wrapper.
- ✅ **File management**: Includes a **“Clear All Documents”** button to remove all uploaded files and reset state.
- ✅ **Visual feedback**: Displays progress bars and success messages (no reruns).

---

## ⚙️ Setup and Installation

Follow these steps to run the application in your environment.

### 1️⃣ Clone the repository

```bash

git clone https://github.com/bhaveshgoyal27/INFO-5940-Codespace/tree/assignment1
cd INFO-5940-Codespace
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set up API credentials
API credentials already set up in devconatiner. If you have to change, please update it there.

### ▶️ Running the Application

Once dependencies and credentials are configured, run the app using:
```bash
streamlit run chat_with_pdf.py
```

Then open the provided local URL (usually http://localhost:8501).

### 🧩 How It Works

Upload files (.txt or .pdf)
The files are processed, chunked, embedded using OpenAI’s text-embedding-3-large, and stored in a local Chroma vector database.

Ask questions in the chat input box
Your question is converted to a vector, and the most relevant document chunks are retrieved.
The system then passes this context to GPT-4o, which generates a concise answer.

Clear all documents
You can remove all uploaded files and reset embeddings using the “Clear All Documents” button.

UI feedback
Progress bars and success banners provide confirmation after uploads and deletions, ensuring smooth interaction without page reruns.

### 🧱 Application Architecture

Frontend: Streamlit chat interface (st.chat_message, st.chat_input)

Backend: LangChain RAG pipeline

Embedding Model: text-embedding-3-large (OpenAI)
Vector Database: Chroma
LLM Model: openai.gpt-4o (via LangChain ChatOpenAI)

RAG Flow Summary:

Upload → 2. Chunk → 3. Embed → 4. Store → 5. Retrieve → 6. Answer

### 📂 Project Structure
```bash
📁 INFO-5940-Codespace/
│
├── chat_with_pdf.py       # Main Streamlit application
├── uploaded_docs/         # Temporary folder for uploaded files
├── requirements.txt       # Dependency list
└── README.md              # Documentation
```

### 🧠 Future Improvements

Add .docx and .md file support

Cache embeddings for faster reloads

Persist vectorstore across sessions
