import os
import time
import streamlit as st
from os import environ
from openai import OpenAI
# --- LangChain & OpenAI Imports ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# -----------------------------
# 🔧 Environment Configuration
# -----------------------------


# Initialize the LLM
llm = ChatOpenAI(
    model="openai.gpt-4o",
    temperature=0.2,
)
st.markdown('<style>body{background-color:powderblue;}</style>',unsafe_allow_html=True)

# -----------------------------
# 🌟 Streamlit UI
# -----------------------------
success_placeholder = st.empty()
st.title("🧠 File Q&A with Custom RAG")
st.caption("Upload `.txt` or `.pdf` files and chat with their content using a retrieval-augmented generation (RAG) pipeline.")

# File upload
uploaded_files = st.file_uploader(
    "Upload your documents",
    type=("txt", "pdf"),
    accept_multiple_files=True
)

if st.button("🗑 Clear All Documents"):
    if "vectorstore" in st.session_state:
        st.session_state.vectorstore = None
    if "messages" in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "All documents have been cleared. Upload new files to start again."}
        ]
    # Optionally remove uploaded files from disk
    if os.path.exists("uploaded_docs"):
        import shutil
        shutil.rmtree("uploaded_docs")
    os.makedirs("uploaded_docs", exist_ok=True)
    success_placeholder.success("✅ All uploaded files and embeddings have been removed.")
    time.sleep(2)
    success_placeholder.empty()

# -----------------------------
# 🧩 Session State Initialization
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi 👋! Upload one or more documents and ask me questions about them."}
    ]
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -----------------------------
# 📄 Document Ingestion & Chunking
# -----------------------------
def process_files(files):
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    os.makedirs("uploaded_docs", exist_ok=True)

    for f in files:
        file_path = os.path.join("uploaded_docs", f.name)
        with open(file_path, "wb") as tmp:
            tmp.write(f.getbuffer())

        # Load based on file type
        if f.name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif f.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            st.warning(f"Unsupported file type: {f.name}")
            continue

        documents = loader.load()
        split_docs = text_splitter.split_documents(documents)
        docs.extend(split_docs)
    

    # Create embeddings & store vectors
    embeddings=OpenAIEmbeddings(model="openai.text-embedding-3-large")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    return vectorstore

# Process uploaded files
if uploaded_files:
    with st.spinner("🔍 Processing and embedding your documents..."):
        st.session_state.vectorstore = process_files(uploaded_files)
    success_placeholder.success("✅ Documents uploaded and indexed successfully!")
    time.sleep(2)
    success_placeholder.empty()
    st.session_state.clear_uploader = True
    # st.rerun()


# -----------------------------
# Chat Display
# -----------------------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -----------------------------
# Chat Input & RAG Workflow
# -----------------------------
question = st.chat_input("Ask a question about your uploaded documents", disabled=not st.session_state.vectorstore)

def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

if question and st.session_state.vectorstore:
    # Display user question
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            # Retrieve
            docs = st.session_state.vectorstore.similarity_search(question, k=5)
            context = format_docs(docs)

            # Build prompt
            template = """
            You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise.

            Question: {question}

            Context: {context}

            Answer:
            """
            prompt = PromptTemplate.from_template(template)
            system_instructions = (
                "You are a helpful assistant for question answering.\n"
                "Use ONLY the provided context to answer concisely (<=3 sentences).\n"
                "If the answer isn't in the context, say you don't know.\n\n"
                f"Context:\n{context}"
            )

            # Ask the model
            response = llm.invoke([
                SystemMessage(content=system_instructions),
                HumanMessage(content=question),
            ])

            # Show answer
            st.write(response.content)

            # Optional: Sources
            with st.expander("📚 Sources"):
                for i, d in enumerate(docs, 1):
                    src = d.metadata.get("source", "(no source)")
                    st.markdown(f"[{i}] **{src}** — {d.page_content[:200]}...")

    # Append assistant reply
    st.session_state.messages.append({"role": "assistant", "content": response.content})
