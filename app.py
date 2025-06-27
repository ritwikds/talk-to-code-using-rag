"""
app.py

This is the main entry point for your Codebase Assistant.
It uses:
- AST-aware chunking with docstrings to split code into meaningful blocks
- Chroma vector store for retrieval
- HuggingFace MiniLM embeddings
- Groq's Mixtral model as the LLM
- Gradio UI for chatting with your codebase

Author: Your Name
"""

import gradio as gr
from langchain.chains import RetrievalQA
from langchain.llms import Groq
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os

from ast_chunker import ASTCodeChunker

def load_code_documents(directory="codebase"):
    """
    Loads and chunks all .py files in the given directory using ASTCodeChunker.

    Returns:
        list of Document: The parsed and chunked documents ready for embedding.
    """
    chunker = ASTCodeChunker()
    all_docs = []

    # Iterate over all Python files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith(".py"):
            path = os.path.join(directory, file_name)
            with open(path, "r") as f:
                code = f.read()

            # Use AST chunker to split code into logical chunks with docstrings
            docs = chunker.chunk_code(code, file_path=path)
            all_docs.extend(docs)
    return all_docs

def create_vectorstore(docs):
    """
    Creates a Chroma vector store from given documents.
    Persists embeddings locally in the 'db' folder.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="db")
    return vectorstore

def get_retriever():
    """
    Loads the persisted vector store and returns a retriever
    using MMR (Maximal Marginal Relevance) for multi-hop-style diverse retrieval.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})

def setup_chain():
    """
    Sets up the RetrievalQA chain using Groq's Mixtral model.
    """
    retriever = get_retriever()
    llm = Groq(model="mixtral-8x7b-32768", api_key="your-groq-api-key")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Build the vector store once if it doesn't exist
if not os.path.exists("db/index"):
    docs = load_code_documents()
    create_vectorstore(docs)

# Create the RetrievalQA chain for answering questions
qa_chain = setup_chain()

def chat_with_codebase(question):
    """
    Given a user question, retrieves relevant code chunks and generates an answer.
    """
    return qa_chain.run(question)

# Launch Gradio Chatbot UI
gr.Interface(
    fn=chat_with_codebase,
    inputs="text",
    outputs="text",
    title="Codebase Assistant with AST + Docstrings + Multi-hop Retrieval",
    description=(
        "Ask questions about your codebase. "
        "This assistant uses AST parsing with docstrings for better chunking, "
        "HuggingFace embeddings, Chroma vector store with MMR retrieval, and Groq's Mixtral LLM."
    )
).launch()
