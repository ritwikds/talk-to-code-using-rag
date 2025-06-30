import gradio as gr
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import shutil
import tempfile
from ast_chunker import ASTCodeChunker
from git import Repo

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

def load_code_documents(directory):
    chunker = ASTCodeChunker()
    all_docs = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".py"):
                path = os.path.join(root, file_name)
                with open(path, "r", encoding="utf-8") as f:
                    code = f.read()
                docs = chunker.chunk_code(code, file_path=path)
                all_docs.extend(docs)
    return all_docs

def create_vectorstore(docs, persist_directory):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)
    return vectorstore

def get_retriever(persist_directory):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})

def setup_chain(persist_directory):
    retriever = get_retriever(persist_directory)
    llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=api_key)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def prepare_codebase(repo_url):
    temp_dir = tempfile.mkdtemp()
    try:
        Repo.clone_from(repo_url, temp_dir)
        docs = load_code_documents(temp_dir)
        db_dir = os.path.join(temp_dir, "db")
        create_vectorstore(docs, db_dir)
        return db_dir, temp_dir
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e

def chat_with_repo(history, repo_url, message):
    if not repo_url or not repo_url.strip():
        return history + [[message, "Please provide a valid GitHub repository URL."]]
    try:
        db_dir, temp_dir = prepare_codebase(repo_url)
        qa_chain = setup_chain(db_dir)
        answer = qa_chain.run(message)
        shutil.rmtree(temp_dir)
        return history + [[message, answer]]
    except Exception as e:
        return history + [[message, f"Error: {str(e)}"]]

with gr.Blocks() as demo:
    gr.Markdown("# Codebase Assistant: Chat with any GitHub repo")
    repo_url = gr.Textbox(label="GitHub Repository URL", placeholder="https://github.com/user/repo")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your question")
    state = gr.State([])

    def respond(history, repo_url, message):
        return chat_with_repo(history, repo_url, message), repo_url, ""

    msg.submit(respond, [state, repo_url, msg], [chatbot, repo_url, msg], queue=False)
    chatbot.style(height=400)

demo.launch()
