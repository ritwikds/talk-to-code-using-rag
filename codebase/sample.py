"""
PDF Chatbot with LangChain, Gradio, and Groq

This app allows you to upload a PDF and chat with it.
It:
- Loads and splits the PDF
- Embeds it into a Chroma vector DB
- Uses retrieval-augmented generation (RAG) with query rewriting, reranking, and answer generation
- Provides a Gradio UI for interactive Q&A

Dependencies:
- langchain
- langchain-community
- langchain-groq
- chromadb
- gradio
"""

import os
import re
import uuid

import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema import Document
from langchain_groq import ChatGroq


# === Check Groq API Key ===
print("Groq key exists:", bool(os.getenv("GROQ_API_KEY")))
print("Groq key value:", os.getenv("GROQ_API_KEY"))

# === Embeddings ===
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# === Prompt Templates and Chains ===

# Prompt to rewrite user questions for better retrieval
rewrite_prompt = PromptTemplate.from_template("""
Rewrite the following question to improve document search relevance.

Original Question: {question}

Rewritten Search Query:""")
rewrite_chain = LLMChain(
    llm=ChatGroq(model_name="llama-3.1-8b-instant", temperature=0),
    prompt=rewrite_prompt
)

# Prompt to answer the user's question given context
answer_prompt = PromptTemplate.from_template("""
Use the following context to answer the question. 
If you're unsure, say you don't know. Don't make things up.

Context:
{context}

Question:
{question}

Answer:""")
answer_chain = StuffDocumentsChain(
    llm_chain=LLMChain(
        llm=ChatGroq(model_name="llama-3.1-8b-instant", temperature=0),
        prompt=answer_prompt
    ),
    document_variable_name="context"
)

# Prompt for reranking retrieved documents
rerank_prompt = PromptTemplate.from_template("""
Given the following query and document, rate how relevant the document is to the query on a scale of 1-10.

Query:
{query}

Document:
{document}

Score (1-10):""")
rerank_chain = LLMChain(
    llm=ChatGroq(model_name="llama-3.1-8b-instant", temperature=0),
    prompt=rerank_prompt
)


# === Global State ===
retriever = None
vector_store = None
memory = None


def create_vector_db_for_pdf(docs, file_id=None):
    """
    Creates a persistent Chroma vector store for the given documents.

    Args:
        docs (List[Document]): The split and processed documents.
        file_id (str, optional): A unique ID for the collection (e.g., filename). 
                                 If not given, generates a UUID.

    Returns:
        Chroma: The vector store containing the embedded documents.
    """
    collection_name = file_id or str(uuid.uuid4())

    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=f"./chroma_dbs/{collection_name}"
    )

    vector_db.add_documents(docs)
    vector_db.persist()

    return vector_db


def load_pdf(file):
    """
    Gradio event handler to load a PDF file, split it into chunks,
    embed it, and initialize the retriever.

    Args:
        file (gr.File): Uploaded PDF file.

    Yields:
        str: Status messages for Gradio UI.
        gr.update: Visibility updates for Gradio components.
    """
    global retriever, vector_store, memory

    yield "⏳ Processing PDF...", gr.update(visible=False)

    # Load PDF content
    loader = PyMuPDFLoader(file.name)
    raw_docs = loader.load()

    # Split text into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(raw_docs)

    # Use file name as unique ID for collection
    file_id = os.path.splitext(os.path.basename(file.name))[0]

    # Create and store vector DB
    vector_store = create_vector_db_for_pdf(docs, file_id=file_id)

    # Create retriever with similarity search
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    # Initialize memory for conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    yield "✅ PDF processed. Ask your question!", gr.update(visible=True)


def keyword_match(docs, query):
    """
    Returns top 4 documents with the highest keyword match count.

    Args:
        docs (List[Document]): Documents to search.
        query (str): User's search query.

    Returns:
        List[Document]: Top 4 keyword-matched documents.
    """
    keywords = set(re.findall(r'\b\w+\b', query.lower()))
    scored = []
    for doc in docs:
        content = doc.page_content.lower()
        matches = sum(1 for kw in keywords if kw in content)
        if matches > 0:
            scored.append((matches, doc))

    # Sort by number of matches
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:4]]


def rerank_documents(query, docs):
    """
    Uses the rerank LLM chain to score and rank documents.

    Args:
        query (str): User's search query.
        docs (List[Document]): Candidate documents.

    Returns:
        List[Document]: Top 4 reranked documents.
    """
    scored_docs = []
    for doc in docs:
        score_str = rerank_chain.run({"query": query, "document": doc.page_content})
        try:
            # Extract numeric score from model output
            score = int(re.search(r"\d+", score_str).group())
        except:
            score = 5  # Fallback score
        scored_docs.append((score, doc))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:4]]


def respond(message, history):
    """
    Responds to a user query by:
    - Rewriting the query
    - Running similarity and keyword retrieval
    - Combining and reranking documents
    - Generating an answer

    Args:
        message (str): User's message.
        history (List[Tuple[str, str]]): Conversation history.

    Returns:
        Tuple[str, List[Tuple[str, str]], List[Tuple[str, str]]]:
            - Empty string to clear input box
            - Updated conversation history
            - Same updated history (for Gradio's multiple outputs)
    """
    if not retriever:
        return "", history, history

    # Rewriting the query
    rewritten_query = rewrite_chain.run({"question": message})
    print(f"📝 Rewritten Query: {rewritten_query}")

    # Adjust retrieval 'k' adaptively based on query length
    k = 3 if len(rewritten_query.split()) <= 6 else 8
    retriever.search_kwargs["k"] = k
    sim_docs = retriever.get_relevant_documents(rewritten_query)

    # Keyword matching
    all_docs = vector_store.similarity_search(rewritten_query, k=50)
    keyword_docs = keyword_match(all_docs, rewritten_query)

    # Combine and remove duplicates
    combined_docs = list({doc.page_content: doc for doc in (sim_docs + keyword_docs)}.values())

    # Rerank
    top_docs = rerank_documents(rewritten_query, combined_docs)

    # Combine context for debugging or displaying
    context_texts = [doc.page_content for doc in top_docs]
    combined_context = "\n\n---\n\n".join(context_texts)

    # Generate answer
    answer = answer_chain.run({"input_documents": top_docs, "question": message})

    # Add to chat history
    history.append((
        f"**Q:** {message}\n\n**Context:**\n{combined_context}",
        f"**A:** {answer}"
    ))

    return "", history, history


# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 Chat with your PDF ")

    # File upload section
    file_input = gr.File(label="Upload PDF")
    upload_status = gr.Markdown()
    chat_column = gr.Column(visible=False)

    with chat_column:
        chatbot = gr.Chatbot(label="Document QA Bot")
        msg = gr.Textbox(placeholder="Ask a question and press Enter", show_label=False)
        clear = gr.Button("Clear")

    # File upload triggers PDF processing
    file_input.change(fn=load_pdf, inputs=file_input, outputs=[upload_status, chat_column])

    def respond_wrapper(message, history):
        """
        Wrapper for respond() to fit Gradio interface.

        Returns:
            Updated input box (empty), updated chat history.
        """
        updated_msg, updated_chat, _ = respond(message, history)
        return updated_msg, updated_chat

    # User sends message
    msg.submit(fn=respond_wrapper, inputs=[msg, chatbot], outputs=[msg, chatbot])

    # Clear conversation
    def clear_all():
        """
        Clears the chat and status.
        """
        return [], "", ""

    clear.click(fn=clear_all, outputs=[chatbot, msg, upload_status])


if __name__ == "__main__":
    demo.launch()
