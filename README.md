# Codebase RAG Assistant (Final)

✅ AST-Aware Chunking
✅ Docstring-Inclusive Retrieval
✅ Multi-Hop (MMR) Search
✅ Groq + Mixtral LLM
✅ Gradio Chatbot Interface

## Features

- Uses AST to extract functions and classes as chunks.
- Includes docstrings in embeddings for richer semantic retrieval.
- Multi-hop retrieval with MMR.
- Chatbot UI via Gradio.

## Quick Start

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Add your **Groq API key** in `app.py`:

```python
Groq(api_key="your-groq-api-key")
```

3. Add your `.py` files in the `codebase/` folder.

4. Run:

```
python app.py
```

5. Ask questions like:

- "Where is the login validation?"
- "What does UserManager do?"
- "Show me how users are added."
