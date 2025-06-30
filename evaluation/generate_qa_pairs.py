"""
generate_qa_pairs.py

Uses an LLM to read your codebase and synthesize ~50 question-answer pairs
that you can use as an evaluation set for your RAG assistant.
"""

import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import re

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
# print(api_key)

# Load your big code as a single string
with open("C:/Users/nagasatwik.parla/Downloads/Agents_Practice/codebase_rag_assistant_final/codebase/sample.py", encoding="utf-8", errors="replace") as f:
    code_text = f.read()

# Define the prompt
generation_prompt = PromptTemplate.from_template("""
You are an expert software reviewer.

Given this code:

{code}

Generate a set of 20 realistic questions a developer might ask about this codebase,
along with short, correct answers. The questions should be diverse,
covering functions, classes, variables, design choices, models used, and so on.

Format them as JSON list of objects:

  {{"question": "...", "answer": "..."}},

""")

# LLM setup
llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=api_key, temperature=0.5)
chain = LLMChain(llm=llm, prompt=generation_prompt)

# Run
result = chain.run({"code": code_text})
result_json = re.search(r"\[.*\]", result, re.DOTALL).group(0)

# print(result, type(result))
# Parse and save
try:
    qa_pairs = json.loads(result_json)
except Exception:
    print("Error parsing LLM output. Please fix formatting manually.")
    with open("evaluation/raw_output.txt", "w") as f:
        f.write(result)
    exit()

with open("evaluation/qa_dataset.json", "w") as f:
    json.dump(qa_pairs, f, indent=2)

print("✅ 50 question-answer pairs saved to evaluation/qa_dataset.json")
