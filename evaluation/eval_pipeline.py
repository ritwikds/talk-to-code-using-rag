"""
eval_pipeline.py

Evaluates your Codebase RAG Assistant on the stored QA pairs.
Uses an LLM to score each answer on a scale of 1-10.
"""

import json
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import your actual assistant call here
from app import chat_with_codebase


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Load evaluation data
with open("evaluation/qa_dataset.json") as f:
    qa_pairs = json.load(f)

# LLM scorer setup
scoring_prompt = PromptTemplate.from_template("""
You are a strict evaluator.

Question:
{question}

Reference Answer:
{reference}

Candidate Answer:
{candidate}

Rate how well the candidate answer matches the reference answer.
Respond with just a number from 1 (terrible) to 10 (perfect):
""")
# LLM setup
scoring_llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=api_key, temperature=0.5)
scoring_chain = LLMChain(llm=scoring_llm, prompt=scoring_prompt)

# Run evaluation
results = []
for pair in tqdm(qa_pairs):
    q = pair["question"]
    gold = pair["answer"]

    # Call your RAG assistant
    generated = chat_with_codebase(q)

    # Score using LLM
    score_text = scoring_chain.run({"question": q, "reference": gold, "candidate": generated})
    try:
        score = int(score_text.strip())
    except:
        score = 5  # fallback default

    results.append({
        "question": q,
        "reference": gold,
        "candidate": generated,
        "score": score
    })

# Save results
with open("evaluation/results.json", "w") as f:
    json.dump(results, f, indent=2)

# Print average score
average = sum(r["score"] for r in results) / len(results)
print(f"✅ Evaluation complete. Average score: {average:.2f}/10")
