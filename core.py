import os
import re
import json
import requests
import tempfile
import numpy as np
import faiss
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Check your .env file or environment.")

genai.configure(api_key=GOOGLE_API_KEY)

# Load keywords for optional future logic (like auto-approval/denial)
APPROVE_KEYWORDS = []
DENY_KEYWORDS = []
try:
    with open("keywords.json", "r") as f:
        kws = json.load(f)
        APPROVE_KEYWORDS = [kw.lower() for kw in kws.get("approve_keywords", [])]
        DENY_KEYWORDS = [kw.lower() for kw in kws.get("deny_keywords", [])]
except Exception:
    pass  # Safe fallback if keywords file is missing

# Gemini model configurations
MODEL_EMBED = "models/embedding-001"
MODEL_CHAT = "gemini-2.0-flash-exp"

def download_and_extract_text(pdf_url):
    response = requests.get(pdf_url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    text = ""
    with pdfplumber.open(tmp_file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    os.remove(tmp_file_path)
    return text

def split_chunks(text, max_chars=1500, overlap=200):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    for para in paragraphs:
        if len(para) > max_chars:
            for i in range(0, len(para), max_chars - overlap):
                chunks.append(para[i:i + max_chars])
        elif len(para.strip()) > 100:
            chunks.append(para.strip())
    return chunks

def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        resp = genai.embed_content(
            model=MODEL_EMBED,
            content=chunk,
            task_type="retrieval_document"
        )
        embeddings.append(resp["embedding"])
    return np.array(embeddings).astype("float32")

def embed_query(query):
    resp = genai.embed_content(
        model=MODEL_EMBED,
        content=query,
        task_type="retrieval_query"
    )
    return np.array([resp["embedding"]]).astype("float32")

def search_chunks(query, chunks, embeddings, top_k=5):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    query_emb = embed_query(query)
    _, I = index.search(query_emb, top_k)
    return [chunks[i] for i in I[0]]

def ask_gemini(question, context_chunks):
    context = "\n\n".join([f"Clause {i + 1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
    prompt = f"""
You are an expert policy analyzer.

Given the policy clauses below, answer the question.

QUESTION: {question}

POLICY CLAUSES:
{context}

Only respond with a clear, complete sentence.
"""
    model = genai.GenerativeModel(MODEL_CHAT)
    response = model.generate_content(prompt)
    return response.text.strip()

def process_pdf_and_answer_questions(pdf_url, questions):
    full_text = download_and_extract_text(pdf_url)
    chunks = split_chunks(full_text)
    embeddings = embed_chunks(chunks)
    answers = []
    for q in questions:
        top_chunks = search_chunks(q, chunks, embeddings, top_k=4)
        answer = ask_gemini(q, top_chunks)
        answers.append(answer)
    return answers
