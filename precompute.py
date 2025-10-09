# precompute.py
"""
Crawl EvalAI documentation and precompute FAISS embeddings.

This script:
1. Crawls all documentation pages from EvalAI docs
2. Extracts clean text from HTML
3. Splits into semantic chunks
4. Embeds using Sentence Transformers
5. Saves FAISS index + metadata inside /db

Run locally before deploying to Hugging Face Space:
    python precompute.py
"""

import os
import time
import pickle
import requests
from urllib.parse import urljoin, urlparse, urldefrag
from bs4 import BeautifulSoup
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


# -------------------------------
# Config
# -------------------------------
BASE = "https://evalai.readthedocs.io/en/latest/"
MAX_PAGES = 1200
DELAY = 0.1
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAVE_DIR = os.path.join(os.getcwd(), "db")

SEEN, PAGES = set(), []


# -------------------------------
# Helpers
# -------------------------------
def is_doc_url(u: str) -> bool:
    parsed = urlparse(u)
    if not u.startswith(BASE):
        return False
    if not (u.endswith(".html") or u.endswith("/")):
        return False
    if any(x in u for x in ("/_static/", "/_images/", "genindex.html", "search.html")):
        return False
    return True


def clean_url(u: str) -> str:
    u, _ = urldefrag(u)
    if u.endswith("/"):
        u += "index.html"
    return u


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for bad in soup.select("nav, footer, .toc, .sphinxsidebar, .wy-nav-side, .related"):
        bad.decompose()
    main = soup.select_one("div[role=main]") or soup
    parts = []
    for sel in ["h1", "h2", "h3", "h4", "p", "li", "pre", "code", "td", "th"]:
        for node in main.select(sel):
            txt = " ".join(node.get_text(" ", strip=True).split())
            if txt:
                parts.append(f"<{sel.upper()}> {txt}")
    return "\n".join(parts)


def crawl(start_url: str, max_pages=500, delay=0.2):
    queue = [start_url]
    while queue and len(PAGES) < max_pages:
        url = clean_url(queue.pop(0))
        if url in SEEN:
            continue
        SEEN.add(url)
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
                continue
            text = extract_text(r.text)
            if text.strip():
                PAGES.append({"url": url, "text": text})
        except Exception as e:
            print("⚠️ Skipped:", url, repr(e))
            continue
        try:
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                candidate = urljoin(url, a["href"])
                if is_doc_url(candidate):
                    candidate = clean_url(candidate)
                    if candidate not in SEEN:
                        queue.append(candidate)
        except Exception:
            pass
        time.sleep(delay)


# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting EvalAI docs crawl...\n")
    crawl(BASE, max_pages=MAX_PAGES, delay=DELAY)
    print(f"✅ Collected {len(PAGES)} pages.\n")

    # --- Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=250)
    docs, metas = [], []
    for page in tqdm(PAGES, desc="Splitting pages"):
        chunks = splitter.split_text(page["text"])
        docs.extend(chunks)
        metas.extend([{"source": page["url"]}] * len(chunks))

    documents = [Document(page_content=chunk, metadata=meta) for chunk, meta in zip(docs, metas)]
    print(f"🧩 Total chunks created: {len(documents)}\n")

    # --- Embeddings
    print("🔢 Generating embeddings (this may take a few minutes)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # --- Create FAISS index
    vector_db = FAISS.from_documents(documents, embedding_model)

    # --- Ensure db folder exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Save index to /db
    vector_db.save_local(SAVE_DIR)

    # --- Cleanup misplaced files (edge case)
    for fname in ["index.faiss", "index.pkl", "index_to_docstore.json"]:
        src = os.path.join(os.getcwd(), fname)
        dst = os.path.join(SAVE_DIR, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            os.rename(src, dst)
            print(f"📦 Moved {fname} → db/{fname}")

    print("\n✅ Done! FAISS and metadata saved in ./db/")
    print("📂 Files generated:")
    for f in os.listdir(SAVE_DIR):
        print("  -", f)
    print("\n💾 Ready to push to Hugging Face Space!")
