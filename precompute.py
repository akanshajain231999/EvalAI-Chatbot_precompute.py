# precompute.py
import time, os
from urllib.parse import urljoin, urlparse, urldefrag
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm

BASE = "https://evalai.readthedocs.io/en/latest/"
SEEN, PAGES = set(), []

def is_doc_url(u):
    parsed = urlparse(u)
    if not u.startswith(BASE):
        return False
    if not (u.endswith(".html") or u.endswith("/")):
        return False
    if any(x in u for x in ("/_static/", "/_images/", "genindex.html", "search.html")):
        return False
    return True

def clean_url(u):
    u, _ = urldefrag(u)
    if u.endswith("/"): u = u + "index.html"
    return u

def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for bad in soup.select("nav, footer, .toc, .sphinxsidebar, .wy-nav-side, .related"):
        bad.decompose()
    main = soup.select_one("div[role=main]") or soup
    parts = []
    for sel in ["h1","h2","h3","h4","p","li","pre","code","td","th"]:
        for node in main.select(sel):
            txt = " ".join(node.get_text(" ", strip=True).split())
            if txt:
                parts.append(f"<{sel.upper()}> {txt}")
    return "\n".join(parts)

def crawl(start_url, max_pages=500, delay=0.2):
    queue = [start_url]
    while queue and len(PAGES) < max_pages:
        url = clean_url(queue.pop(0))
        if url in SEEN:
            continue
        SEEN.add(url)
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type",""):
                continue
            text = extract_text(r.text)
            if text.strip():
                PAGES.append({"url": url, "text": text})
        except Exception as e:
            print("Skip:", url, repr(e))
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

if __name__ == "__main__":
    print("Crawling docs (this may take minutes)...")
    crawl(BASE, max_pages=1200, delay=0.1)
    print(f"Collected {len(PAGES)} pages")

    # chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=250)
    docs, metas = [], []
    for page in PAGES:
        chunks = splitter.split_text(page["text"])
        docs.extend(chunks)
        metas.extend([{"source": page["url"]}] * len(chunks))

    documents = [Document(page_content=chunk, metadata=meta) for chunk, meta in zip(docs, metas)]
    print("Total chunks:", len(documents))

    # Embeddings: CPU by default
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # recommended small CPU-friendly
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # --- Create FAISS index ---
    vector_db = FAISS.from_documents(documents, embedding_model)

    # --- Create folder explicitly ---
    save_dir = os.path.join(os.getcwd(), "db")
    os.makedirs(save_dir, exist_ok=True)

    # --- Save index to that folder ---
    vector_db.save_local(save_dir)

    # --- Move files manually if saved in wrong location ---
    for fname in ["index.faiss", "index.pkl", "index_to_docstore.json"]:
        src = os.path.join(os.getcwd(), fname)
        dst = os.path.join(save_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            os.rename(src, dst)
            print(f"Moved {fname} → db/{fname}")

    # --- Print verification summary ---
    print("\n✅ Saved FAISS index and metadata inside ./db/")
    for f in os.listdir(save_dir):
        print(" -", f)
