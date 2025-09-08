# app/streamlit_app.py — Cloud-stable drop-in
import os, sys
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader

# ---- Disable file watcher (Cloud inotify fix) ----
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"

# ---- sqlite3 >=3.35 shim BEFORE importing chromadb ----
try:
    import pysqlite3  # provided by pysqlite3-binary
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import chromadb
from chromadb.utils import embedding_functions

st.set_page_config(page_title="PAPL Copilot — Cloud Demo", layout="wide")

def pick_writable_dir(candidates):
    for d in candidates:
        if not d:
            continue
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".write_test")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            return d
        except Exception:
            continue
    raise RuntimeError("No writable directory found for Chroma persistence. Tried: " + ", ".join([c for c in candidates if c]))

# Prefer CHROMA_DIR (if set), then /mount/data/chroma (persistent on Cloud), then /tmp/chroma
PERSIST_DIR = pick_writable_dir([os.environ.get("CHROMA_DIR", ""), "/mount/data/chroma", "/tmp/chroma"])

CFG = {
    "persist_dir": PERSIST_DIR,
    "collection_name": "papl_chunks",
    "default_version": "2025-26",
    "pdf_path": "data/NDIS_PAPL_2025-26.pdf",
    "top_k": 12,
    "ctx_k": 6,
    "max_width_px": 1200,
}

# ---- Styles ----
st.markdown(f"""
<style>
.block-container {{ max-width: {CFG["max_width_px"]}px; padding-top: .5rem; padding-bottom: 3rem; }}
html, body, [class*='css'] {{ font-size: 18px !important; line-height: 1.6 !important; color: #222 !important; }}
.answer-box {{ border: 1px solid #dfe3e8; background:#fff; border-radius:14px; padding:18px 20px; color:#222; }}
.result-card {{ border: 1px solid #e6e6e6; border-radius:14px; padding:14px 16px; margin-bottom:12px; background:#fff; color:#222; }}
@media (prefers-color-scheme: dark) {{
  html, body, [class*='css'] {{ color: #eaeef2 !important; }}
  .answer-box, .result-card {{ background:#121417; border-color:#2a2f36; color:#eaeef2; }}
}}
</style>
""", unsafe_allow_html=True)

# ---- OpenAI client (secrets + env; v1 and legacy 0.x) ----
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or (getattr(st, "secrets", {}).get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
OPENAI_MODE, oai_client = None, None
if OPENAI_KEY:
    try:
        from openai import OpenAI
        oai_client = OpenAI()
        OPENAI_MODE = "v1"
    except Exception:
        try:
            import openai as _openai
            _openai.api_key = OPENAI_KEY
            oai_client = _openai
            OPENAI_MODE = "v0"
        except Exception as e:
            st.warning(f"OpenAI SDK not available: {e}")
            oai_client = None

SYSTEM_PROMPT = """You are a careful assistant answering questions about the NDIS Pricing Arrangements and Price Limits (PAPL).
Rules:
1) Answer ONLY using the supplied CONTEXT passages.
2) If the answer is not explicitly supported by the CONTEXT, reply exactly: "I can’t find that in the PAPL context provided."
3) Always include citations that reference the PAPL version and page numbers; include clause references when available.
4) Keep answers concise, plain UK English, and use AUD$ where prices are quoted.
5) If the user asks for advice beyond the PAPL’s scope (e.g., clinical, legal, policy positions), respond: "Out of scope for PAPL. Please consult the official guidance."
"""

# ---- Chroma PersistentClient (0.4.x) ----
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CFG["persist_dir"])
    return client.get_or_create_collection(CFG["collection_name"])

col = get_collection()

# ---- Utils ----
def split_chunks(text: str, chunk_chars=1800, overlap=220):
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        window = text[start:end]
        cut = window.rfind(". ")
        if cut == -1 or cut < chunk_chars * 0.6:
            cut = len(window)
        piece = window[:cut].strip()
        if piece:
            chunks.append(piece)
        if end == n:
            break
        start = max(0, start + cut - overlap)
    return chunks

def ingest_now():
    if not OPENAI_KEY:
        st.error("OPENAI_API_KEY missing. Add it in Streamlit Cloud → Settings → Secrets.")
        return False
    if not os.path.exists(CFG["pdf_path"]):
        st.error(f"PDF not found at {CFG['pdf_path']}. Commit it to the repo.")
        return False

    ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_KEY, model_name="text-embedding-3-small")
    reader = PdfReader(CFG["pdf_path"])
    ids, docs, metas = [], [], []
    doc_id = 0
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        txt = " ".join(raw.split())
        if not txt:
            continue
        for j, piece in enumerate(split_chunks(txt), start=1):
            meta = {"papl_version": CFG["default_version"], "page": i+1,
                    "section_title": "", "clause_ref": "", "source_pdf_path": CFG["pdf_path"]}
            ids.append(f"p{i+1}_c{j}_{doc_id}")
            docs.append(piece)
            metas.append(meta)
            doc_id += 1
    if not ids:
        st.error("No text could be extracted from the PDF.")
        return False
    for k in range(0, len(ids), 256):
        col.upsert(ids=ids[k:k+256], documents=docs[k:k+256], metadatas=metas[k:k+256])
    st.success(f"Ingested {len(ids)} chunks into collection '{CFG['collection_name']}'.")
    return True

def retrieve(query: str, version: str, top_k: int = 12):
    res = col.query(query_texts=[query], n_results=top_k, where={"papl_version": version})
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] or []
    rows = []
    for i, (d, m) in enumerate(zip(docs, metas)):
        rows.append({
            "rank": i + 1,
            "score": dists[i] if i < len(dists) else None,
            "preview": (d[:360] + "…") if len(d) > 360 else d,
            "page": m.get("page"),
            "section": m.get("section_title", ""),
            "clause_ref": m.get("clause_ref", ""),
            "papl_version": m.get("papl_version", ""),
            "pdf": m.get("source_pdf_path", ""),
            "full_text": d,
            "_meta": m,
        })
    return rows

def answer_with_llm(question: str, ctx_blocks):
    if oai_client is None:
        return None
    context_text = "\n\n".join(
        f"[Source: {m.get('papl_version','?')} {m.get('clause_ref','')} p.{m.get('page','?')}] {t}"
        for (t, m) in ctx_blocks
    )
    user = f"Question: {question}\n\nCONTEXT:\n{context_text}\n\nAnswer briefly with citations."
    try:
        if OPENAI_MODE == "v1":
            resp = oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
                temperature=0,
            )
            return resp.choices[0].message.content
        resp = oai_client.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user}],
            temperature=0,
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        return None

# ---------------- UI ----------------
st.title("NDIS PAPL — Cloud Q&A Demo")
st.caption("Non-authoritative prototype. Verify in the official PAPL before use.")
st.markdown("""
### Welcome to the PAPL Copilot

This tool was developed to make the **NDIS Pricing Arrangements and Price Limits (PAPL)** easier to access and use.  
The official PAPL is a large, technical PDF document. While authoritative, it can be difficult to search and interpret quickly.  
This app allows you to:

- **Ask questions in plain English** and receive concise answers linked to the PAPL.
- **See the exact source passages** (with page references) that support the answer.
- **Rebuild the index** whenever a new version of the PAPL PDF is added.

---

#### How to use the app
1. If this is your first time running it, click **Build index now** to load the PAPL PDF into the search index.  
2. Enter a question in the box (e.g. *"What is the price limit for low-cost AT?"*).  
3. Read the answer and check the cited sources for confirmation.  

⚠️ **Note:** This is a prototype tool. It is not an official source of truth — always confirm important details in the official PAPL PDF.
""")
st.info(f"Chroma dir: {CFG['persist_dir']}")

# Index status
try:
    _probe = col.count() if hasattr(col, "count") else None
    _empty_index = (_probe == 0) if _probe is not None else False
except Exception:
    _empty_index = True

if _empty_index:
    st.warning("Vector index empty. Click **Build index now** to ingest the PAPL PDF.")
    if st.button("Build index now"):
        if ingest_now():
            st.rerun()  # <-- ONLY use st.rerun()

q = st.text_input("Ask a question", placeholder="Type your question and press Enter…")
if q:
    rows = retrieve(q, CFG["default_version"], top_k=CFG["top_k"])
    if not rows:
        st.warning("No relevant passages found.")
    else:
        ctx_blocks = [(r["full_text"], r["_meta"]) for r in rows[:CFG["ctx_k"]]]
        ans = answer_with_llm(q, ctx_blocks)
        st.markdown("### Answer")
        if ans:
            st.markdown(f'<div class="answer-box">{ans}</div>', unsafe_allow_html=True)
        else:
            st.info("Local mode (no API key set): showing top sources only.")
        st.markdown("### Sources")
        for i, r in enumerate(rows[:CFG["ctx_k"]]):  # simple list for Cloud
            st.markdown(f"- **p.{r['page']}** {r['preview']}")
