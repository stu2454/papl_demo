import os, pathlib
import streamlit as st
import chromadb
import pandas as pd
st.set_page_config(page_title="PAPL Copilot — Demo", layout="wide")

# ---- Styles (dark-mode safe) ----
st.markdown("""
<style>
.block-container { max-width: 1200px; padding-top: .5rem; padding-bottom: 3rem; }
html, body, [class*='css'] { font-size: 18px !important; line-height: 1.6 !important; color: #222 !important; }
.answer-box { border: 1px solid #dfe3e8; background:#fff; border-radius:14px; padding:18px 20px; color:#222; }
.result-card { border: 1px solid #e6e6e6; border-radius: 14px; padding: 14px 16px; margin-bottom: 12px; background:#fff; color:#222; }
@media (prefers-color-scheme: dark) {
  html, body, [class*='css'] { color: #eaeef2 !important; }
  .answer-box, .result-card { background:#121417; border-color:#2a2f36; color:#eaeef2; }
}
</style>
""", unsafe_allow_html=True)


# ---- OpenAI client (secrets + env; v1 and legacy 0.x) ----
import os
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or (getattr(st, "secrets", {}).get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
OPENAI_MODE = None  # "v1" | "v0" | None
oai_client = None
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
            OPENAI_MODE = None



# Config
CFG = {
    "persist_dir": "data/chroma",
    "collection_name": "papl_chunks",
    "default_version": "2025-26",
    "top_k": 12,
    "ctx_k": 6,
}

# Optional OpenAI
OPENAI = None
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    try:
        from openai import OpenAI
        OPENAI = OpenAI()
    except Exception as e:
        st.warning(f"OpenAI SDK not available: {e}")

SYSTEM_PROMPT = open("prompts/system_papl.txt", "r", encoding="utf-8").read()

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CFG["persist_dir"])
    return client.get_collection(CFG["collection_name"])

col = get_collection()

def retrieve(query, version, top_k=12):
    res = col.query(query_texts=[query], n_results=top_k, where={"papl_version": version})
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] or res.get("embeddings", [[]])[0]
    rows = []
    for i, (d, m) in enumerate(zip(docs, metas)):
        rows.append({
            "rank": i+1,
            "score": dists[i] if dists else None,
            "preview": (d[:240] + "…") if len(d) > 240 else d,
            "page": m.get("page"),
            "section": m.get("section_title",""),
            "clause_ref": m.get("clause_ref",""),
            "papl_version": m.get("papl_version",""),
            "pdf": m.get("source_pdf_path",""),
            "full_text": d,
            "_meta": m
        })
    return rows

def answer_with_llm(question, ctx_blocks):
    if not OPENAI:
        # Sovereign/local mode: show sources only
        return None
    context_text = "\n\n".join(
        f"[Source: {m.get('papl_version','?')} {m.get('clause_ref','')} p.{m.get('page','?')}] {t}"
        for (t,m) in ctx_blocks
    )
    user = f"Question: {question}\n\nCONTEXT:\n{context_text}\n\nAnswer briefly with citations."
    chat = OPENAI.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":user}],
        temperature=0
    )
    return chat.choices[0].message.content

st.title("NDIS PAPL — Interactive Q&A (Demo)")
st.caption("Non-authoritative prototype. Verify in the official PAPL before use.")

version = st.sidebar.selectbox("PAPL version", [CFG["default_version"]], index=0)
q = st.text_input("Ask a question (e.g., “What’s the cancellation policy for therapy?”)")

if q:
    rows = retrieve(q, version, top_k=CFG["top_k"])
    if not rows:
        st.warning("No relevant passages found. Try a different query or check that the index was built.")
    else:
        ctx_blocks = [(r["full_text"], r["_meta"]) for r in rows[:CFG["ctx_k"]]]
        ans = answer_with_llm(q, ctx_blocks)
        if ans:
            st.markdown(ans)
        else:
            st.info("Local mode (no API key set): showing top sources only.")
        with st.expander("Sources"):
            for r in rows[:CFG["ctx_k"]]:
                link = f"{r['pdf']}#page={r['page']}" if r["pdf"] else ""
                cite = f"(PAPL {r['papl_version']}, p.{r['page']}" + (f", {r['clause_ref']}" if r["clause_ref"] else "") + ")"
                st.markdown(f"- **{r['section'] or 'Untitled section'}** {cite}  " + (f"[Open PDF]({link})" if link else ""))
                st.write(r["preview"])
        # show table for debugging
        with st.expander("Diagnostics (top matches)"):
            st.dataframe(pd.DataFrame(rows)[["rank","score","page","section","clause_ref"]])
else:
    st.write("Type a question above, then press Enter.")

st.divider()
st.caption("Index path: data/chroma • Collection: papl_chunks • Port: 8520")
# ---- Cloud-safe Chroma + ingestion helpers (appended) ----
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader

def split_chunks(text: str, chunk_chars=1800, overlap=220):
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        window = text[start:end]
        cut = window.rfind(". ")
        if cut == -1 or cut < chunk_chars * 0.6:
            cut = len(window)
        piece = window[:cut].strip()
        if piece: chunks.append(piece)
        if end == n: break
        start = max(0, start + cut - overlap)
    return chunks

@st.cache_resource
def get_collection_cloud():
    import chromadb
    persist_dir = CFG.get("persist_dir","data/chroma") if isinstance(CFG, dict) else "data/chroma"
    name = CFG.get("collection_name","papl_chunks") if isinstance(CFG, dict) else "papl_chunks"
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name)

# Override any prior 'col' with the cloud-safe one
try:
    col = get_collection_cloud()
except Exception as _e:
    # fallback: keep existing col if defined
    pass

def ingest_now(pdf_path: str=None, version: str=None, persist_dir: str=None, collection_name: str=None):
    pdf_path = pdf_path or (CFG.get("pdf_path") if isinstance(CFG, dict) else "data/NDIS_PAPL_2025-26.pdf")
    version = version or (CFG.get("default_version") if isinstance(CFG, dict) else "2025-26")
    persist_dir = persist_dir or (CFG.get("persist_dir") if isinstance(CFG, dict) else "data/chroma")
    collection_name = collection_name or (CFG.get("collection_name") if isinstance(CFG, dict) else "papl_chunks")
    if not OPENAI_KEY:
        st.error("OPENAI_API_KEY missing. Add it in Streamlit Cloud → Settings → Secrets."); return False
    if not os.path.exists(pdf_path):
        st.error(f"PDF not found at {pdf_path}. Commit it to the repo."); return False
    ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_KEY, model_name="text-embedding-3-small")
    reader = PdfReader(pdf_path)
    ids, docs, metas = [], [], []
    doc_id = 0
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        txt = " ".join(raw.split())
        if not txt: continue
        for j, piece in enumerate(split_chunks(txt), start=1):
            meta = {"papl_version": version, "page": i+1, "section_title": "", "clause_ref": "", "source_pdf_path": pdf_path}
            ids.append(f"p{i+1}_c{j}_{doc_id}"); docs.append(piece); metas.append(meta); doc_id += 1
    if not ids:
        st.error("No text could be extracted from the PDF."); return False
    client = chromadb.PersistentClient(path=persist_dir)
    coll = client.get_or_create_collection(collection_name, embedding_function=ef)
    for k in range(0, len(ids), 256):
        coll.upsert(ids=ids[k:k+256], documents=docs[k:k+256], metadatas=metas[k:k+256])
    st.success(f"Ingested {len(ids)} chunks into collection '{collection_name}'.")
    return True

# Guard: if index empty, offer one-click build (place this before main query UI)
try:
    _peek = col.peek()
    _empty_index = (not _peek) or (not _peek.get("ids"))
except Exception:
    _empty_index = True

if _empty_index and st.button("Build index now"):
    if ingest_now():
        st.experimental_rerun()
