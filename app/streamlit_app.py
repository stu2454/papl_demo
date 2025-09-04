# PAPL Copilot — Cloud Demo (sqlite shim + Chroma 0.4.22)
# Drop-in for app/streamlit_app.py

import os, sys
import streamlit as st

# ---- sqlite3 >=3.35 shim (must be BEFORE chromadb import) ----
try:
    import pysqlite3  # provided by pysqlite3-binary in requirements
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except Exception:
    # If the module isn't present, Chroma may error at import time.
    # Ensure pysqlite3-binary is listed in requirements.txt on Streamlit Cloud.
    pass

import pandas as pd
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

st.set_page_config(page_title='PAPL Copilot — Cloud Demo', layout='wide')

CFG = {
    'persist_dir': os.environ.get('CHROMA_DIR', '/home/adminuser/chroma'),
    'collection_name': 'papl_chunks',
    'default_version': '2025-26',
    'pdf_path': 'data/NDIS_PAPL_2025-26.pdf',
    'top_k': 12,
    'ctx_k': 6,
    'max_width_px': 1200,
}

st.markdown(f"""
<style>
.block-container {{ max-width: {CFG['max_width_px']}px; padding-top: .5rem; padding-bottom: 3rem; }}
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
OPENAI_KEY = os.getenv('OPENAI_API_KEY') or (getattr(st, 'secrets', {}).get('OPENAI_API_KEY') if hasattr(st, 'secrets') else None)
OPENAI_MODE = None  # 'v1' | 'v0' | None
oai_client = None
if OPENAI_KEY:
    try:
        from openai import OpenAI
        oai_client = OpenAI()
        OPENAI_MODE = 'v1'
    except Exception:
        try:
            import openai as _openai
            _openai.api_key = OPENAI_KEY
            oai_client = _openai
            OPENAI_MODE = 'v0'
        except Exception as e:
            st.warning(f'OpenAI SDK not available: {e}')
            oai_client = None
            OPENAI_MODE = None

SYSTEM_PROMPT = """You are a careful assistant answering questions about the NDIS Pricing Arrangements and Price Limits (PAPL).
Rules:
1) Answer ONLY using the supplied CONTEXT passages.
2) If the answer is not explicitly supported by the CONTEXT, reply exactly: "I can’t find that in the PAPL context provided."
3) Always include citations that reference the PAPL version and page numbers; include clause references when available.
4) Keep answers concise, plain UK English, and use AUD$ where prices are quoted.
5) If the user asks for advice beyond the PAPL’s scope (e.g., clinical, legal, policy positions), respond: "Out of scope for PAPL. Please consult the official guidance."
"""

@st.cache_resource
def get_collection():
    os.makedirs(CFG['persist_dir'], exist_ok=True)
    client = chromadb.Client(Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=CFG['persist_dir'],
        anonymized_telemetry=False
    ))
    return client.get_or_create_collection(CFG['collection_name'])

col = get_collection()

def split_chunks(text: str, chunk_chars=1800, overlap=220):
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        window = text[start:end]
        cut = window.rfind('. ')
        if cut == -1 or cut < chunk_chars * 0.6:
            cut = len(window)
        piece = window[:cut].strip()
        if piece: chunks.append(piece)
        if end == n: break
        start = max(0, start + cut - overlap)
    return chunks

def ingest_now():
    if not OPENAI_KEY:
        st.error('OPENAI_API_KEY missing. Add it in Streamlit Cloud → Settings → Secrets.'); return False
    if not os.path.exists(CFG['pdf_path']):
        st.error(f"PDF not found at {CFG['pdf_path']}. Commit it to the repo."); return False

    ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_KEY, model_name='text-embedding-3-small')
    reader = PdfReader(CFG['pdf_path'])
    ids, docs, metas = [], [], []
    doc_id = 0
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ''
        txt = ' '.join(raw.split())
        if not txt: continue
        for j, piece in enumerate(split_chunks(txt), start=1):
            meta = {'papl_version': CFG['default_version'], 'page': i+1,
                    'section_title': '', 'clause_ref': '', 'source_pdf_path': CFG['pdf_path']}
            ids.append(f'p{i+1}_c{j}_{doc_id}'); docs.append(piece); metas.append(meta); doc_id += 1
    if not ids:
        st.error('No text could be extracted from the PDF.'); return False
    for k in range(0, len(ids), 256):
        col.upsert(ids=ids[k:k+256], documents=docs[k:k+256], metadatas=metas[k:k+256])
    st.success(f"Ingested {len(ids)} chunks into collection '{CFG['collection_name']}'.")
    return True

def retrieve(query: str, version: str, top_k: int = 12):
    res = col.query(query_texts=[query], n_results=top_k, where={'papl_version': version})
    docs = res.get('documents', [[]])[0]
    metas = res.get('metadatas', [[]])[0]
    dists = res.get('distances', [[]])[0] or []
    rows = []
    for i, (d, m) in enumerate(zip(docs, metas)):
        rows.append({
            'rank': i + 1,
            'score': dists[i] if i < len(dists) else None,
            'preview': (d[:360] + '…') if len(d) > 360 else d,
            'page': m.get('page'),
            'section': m.get('section_title', ''),
            'clause_ref': m.get('clause_ref', ''),
            'papl_version': m.get('papl_version', ''),
            'pdf': m.get('source_pdf_path', ''),
            'full_text': d,
            '_meta': m,
        })
    return rows

def answer_with_llm(question: str, ctx_blocks):
    if not oai_client: return None
    context_text = '\n\n'.join(
        f"[Source: {m.get('papl_version','?')} {m.get('clause_ref','')} p.{m.get('page','?')}] {t}"
        for (t, m) in ctx_blocks
    )
    user = f"Question: {question}\n\nCONTEXT:\n{context_text}\n\nAnswer briefly with citations."
    if OPENAI_MODE == 'v1':
        resp = oai_client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                      {'role': 'user', 'content': user}],
            temperature=0,
        )
        return resp.choices[0].message.content
    resp = oai_client.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                  {'role': 'user', 'content': user}],
        temperature=0,
    )
    return resp['choices'][0]['message']['content']

# ---------------- UI ----------------
st.title('NDIS PAPL — Cloud Q&A Demo')
st.caption('Non-authoritative prototype. Verify in the official PAPL before use.')

# index status
try:
    _peek = col.get(ids=['__healthcheck__'])
    _empty_index = not _peek or not _peek.get('ids')
except Exception:
    _empty_index = True

if _empty_index:
    st.warning('Vector index empty. Click **Build index now** to ingest the PAPL PDF.')
    if st.button('Build index now'):
        if ingest_now(): st.experimental_rerun()

q = st.text_input('Ask a question', placeholder='Type your question and press Enter…')
if q:
    rows = retrieve(q, CFG['default_version'], top_k=CFG['top_k'])
    if not rows:
        st.warning('No relevant passages found.')
    else:
        ctx_blocks = [(r['full_text'], r['_meta']) for r in rows[:CFG['ctx_k']]]
        ans = answer_with_llm(q, ctx_blocks)
        st.markdown('### Answer')
        if ans:
            st.markdown(f'<div class="answer-box">{ans}</div>', unsafe_allow_html=True)
        else:
            st.info('Local mode (no API key set): showing top sources only.')
        st.markdown('### Sources')
        for i, r in enumerate(rows[:CFG['ctx_k']]):
            st.markdown(f"- **p.{r['page']}** {r['preview']}")
