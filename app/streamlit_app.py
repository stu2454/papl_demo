import os
import chromadb
import pandas as pd
import streamlit as st

# -----------------------------
# Page config MUST be first st.* call
# -----------------------------
st.set_page_config(page_title="PAPL Copilot — Demo", layout="wide")

# -----------------------------
# Config
# -----------------------------
CFG = {
    "persist_dir": "data/chroma",
    "collection_name": "papl_chunks",
    "default_version": "2025-26",
    "top_k": 12,
    "ctx_k": 6,
    "max_width_px": 1280,
}

# -----------------------------
# Styles (dark-mode safe, minimal chrome)
# -----------------------------
st.markdown(f"""
<style>
/* widen content */
.block-container {{
  max-width: {CFG["max_width_px"]}px;
  padding-top: 0.5rem;
  padding-bottom: 3rem;
}}

/* base typography */
html, body, [class*="css"] {{
  font-size: 20px !important;
  line-height: 1.6 !important;
  color: #222 !important;
}}

/* --- App Bar --- */
.appbar {{
  position: sticky;
  top: 0;
  z-index: 1000;
  background: rgba(255,255,255,0.85);
  backdrop-filter: saturate(180%) blur(8px);
  -webkit-backdrop-filter: saturate(180%) blur(8px);
  border-bottom: 1px solid #eaeaea;
}}
.appbar-inner {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 0 10px 0;
}}
.appbar-title {{
  font-weight: 800;
  font-size: 30px;
  margin: 0;
}}
.appbar-sub {{
  font-size: 14px;
  color: #6f7782;
  margin-top: -4px;
}}

/* --- Toolbar (search + filters) --- */
.toolbar {{
  display: grid;
  grid-template-columns: 1fr 220px 220px 120px;
  gap: 12px;
  padding: 12px 0 14px 0;
  border-bottom: 1px solid #f0f0f0;
}}
.stTextInput input {{
  height: 3.4rem;
  font-size: 20px;
  padding: 0.4rem 0.8rem;
}}
/* smaller field labels to reduce visual noise */
label, .stSelectbox label {{
  font-size: 14px !important;
}}

/* Cards & answer box */
.result-card {{
  border: 1px solid #e6e6e6;
  border-radius: 14px;
  padding: 14px 16px;
  margin-bottom: 12px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  background: #fff;
  color: #222;
}}
.answer-box {{
  border: 1px solid #dfe3e8;
  background: #fff;
  border-radius: 14px;
  padding: 20px 22px;
  font-size: 20px;
  color: #222;
}}
.result-head {{
  display:flex; gap:.5rem; align-items:center; margin-bottom:.25rem;
}}
.cite-chip {{
  display:inline-block; font-size: 14px; padding: 2px 8px; border-radius: 999px;
  background:#f4f6f8; border:1px solid #e6e6e6; margin-right:6px;
}}
.small-muted {{ color:#6f7782; font-size: 14px; }}
.pdf-link a {{ text-decoration:none; border-bottom: 1px dashed #999; }}

/* Dark-mode safety */
@media (prefers-color-scheme: dark) {{
  .appbar {{ background: rgba(16,16,16,0.6); border-bottom-color: #2a2a2a; }}
  .appbar-title {{ color: #f5f7fa; }}
  .appbar-sub {{ color: #9aa1a9; }}
  .toolbar {{ border-bottom-color: #2a2a2a; }}
  html, body, [class*="css"] {{ color: #eaeef2 !important; }}
  .result-card, .answer-box {{ background: #121417; border-color: #2a2f36; color: #eaeef2; }}
  .cite-chip {{ background:#1b1f24; border-color:#2a2f36; }}
  .pdf-link a {{ border-bottom-color:#8a98a8; }}
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# System prompt
# -----------------------------
SYSTEM_PROMPT = """You are a careful assistant answering questions about the NDIS Pricing Arrangements and Price Limits (PAPL).
Rules:
1) Answer ONLY using the supplied CONTEXT passages.
2) If the answer is not explicitly supported by the CONTEXT, reply exactly: "I can’t find that in the PAPL context provided."
3) Always include citations that reference the PAPL version and page numbers; include clause references when available.
4) Keep answers concise, plain UK English, and use AUD$ where prices are quoted.
5) If the user asks for advice beyond the PAPL’s scope (e.g., clinical, legal, policy positions), respond: "Out of scope for PAPL. Please consult the official guidance."
Citations style: (PAPL {papl_version}, p.{page}, {clause_ref}) — omit {clause_ref} if empty.
"""

# -----------------------------
# OpenAI SDK compatibility
# -----------------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODE, oai_client = None, None
if OPENAI_KEY:
    try:
        from openai import OpenAI  # v1+
        oai_client = OpenAI()
        OPENAI_MODE = "v1"
    except Exception:
        try:
            import openai as _openai  # legacy
            _openai.api_key = OPENAI_KEY
            oai_client = _openai
            OPENAI_MODE = "v0"
        except Exception as e:
            st.warning(f"OpenAI SDK not available: {e}")
            oai_client = None

# -----------------------------
# Chroma collection (cached)
# -----------------------------
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CFG["persist_dir"])
    return client.get_collection(CFG["collection_name"])
col = get_collection()

def retrieve(query: str, version: str, top_k: int = 12):
    res = col.query(query_texts=[query], n_results=top_k, where={"papl_version": version})
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] or res.get("embeddings", [[]])[0]
    rows = []
    for i, (d, m) in enumerate(zip(docs, metas)):
        rows.append({
            "rank": i + 1,
            "score": dists[i] if dists else None,
            "preview": (d[:380] + "…") if len(d) > 380 else d,
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
    if not oai_client:
        return None
    context_text = "\n\n".join(
        f"[Source: {m.get('papl_version','?')} {m.get('clause_ref','')} p.{m.get('page','?')}] {t}"
        for (t, m) in ctx_blocks
    )
    user = f"Question: {question}\n\nCONTEXT:\n{context_text}\n\nAnswer briefly with citations."
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

# -----------------------------
# App Bar + Toolbar
# -----------------------------
st.markdown('<div class="appbar">', unsafe_allow_html=True)
st.markdown('<div class="appbar-inner">', unsafe_allow_html=True)
st.markdown('<div><div class="appbar-title">NDIS PAPL — Interactive Q&A</div><div class="appbar-sub">Non‑authoritative prototype. Verify in the official PAPL before use.</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# toolbar
st.markdown('<div class="toolbar">', unsafe_allow_html=True)
q = st.text_input("Ask a question", placeholder="Type your question and press Enter…")
version = st.selectbox("PAPL version", [CFG["default_version"]], index=0)
category = st.selectbox("Category (optional)", ["All", "Core", "Capacity Building", "Capital"], index=0)
search_clicked = st.button("Search", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # end appbar

st.write("")

# -----------------------------
# Retrieval + Answer + Sources
# -----------------------------
if q or search_clicked:
    rows = retrieve(q, version, top_k=CFG["top_k"])
    if not rows:
        st.warning("No relevant passages found. Try refining your question.")
    else:
        ctx_blocks = [(r["full_text"], r["_meta"]) for r in rows[:CFG["ctx_k"]]]
        ans = answer_with_llm(q, ctx_blocks)

        st.markdown("### Answer")
        if ans:
            st.markdown(f'<div class="answer-box">{ans}</div>', unsafe_allow_html=True)
        else:
            st.info("Local mode (no API key set): showing top sources only.")

        st.markdown("### Sources")
        colA, colB = st.columns(2)
        for i, r in enumerate(rows[:CFG["ctx_k"]]):
            link = f"{r['pdf']}#page={r['page']}" if r["pdf"] else ""
            cite = f"(PAPL {r['papl_version']}, p.{r['page']}" + (f", {r['clause_ref']}" if r["clause_ref"] else "") + ")"
            card_html = f"""
            <div class="result-card">
              <div class="result-head">
                <span class="cite-chip">#{i+1}</span>
                <strong>{r['section'] or 'Untitled section'}</strong>
              </div>
              <div class="small-muted">{cite} &nbsp; <span class="pdf-link">{f'<a href="{link}" target="_blank">Open PDF</a>' if link else ''}</span></div>
              <div style="margin-top:.5rem;">{r['preview']}</div>
            </div>
            """
            (colA if i % 2 == 0 else colB).markdown(card_html, unsafe_allow_html=True)

        with st.expander("Diagnostics (top matches)"):
            st.dataframe(pd.DataFrame(rows)[["rank","score","page","section","clause_ref"]])

st.divider()
mode = "OpenAI (gpt-4o-mini)" if oai_client else "Local/Sovereign (no API key)"
st.caption(f"Mode: {mode} • Index path: {CFG['persist_dir']} • Collection: {CFG['collection_name']} • Port: 8520")
