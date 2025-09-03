import os, pathlib
import streamlit as st
import chromadb
import pandas as pd

st.set_page_config(page_title="PAPL Copilot — Demo", layout="wide")



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