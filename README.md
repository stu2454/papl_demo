# PAPL Copilot — Streamlit Q&A Demo

This project demonstrates how the **NDIS Pricing Arrangements and Price Limits (PAPL)** can be transformed from a static PDF into a **searchable, responsive digital assistant**.  
It uses **Streamlit** for the UI, **ChromaDB** for retrieval, and (optionally) **OpenAI GPT models** for grounded answers.

---

## ✨ Features

- **Modern UI**: Clean app bar, sticky search toolbar, dark-mode safe styling.
- **Free-text Q&A**: Ask questions like *“Explain claiming for support items that have a price limit”*.
- **Grounded answers**: Outputs short, cited text with references to PAPL pages/clauses.
- **Sources grid**: See the retrieved passages side by side, with “Open PDF” links.
- **Sovereign mode**: Works offline with only local embeddings (no API key needed).
- **Cloud-ready**: Containerised with Docker, deployable to GitHub Codespaces, Render, or Streamlit Cloud.

---

## 📂 Project Layout

```
.
├─ app/
│  └─ streamlit_app.py       # Main UI (drop-in v3 with contemporary styling)
├─ scripts/
│  ├─ chunk_pdf.py           # Chunk the PAPL PDF into overlapping text blocks
│  └─ ingest_papl.py         # Ingest chunks into a Chroma index
├─ data/
│  ├─ NDIS_PAPL_2025-26.pdf  # (add this yourself, from ndis.gov.au)
│  └─ chroma/                # Persistent vector index (auto-generated)
├─ prompts/
│  └─ system_papl.txt        # Strict system prompt (UK English, AUD$, citations)
├─ config.yaml               # Chunking + index config
├─ requirements.txt          # Python dependencies
├─ Dockerfile                # Build the container
├─ docker-compose.yml        # Run ingestion + app locally
└─ README.md                 # This file
```

---

## 🚀 Quick Start (Docker)

### 1. Clone the repo
```bash
git clone https://github.com/your-org/papl-copilot.git
cd papl-copilot
```

### 2. Add the PAPL PDF
Download the official PAPL (e.g., **NDIS Pricing Arrangements and Price Limits 2025-26**) from  
[NDIS Pricing Arrangements](https://www.ndis.gov.au/providers/pricing-arrangements)  
and save it as:

```
data/NDIS_PAPL_2025-26.pdf
```

### 3. Build the image
```bash
docker compose build
```

### 4. Ingest the PDF into Chroma
```bash
docker compose run --rm ingest
```
This chunks the PDF and writes a persistent index into `data/chroma/`.

### 5. Run the app
```bash
# Optional: export OPENAI_API_KEY=sk-...   # enables GPT answers
docker compose up app
```

Open [http://localhost:8520](http://localhost:8520).

---

## 🔑 API Key (Optional)

- If no key is set → **Local/Sovereign mode** (retrieval only, shows top sources).
- If `OPENAI_API_KEY` is set → uses **`gpt-4o-mini`** to generate grounded answers.

### Local setup
Provide the key in `.env` (ignored by Git):
```
OPENAI_API_KEY=sk-...
```
Make sure `.env` is listed in your `.gitignore`.

### Streamlit Cloud setup
1. Deploy your repo to [Streamlit Cloud](https://streamlit.io/cloud).  
2. In your app dashboard, go to **Settings → Secrets**.  
3. Add:
   ```toml
   OPENAI_API_KEY="sk-..."
   ```
4. The app will pick it up automatically (via `os.getenv`).  
No need to commit `.env` — keep that local only.

---

## 🛠️ Development (local without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/chunk_pdf.py --config config.yaml
python scripts/ingest_papl.py --config config.yaml
streamlit run app/streamlit_app.py --server.port=8520
```

---

## ⚖️ Notes

- **Non-authoritative**: Always verify results against the official PAPL.
- **Versioning**: Update `config.yaml` and re-ingest whenever NDIA publishes a new PAPL version.
- **UI tweaks**: Fonts, colours, and layout can be tuned in the CSS block at the top of `streamlit_app.py`.
- **Future options**: Replace Streamlit with a React + FastAPI stack for even more polished UI.

---

## 📜 License

MIT (adjust as needed).
