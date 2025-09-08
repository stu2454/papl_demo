# NDIS PAPL Copilot — Streamlit Demo

A prototype tool that transforms the static **NDIS Pricing Arrangements and Price Limits (PAPL)** PDF into an **accessible, searchable, and intelligent Q&A assistant**.  
Built with [Streamlit](https://streamlit.io/), [ChromaDB](https://www.trychroma.com/), and [OpenAI](https://platform.openai.com/).

---

## Background & Rationale

The *NDIS Pricing Arrangements and Price Limits (PAPL)* is the central reference for providers, participants, and planners. It sets out the rules, item descriptions, and price caps that govern how supports are funded under the Scheme. While essential, the PAPL is a large, technical, and static PDF document. Users often struggle to locate relevant clauses, interpret pricing limits, and connect information across sections.  

This creates several challenges:

- **Accessibility** – Not all users have the time or expertise to scan through hundreds of pages to find a single rule. People with disability, families, and smaller providers can be disadvantaged if they cannot easily locate key information.  
- **Responsiveness** – Planners and policy staff often need to answer time-sensitive questions. A static PDF slows down their work and increases the risk of inconsistent interpretations.  
- **System integrity** – When answers are difficult to find, reliance on informal knowledge, guesswork, or out-of-date advice increases. This undermines consistent decision-making and erodes trust in the Scheme’s governance.  

Transforming the PAPL into a **searchable, interactive, and intelligent tool** addresses these challenges directly. By layering natural-language search and AI-driven question answering on top of the official document:

- Users can **ask questions in plain English** and receive concise, cited answers linked to the authoritative PAPL text.  
- The system supports **faster, more consistent policy application**, reducing variation in decision-making.  
- Accessibility is improved, making the PAPL more usable to participants and providers, not just specialists.  
- Policy staff gain a **living tool** that can evolve toward richer analytics (e.g., usage patterns, frequently asked questions) to inform future PAPL revisions.  

In short, this project demonstrates how a static regulatory document can be re-imagined as an accessible, responsive, and trusted digital service — improving equity, efficiency, and system learning across the NDIS.

---

## Features

- Uploads and ingests the **PAPL PDF** into a local vector database.  
- Supports **plain English queries** with semantic search across PAPL clauses.  
- Uses an **LLM** to generate concise answers with citations back to the source.  
- Provides a **“Build index now”** button to re-ingest the document.  
- Runs fully containerised with **Docker** for reproducibility.  
- Designed for deployment to **Streamlit Cloud** or local testing.

---

## Requirements

- Python 3.11  
- Dependencies (see `requirements.txt`):
  - `streamlit`
  - `chromadb==0.4.22`
  - `duckdb`
  - `pysqlite3-binary`
  - `openai`
  - `PyPDF2`
  - `pandas`, `numpy`, `PyYAML`

---

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/your-org/papl_demo.git
   cd papl_demo
   ```

2. Add the PAPL PDF:
   ```bash
   mkdir -p data
   cp NDIS_PAPL_2025-26.pdf data/
   ```

3. Create a `.env` file for **local development**:
   ```bash
   OPENAI_API_KEY=sk-your-real-key
   ```

4. For **Streamlit Cloud deployment**, set secrets in **Settings → Secrets**:
   ```toml
   OPENAI_API_KEY="sk-your-real-key"
   CHROMA_DIR="/mount/data/chroma"
   ```

5. Build and run locally:
   ```bash
   docker compose up --build
   ```

6. Open [http://localhost:8520](http://localhost:8520) in your browser.

---

## Usage

- On first load, click **Build index now** to ingest the PAPL PDF.  
- Enter a question in the text box (e.g. *“What is the price limit for low-cost assistive technology?”*).  
- The tool will return:
  - A concise answer (if an API key is configured).  
  - A list of relevant source passages with page numbers.  

---

## Limitations

- Prototype only: **not an authoritative source**. Always confirm against the official PAPL PDF.  
- Accuracy depends on the quality of PDF text extraction and embeddings.  
- Context length is limited; long or complex queries may truncate context.  

---

## License

MIT — see [LICENSE](LICENSE).
