#!/usr/bin/env python
import argparse, pathlib, yaml, json, re, csv
from PyPDF2 import PdfReader

def normalise_ws(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_chunks(text: str, chunk_chars: int, overlap: int):
    if chunk_chars <= 0:
        return [text]
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
        if end == n: break
        start = max(0, start + cut - overlap)
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    pdf_path = pathlib.Path(cfg["pdf_path"])
    out_path = pathlib.Path("data") / f"papl_chunks_{cfg['papl_version'].replace('/','-')}.jsonl"
    chunk_chars = int(cfg.get("chunk_chars", 1800))
    overlap = int(cfg.get("chunk_overlap", 220))
    max_chunks = int(cfg.get("max_chunks", 0))

    section_map_csv = cfg.get("section_map_csv") or ""
    ranges = []
    if section_map_csv:
        with open(section_map_csv, newline="") as f:
            for row in csv.DictReader(f):
                ranges.append((int(row["start_page"]), int(row["end_page"]), row["section_title"]))

    def page_section(p):
        for s, e, title in ranges:
            if s <= p <= e:
                return title
        return ""

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    with open(out_path, "w", encoding="utf-8") as w:
        doc_id = 0
        for i in range(total_pages):
            raw = reader.pages[i].extract_text() or ""
            txt = normalise_ws(raw)
            if not txt:
                continue
            for j, piece in enumerate(split_chunks(txt, chunk_chars, overlap), start=1):
                meta = {"papl_version": cfg["papl_version"], "page": i+1,
                        "section_title": page_section(i+1), "clause_ref": "",
                        "source_pdf_path": str(pdf_path).replace("\\","/")}
                rec = {"id": f"p{i+1}_c{j}_{doc_id}", "text": piece, "metadata": meta}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                doc_id += 1
                if max_chunks and doc_id >= max_chunks:
                    break
            if max_chunks and doc_id >= max_chunks: break
    print(f"Wrote chunks to {out_path}")

if __name__ == "__main__":
    main()
