#!/usr/bin/env python
import argparse, json, pathlib, yaml, sys
import chromadb
from chromadb.utils import embedding_functions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--jsonl")
    ap.add_argument("--openai", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    persist_dir = pathlib.Path(cfg.get("persist_dir", "data/chroma"))
    coll_name = cfg.get("collection_name", "papl_chunks")

    if args.jsonl:
        jsonl = pathlib.Path(args.jsonl)
    else:
        jsonl = pathlib.Path("data") / f"papl_chunks_{cfg['papl_version'].replace('/','-')}.jsonl"

    client = chromadb.PersistentClient(path=str(persist_dir))

    if args.openai:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY not set", file=sys.stderr); sys.exit(1)
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name="text-embedding-3-small"
        )
        col = client.get_or_create_collection(coll_name, embedding_function=ef)
    else:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            def _embed(texts): 
                return model.encode(texts, normalize_embeddings=True).tolist()
            col = client.get_or_create_collection(coll_name, embedding_function=_embed)
        except Exception as e:
            print("WARNING: sentence-transformers not available, using default embeddings:", e, file=sys.stderr)
            col = client.get_or_create_collection(coll_name)

    ids, docs, metas = [], [], []
    with open(jsonl, "r", encoding="utf-8") as r:
        for line in r:
            rec = json.loads(line)
            ids.append(rec["id"]); docs.append(rec["text"]); metas.append(rec["metadata"])

    for i in range(0, len(ids), 256):
        col.upsert(ids=ids[i:i+256], documents=docs[i:i+256], metadatas=metas[i:i+256])

    print(f"Ingested {len(ids)} chunks into '{coll_name}' at {persist_dir}")

if __name__ == "__main__":
    main()
