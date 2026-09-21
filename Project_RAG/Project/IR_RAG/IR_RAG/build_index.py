"""
build_index.py
──────────────
Reads Jumia_Real_Final_Clean.csv, creates a sentence-embedding for every
product, and saves a FAISS index + metadata file so rag.py can query them.

Run once:
    python build_index.py
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH      = "data/Jumia_Real_Final_Clean.csv"
INDEX_PATH    = "data/jumia.index"
METADATA_PATH = "data/jumia_meta.pkl"
MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"   # ~80 MB, CPU-fast
# ─────────────────────────────────────────────────────────────────────────────


def build_product_text(row: dict) -> str:
    """
    Combine fields into a single searchable sentence.
    This is what gets embedded – keep it rich so semantic search works well.
    """
    features = row.get("Key_Features", "")
    if features in ("Not Available", "", None):
        features = ""
    else:
        # Trim to avoid overly long embeddings
        features = features[:500]

    price = row.get("Price", "")
    try:
        price_fmt = f"{float(price):,.0f} EGP"
    except (ValueError, TypeError):
        price_fmt = str(price)

    text = (
        f"Product: {row.get('Name', '')}. "
        f"Brand: {row.get('Brand', '')}. "
        f"Category: {row.get('Category', '')}. "
        f"Price: {price_fmt}. "
    )
    if features:
        text += f"Features: {features}."
    return text.strip()


def main():
    print("📦  Loading CSV …")
    df = pd.read_csv(CSV_PATH)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
    df = df[df["Price"] > 0].reset_index(drop=True)
    print(f"    {len(df)} valid products loaded.")

    print(f"\n🤖  Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("\n✍️  Building product texts …")
    texts = [build_product_text(row) for row in df.to_dict(orient="records")]

    print("\n🔢  Encoding embeddings (this takes ~30 s on CPU) …")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine sim via inner product
    )

    print("\n🗂️  Building FAISS index …")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product == cosine sim when normalised
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, INDEX_PATH)
    print(f"    FAISS index saved → {INDEX_PATH}  ({index.ntotal} vectors)")

    print("\n💾  Saving metadata …")
    metadata = df.to_dict(orient="records")
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"    Metadata saved → {METADATA_PATH}")

    print("\n✅  Index build complete!  Run `python rag.py` to start chatting.")


if __name__ == "__main__":
    main()
