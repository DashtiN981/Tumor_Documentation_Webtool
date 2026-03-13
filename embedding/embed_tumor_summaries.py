# embed_tumor_summaries.py
# -*- coding: utf-8 -*-
"""
Embed tumor summaries (summary_text) from AllTumorReport_ExtractedData.csv
using all-MiniLM-L6-v2 and save as tumor_summaries_embeddings.json.

Input:
    ./data/AllTumorReport_ExtractedData.csv

Output:
    ./embedding/tumor_summaries_embeddings.json
"""

import os
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# --------- PATHS (adjust if needed) ----------
DATA_DIR = Path("./data")
CSV_NAME = "AllTumorReport_ExtractedData3.csv"

OUT_DIR = Path("./embedding")
OUTPUT_JSON = OUT_DIR / "TumorSummary_embeddings_bge_m3.json"

MODEL_NAME = "BAAI/bge-m3" #all-MiniLM-L6-v2
# ---------------------------------------------


def main():

    csv_path = DATA_DIR / CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if "summary_text" not in df.columns:
        raise ValueError("Column 'summary_text' is missing in CSV.")
    if "doc_id" not in df.columns:
        raise ValueError("Column 'doc_id' is missing in CSV.")

    # Clean & prepare summary_text
    df["doc_id"] = df["doc_id"].astype(str)
    df["summary_text"] = df["summary_text"].fillna("").astype(str)

    mask_nonempty = df["summary_text"].str.strip().ne("")
    df_use = df[mask_nonempty].reset_index(drop=True)

    print(f"[INFO] Total rows in CSV: {len(df)}")
    print(f"[INFO] Rows with non-empty summary_text: {len(df_use)}")

    texts = df_use["summary_text"].tolist()

    print(f"[INFO] Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("[INFO] Encoding summaries (batch mode)...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    records = []

    print("[INFO] Building JSON records...")
    for i, row in tqdm(df_use.iterrows(), total=len(df_use)):

        emb = embeddings[i].tolist()

        def split_codes(v):
            if isinstance(v, float) and pd.isna(v):
                return []
            s = str(v).strip()
            if not s:
                return []
            return [x.strip() for x in s.split(";") if x.strip()]

        rec = {
            "doc_id": row["doc_id"],
            "summary_text": row["summary_text"],

            # Ground truth fields from CSV
            "icdo_topography_primary": str(row.get("icdo_topography_primary", "")),
            "icdo_morphology_primary": str(row.get("icdo_morphology_primary", "")),

            "icdo_topography_all": split_codes(row.get("icdo_topography_all", "")),
            "icdo_morphology_all": split_codes(row.get("icdo_morphology_all", "")),
            "icd10_all": split_codes(row.get("icd10_all", "")),

            # Embedding vector
            "embedding": emb
        }

        records.append(rec)

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved tumor summary embeddings to: {OUTPUT_JSON}")
    print(f"[OK] Total embedded items: {len(records)}")


if __name__ == "__main__":
    main()
