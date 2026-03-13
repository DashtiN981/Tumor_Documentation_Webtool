from sentence_transformers import SentenceTransformer
import pandas as pd
import json
from tqdm import tqdm

# Load the ICD-O Morphology file
df = pd.read_excel("./data/ICD-O-3_Morph_20240228.xlsx")

# Extract codes and terms
codes = df["Code"].astype(str).tolist()
terms = df["ShortDescription"].astype(str).tolist()

model = SentenceTransformer("BAAI/bge-m3") #all-MiniLM-L6-v2

records = []

for code, term in tqdm(zip(codes, terms), total=len(terms), desc="Embedding Morphology"):
    term = term.strip()
    if not term or term.lower() == "nan":
        continue

    emb = model.encode(term).tolist()

    records.append({
        "code": code,
        "term": term,
        "embedding": emb
    })

# Save JSON
with open("./embedding/Morph_embeddings_bge_m3.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("Done! Morph_embeddings.json created.")
