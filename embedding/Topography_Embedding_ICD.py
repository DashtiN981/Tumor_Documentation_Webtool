from sentence_transformers import SentenceTransformer
import pandas as pd
import json
from tqdm import tqdm

df = pd.read_excel("./data/ICD-O-3_Topo_20240228.xlsx")

codes = df["Code"].astype(str).tolist()
terms = df["ShortDescription"].astype(str).tolist()

model = SentenceTransformer("BAAI/bge-m3") #all-MiniLM-L6-v2

records = []

for code, term in tqdm(zip(codes, terms), total=len(terms), desc="Embedding Topography"):
    term = term.strip()
    if not term or term.lower() == "nan":
        continue

    emb = model.encode(term).tolist()

    records.append({
        "code": code,
        "term": term,
        "embedding": emb
    })

with open("./embedding/Topo_embeddings_bge_m3.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("Done! Topo_embeddings.json created.")
