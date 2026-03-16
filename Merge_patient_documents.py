"""
File Name: merge_patient_documents.py
Author: Naghme Dashti (project version)

Description:
------------
This script merges multiple OCR-generated `.md or .txt` documents belonging to 
the same patient into a single unified Markdown file. The patient ID is
determined using the prefix before the first underscore "_" in the file name.

Example:
    P001_discharge_04-11-2025.txt
    P001_followup_15-12-2025.txt

Both belong to patient "P001".

Output:
-------
For each patient, one merged file is created:
    merged_patients/P001.md

Each merged file contains all documents in order (sorted by date if available).
"""

import os
from pathlib import Path
from collections import defaultdict
import re

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------

INPUT_DIR = Path("./main_data/text")
OUTPUT_DIR = Path("./main_data/merged_patients")
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------
# Helper function: Extract patient_id from file name
# -----------------------------------------------------------

def extract_patient_id(filename: str) -> str:
    base = os.path.basename(filename)
    return base.split("_", 1)[0]

# -----------------------------------------------------------
# Helper function: extract a sorting key
# Supports DD-MM-YYYY in file names
# -----------------------------------------------------------

def extract_sort_key(filename: str) -> str:
    base = os.path.basename(filename)

    # Match DD-MM-YYYY
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})", base)
    if m:
        day, month, year = m.groups()
        return f"{year}-{month}-{day}_{base}"

    return base

# -----------------------------------------------------------
# STEP 1 — Group all .md and .txt files by patient_id
# -----------------------------------------------------------

patient_docs = defaultdict(list)

for pattern in ("*.md", "*.txt"):
    for path in INPUT_DIR.glob(pattern):
        patient_id = extract_patient_id(path.name)
        patient_docs[patient_id].append(path)

print(f"Found {sum(len(v) for v in patient_docs.values())} files for {len(patient_docs)} patients.")

# -----------------------------------------------------------
# STEP 2 — Merge documents per patient
# -----------------------------------------------------------

for patient_id, paths in patient_docs.items():
    paths_sorted = sorted(paths, key=lambda p: extract_sort_key(p.name))

    merged_parts = [f"# Patient {patient_id} - Merged OCR Documents\n"]

    for idx, p in enumerate(paths_sorted, start=1):
        doc_title = p.name

        with p.open("r", encoding="utf-8") as f:
            content = f.read()

        merged_parts.append(f"## Document {idx}: {doc_title}\n")
        merged_parts.append(content.strip() + "\n")
        merged_parts.append("\n---\n")

    merged_text = "\n".join(merged_parts).rstrip("\n-")

    out_path = OUTPUT_DIR / f"{patient_id}.md"

    with out_path.open("w", encoding="utf-8") as f:
        f.write(merged_text)

    print(f"Merged {len(paths_sorted)} documents for patient {patient_id} -> {out_path}")