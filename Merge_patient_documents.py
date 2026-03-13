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
    merged_patients/P001_merged.md

Each merged file contains all documents in order (sorted by date if available).
"""

import os
from pathlib import Path
from collections import defaultdict
import re

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------

# Path to the directory containing all .md OCR files
INPUT_DIR = Path("./main_data/txtfiles")   # ← REPLACE with your actual directory

# Output directory for merged patient documents
OUTPUT_DIR = Path("./main_data/merged_patients")
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------
# Helper function: Extract patient_id from file name
# -----------------------------------------------------------

def extract_patient_id(filename: str) -> str:
    """
    Extracts the patient ID from the file name.
    Patient ID = prefix before the first underscore "_".

    Example:
        "P001_discharge_04-11-2025.txt" → "P001"
    """
    base = os.path.basename(filename)
    return base.split("_", 1)[0]


# -----------------------------------------------------------
# Helper function: extract a sorting key (prefer date if exists)
# -----------------------------------------------------------

def extract_sort_key(filename: str) -> str:
    """
    Generates a sorting key for ordering documents inside a patient group.

    If the filename contains a date in YYYY-MM-DD format, 
    the date is used as the primary sorting key.
    Otherwise, the filename itself is used.

    Example:
        "P001_discharge_04-11-2025.txt" → "2025-11-04_P001_discharge_04-11-2025.txt"
    """
    base = os.path.basename(filename)
    
    # Search for YYYY-MM-DD date pattern
    m = re.search(r"\d{4}-\d{2}-\d{2}", base)
    if m:
        return m.group(0) + "_" + base
    return base


# -----------------------------------------------------------
# STEP 1 — Group all .md files by patient_id
# -----------------------------------------------------------

patient_docs = defaultdict(list)

for path in INPUT_DIR.glob("*.md"):
    patient_id = extract_patient_id(path.name)
    patient_docs[patient_id].append(path)


# -----------------------------------------------------------
# STEP 2 — Merge documents per patient
# -----------------------------------------------------------

for patient_id, paths in patient_docs.items():
    
    # Sort documents using extracted sort keys
    paths_sorted = sorted(paths, key=lambda p: extract_sort_key(p.name))
    
    merged_parts = [f"# Patient {patient_id} – Merged OCR Documents\n"]

    # Append each file's content into the merged Markdown
    for idx, p in enumerate(paths_sorted, start=1):
        doc_title = p.name
        
        with p.open("r", encoding="utf-8") as f:
            content = f.read()
        
        merged_parts.append(f"## Document {idx}: {doc_title}\n")
        merged_parts.append(content.strip() + "\n")
        merged_parts.append("\n---\n")

    # Join all parts and clean trailing separators
    merged_text = "\n".join(merged_parts).rstrip("\n-")

    # Output path for this patient's merged file
    out_path = OUTPUT_DIR / f"{patient_id}.md"
    
    with out_path.open("w", encoding="utf-8") as f:
        f.write(merged_text)

    print(f"Merged {len(paths_sorted)} documents for patient {patient_id} → {out_path}")
