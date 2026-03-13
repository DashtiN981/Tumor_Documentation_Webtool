# ICD-O RAG Pipeline

Python scripts for OCR-based clinical document processing, tumor summarization, embedding generation, and ICD-O topography/morphology prediction using retrieval-augmented generation (RAG).

## Overview

This repository contains scripts for:

- converting patient PDF/image documents into OCR text outputs
- merging multiple OCR-generated documents per patient
- extracting and summarizing tumor-related information from OCR outputs
- generating embeddings for tumor summaries
- generating embeddings for ICD-O morphology and topography dictionaries
- running an LLM-assisted RAG pipeline for ICD-O code prediction
- evaluating retrieval quality and final coding performance

This project is intended for research and prototyping with synthetic or non-sensitive sample clinical data.

## Workflow

The recommended workflow is:

1. Run OCR on patient PDF/image files
2. Merge OCR-generated documents belonging to the same patient
3. Extract tumor summaries from the merged documents
4. Create tumor summary embeddings
5. Create ICD-O morphology embeddings
6. Create ICD-O topography embeddings
7. Run the RAG prediction pipeline
8. Evaluate retrieval quality
9. Evaluate final predictions

## Project Structure

- `PaddleOCR / OCR script` — converts PDF/image files into OCR text or Markdown files  
  **Note:** this script is part of the workflow but is not included in this repository snapshot

- `Merge_patient_documents.py` — merges multiple OCR-generated Markdown files per patient

- `ExtractAllFilesInfo_Debug.py` — extracts and summarizes tumor-related information from clinical text/PDF/OCR outputs

- `embed_tumor_summaries.py` — generates embeddings for tumor summaries

- `Morphology_Embedding_ICD.py` — generates embeddings for ICD-O morphology terms

- `Topography_Embedding_ICD.py` — generates embeddings for ICD-O topography terms

- `ICDO_RAG_FromSummary.py` — main RAG-based ICD-O prediction pipeline

- `Evaluate_icdo_retrieval_recall.py` — evaluates retrieval quality (Recall@K)

- `Evaluate_icdo_rag_anymatch.py` — evaluates final predictions against ground truth

## Requirements

- Python 3.10+
- pip
- a virtual environment is recommended

## Installation

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Configuration

This project uses an OpenAI-compatible API endpoint for extraction and RAG-based prediction.

Do not hardcode credentials in the source code.
Instead, use environment variables or a local .env file.

Example:

OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=http://pluto/v1
MODEL_NAME=your_model_name_here

## Input Data

The pipeline starts from patient documents such as:

  PDF files
  scanned image files
  OCR-generated text or Markdown files
  Typical intermediate and downstream files include:
  OCR output files
  merged patient documents
  extracted tumor summary CSV/JSON files
  tumor summary embeddings
  ICD-O morphology embeddings
  ICD-O topography embeddings
  prediction JSON files
  evaluation outputs

## Usage
1. Run OCR on PDF/image files
First, convert raw patient PDF/image files into OCR outputs such as:
.md
.txt
.json

Important:
The OCR script (for example, based on PaddleOCR) is part of the overall pipeline, but it is not included in this repository snapshot.

2. Merge OCR-generated patient documents
python .\Merge_patient_documents.py
This script groups .md OCR files by patient ID and creates one merged Markdown file per patient.

3. Extract tumor summaries
python .\ExtractAllFilesInfo_Debug.py
This script extracts and summarizes tumor-related information from OCR/text/PDF inputs and also derives ICD-like codes found in the raw text.

4. Create tumor summary embeddings
python .\embed_tumor_summaries.py
This script embeds rows with non-empty summary_text and stores JSON records containing doc_id, summary_text, ground-truth fields, and embedding vectors.

5. Create ICD-O morphology embeddings
python .\Morphology_Embedding_ICD.py
This script reads ./data/ICD-O-3_Morph_20240228.xlsx and writes ./embedding/Morph_embeddings_bge_m3.json.

6. Create ICD-O topography embeddings
python .\Topography_Embedding_ICD.py
This script reads ./data/ICD-O-3_Topo_20240228.xlsx and writes ./embedding/Topo_embeddings_bge_m3.json.

7. Run the ICD-O RAG pipeline
python .\ICDO_RAG_FromSummary.py
The pipeline loads precomputed embeddings for:
  tumor summaries
  ICD-O morphology
  ICD-O topography

Then it performs hybrid retrieval (semantic + lexical), optionally applies organ-family filtering for topography candidates, prompts the LLM to choose one morphology code and one topography code, and saves the predictions to JSON.

8. Evaluate retrieval quality
python .\Evaluate_icdo_retrieval_recall.py
This script measures Recall@K-style retrieval performance using candidate morphology and topography code lists compared against ground truth.

9. Evaluate final predictions
python .\Evaluate_icdo_rag_anymatch.py
This script evaluates final morphology and topography predictions against ground truth using exact and any-match style metrics.

## Expected Files
Examples of files used by the scripts include:
  ./data/ICD-O-3_Morph_20240228.xlsx
  ./data/ICD-O-3_Topo_20240228.xlsx
  tumor summary CSV file with summary_text
  prediction JSON files in results/
  embedding JSON files in embedding/

## Output Files

Typical outputs include:
  merged patient Markdown files
  extracted tumor summary CSV/JSON files
  ./embedding/Morph_embeddings_bge_m3.json
  ./embedding/Topo_embeddings_bge_m3.json
  tumor summary embedding JSON files
  results/ICDO_RAG_predictions_*.json
  evaluation metrics printed to console or saved downstream

##Notes
The full pipeline starts with OCR on PDF/image patient documents.
Before running the main RAG script, you should already have:
  OCR outputs
  merged patient files
  extracted tumor summaries
  tumor summary embeddings
  ICD-O morphology embeddings
  ICD-O topography embeddings
The OCR script is part of the project workflow but is not included in this repository snapshot.
Do not commit private credentials, sensitive datasets, or large generated artifacts.
This repository is intended for code and safe sample/demo data only