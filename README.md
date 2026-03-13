# ICD-O RAG Pipeline

A Python-based research pipeline for OCR-based processing of synthetic clinical oncology documents, tumor summarization, embedding generation, and ICD-O topography/morphology prediction using retrieval-augmented generation (RAG).

## Overview

This repository contains scripts for:

- converting clinical PDF/image documents into OCR text outputs
- merging multiple OCR-generated documents per patient
- extracting and summarizing tumor-related information from OCR outputs
- generating embeddings for tumor summaries
- generating embeddings for ICD-O morphology and topography dictionaries
- running an LLM-assisted RAG pipeline for ICD-O code prediction
- evaluating retrieval quality and final prediction performance

This repository is designed for experimentation with synthetic or non-sensitive sample clinical data.

## Main Workflow

The recommended workflow is:

1. Convert patient PDF/image files into OCR-readable text or markdown files
2. Merge OCR documents belonging to the same patient
3. Extract tumor-related summaries from the merged documents
4. Create embeddings for tumor summaries
5. Create embeddings for ICD-O morphology terms
6. Create embeddings for ICD-O topography terms
7. Run the ICD-O RAG prediction pipeline
8. Evaluate retrieval and final coding performance

## Repository Structure

- `OCR/` or `PaddleOCR script`  
  Converts patient PDFs/images into OCR text/markdown outputs  
  **Note:** this script is part of the project workflow but is not included in this repository snapshot

- `Merge_patient_documents.py`  
  Merge multiple OCR-generated documents per patient into one file

- `ExtractAllFilesInfo_Debug.py`  
  Extract and summarize tumor-related information from OCR/text/PDF files

- `embed_tumor_summaries.py`  
  Generate embeddings for tumor summaries

- `Morphology_Embedding_ICD.py`  
  Generate embeddings for ICD-O morphology terms

- `Topography_Embedding_ICD.py`  
  Generate embeddings for ICD-O topography terms

- `ICDO_RAG_FromSummary.py`  
  Main RAG pipeline for ICD-O morphology and topography prediction

- `Evaluate_icdo_retrieval_recall.py`  
  Evaluate retrieval quality (Recall@K)

- `Evaluate_icdo_rag_anymatch.py`  
  Evaluate final ICD-O predictions against ground truth

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
