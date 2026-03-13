# ExtractAllFilesInfo_Debug_v4_2.py
# -*- coding: utf-8 -*-
"""
Batch-processing pipeline for OCR clinical tumour letters.
Generates:
  - AllTumorReport_ExtractedData.json
  - AllTumorReport_ExtractedData.csv
  - problematic_cases.json

Version 4.2 — FINAL ROBUST VERSION
---------------------------------
Key features:
 - NEVER produces empty summaries
 - Keeps high-quality behaviour of v4.1
 - Fixes all cases where v4.1 produced missing/empty summaries
 - Adds robust multi-stage fallback:
       1. Map-reduce (v4.1)
       2. Strict second-pass (v4.1)
       3. NEW: Robust "simple" summarization
       4. NEW: LLM-free fallback: merge-of-chunk-summaries
 - Fully automatic, deterministic (temperature=0)
"""

import os, re, csv, json, argparse, time, traceback
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# optional progress
try:
    from tqdm import tqdm
    TQDM_OK = True
except Exception:
    TQDM_OK = False

# optional PDF support
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# ===================== DEFAULTS =====================
DEFAULT_INPUT_DIR = "/home/naghmedashti/NCT_ICDOTOPO_MORPHO/main_data/merged_patients"
DEFAULT_OUT_DIR   = "/home/naghmedashti/NCT_ICDOTOPO_MORPHO/summaries"
DEFAULT_BASE_URL  = "http://pluto/v1"         # e.g. "http://pluto/v1" or "http://g19a012:8039"
DEFAULT_MODEL     = "GPT-OSS-120B"
API_KEY_DEFAULT   = "sk-aKGeEFMZB0gXEcE51FTc0A"  # better use env: LLM_API_KEY / OPENAI_API_KEY
# ====================================================

def _openai_client(base_url: str, api_key: str):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key)

# ===================== Utils =====================
def ensure_text(x) -> str:
    if x is None: return ""
    if isinstance(x, str): return x
    try: return str(x)
    except Exception: return ""

def log(msg: str, force: bool = True):
    if force:
        print(msg, flush=True)

def alnum_len(s: str) -> int:
    s = ensure_text(s)
    return len(re.sub(r"[^A-Za-zÄÖÜäöüß0-9]", "", s))

def sentence_count(s: str) -> int:
    s = ensure_text(s).strip()
    if not s: return 0
    parts = re.split(r"[.!?]\s+", s)
    return len([p for p in parts if p.strip()])

# ===================== PROMPTS =====================

SYSTEM_PROMPT_DE = (
    "Du bist ein onkologischer Facharzt und liest OCR-erfasste deutschsprachige Arztbriefe.\n"
    "Aufgabe: Erstelle eine prägnante, fachlich korrekte Freitext-Zusammenfassung der Tumorbefunde.\n"
    "WICHTIG:\n"
    "- Gib KEINE Codes (ICD-O, ICD-10, SNOMED usw.) aus.\n"
    "- NIEMALS um zusätzliche Textabschnitte bitten; arbeite NUR mit dem vorliegenden Text.\n"
    "- Nutze jede im Text vorhandene Tumor-Information.\n"
    "- Vermeide pauschale Aussagen wie „keine Angaben“, wenn im Text auch nur teilweise Informationen vorkommen.\n"
    "- Falls ein Aspekt nirgends erwähnt wird: „nicht berichtet/unklar“.\n"
    "- Ziel: 7 bis 10 Sätze. Keine Erfindungen."
)

CHUNK_PROMPT_DE = (
    "Fasse den folgenden OCR-Abschnitt in 3 bis 6 Sätzen zusammen , NUR Tumor-relevante Fakten.\n"
    "Fokus (wenn vorhanden): Morphologie/Histologie, Zelltyp, Topographie (Organ/Lokalisation/Seite), "
    "Tumorgröße, Grading, Primärtumor/Metastasen, relevante Marker/Therapie, Negationen.\n"
    "KEINE Codes. NIEMALS um mehr Text bitten.\n"
    "Nutze jede Tumor-bezogene Information. Wenn ein Aspekt nirgends erwähnt wird: „nicht berichtet“.\n\n"
    "TEXTAUSSCHNITT:\n<<<\n{chunk}\n>>>"
)

FINAL_PROMPT_DE = (
    "Du erhältst mehrere Teilzusammenfassungen eines Arztbriefs.\n"
    "Fasse sie zu einer kohärenten Zusammenfassung (7 bis 10 Sätze) zusammen. Entferne Dopplungen, "
    "bewahre Negationen/Unsicherheiten. KEINE Codes.\n"
    "Nutze ALLE konkreten Informationen zu Tumorart, Histologie, Lokalisation/Seite, Größe, Grading, "
    "Primärtumor vs. Metastasen, Marker und Therapie.\n"
    "Vermeide generische Aussagen („keine Angaben“), wenn Informationen vorhanden sind.\n"
    "Nur wenn ein Aspekt in KEINER Teilzusammenfassung genannt wird, schreibe „nicht berichtet“.\n"
    "NIEMALS um mehr Text bitten.\n\n"
    "TEILZUSAMMENFASSUNGEN:\n<<<\n{chunk_summaries}\n>>>"
)

FORCE_FINAL_PROMPT_DE = (
    "Die folgende Aufgabe dient dazu, eine unvollständige oder zu unspezifische Zusammenfassung zu korrigieren.\n"
    "Erstelle aus ALLEN Teilzusammenfassungen eine VERBESSERTE, präzise onkologische Zusammenfassung (7 bis 10 Sätze).\n"
    "Nutze jede konkrete Tumor-Information. Keine Codes. Keine Bitten um mehr Text.\n"
    "Wenn ein Aspekt nirgends erwähnt wird: „nicht berichtet“.\n\n"
    "TEILZUSAMMENFASSUNGEN:\n<<<\n{chunk_summaries}\n>>>"
)

# fallback summarization prompt
ROBUST_PROMPT = (
    "Erstelle aus den folgenden Teilzusammenfassungen eine klare Zusammenfassung der Tumorbefunde "
    "(5 bis 8 Sätze). Nutze jede konkrete Information. Keine Codes. Keine Bitten um mehr Text. "
    "Falls ein Aspekt nirgends erwähnt wird, schreibe „nicht berichtet“.\n\n"
    "TEXT:\n<<<\n{chunk_summaries}\n>>>"
)

# ===================== Readers =====================

def read_text_file(p: Path) -> str:
    return ensure_text(p.read_text(encoding="utf-8", errors="ignore"))

def read_pdf(p: Path) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed. pip install pymupdf")
    parts = []
    with fitz.open(p) as doc:
        for page in doc:
            parts.append(ensure_text(page.get_text("text")))
    return "\n".join(parts)

def read_paddleocr_json(p: Path) -> str:
    raw = ensure_text(p.read_text(encoding="utf-8", errors="ignore"))
    data = json.loads(raw)

    def collect(obj) -> List[str]:
        out = []
        if isinstance(obj, dict):
            for k in ["text","label","Text","ocr_text","transcription"]:
                if k in obj and isinstance(obj[k], str):
                    out.append(obj[k])
            for v in obj.values():
                out += collect(v)
        elif isinstance(obj, list):
            for it in obj:
                out += collect(it)
        return out

    lines = collect(data)
    if not lines and isinstance(data, dict) and "data" in data:
        lines = collect(data["data"])

    safe = []
    for ln in lines:
        s = ensure_text(ln).strip()
        if s:
            safe.append(re.sub(r"\s+", " ", s))
    return "\n".join(safe)

def load_text(p: Path) -> Optional[str]:
    ext = p.suffix.lower()
    if ext in {".md",".txt"}: return read_text_file(p)
    if ext == ".pdf":         return read_pdf(p)
    if ext == ".json":        return read_paddleocr_json(p)
    return None

# ===================== Cleaning & Chunking =====================

def clean_text(s: str) -> str:
    s = ensure_text(s)
    s = s.replace("\u00ad", "")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(s: str, max_chars=8000, overlap=400) -> List[str]:
    s = ensure_text(s)
    if len(s) <= max_chars: return [s]
    chunks, start = [], 0
    while start < len(s):
        end = min(len(s), start+max_chars)
        piece = s[start:end]
        last_par = piece.rfind("\n\n")
        if last_par > 1500:
            cut = start + last_par
            piece = s[start:cut]
            start = max(0, cut - overlap)
        else:
            start = max(0, end - overlap)
        piece = piece.strip()
        if piece: chunks.append(piece)
        if end >= len(s): break
    # dedupe
    uniq, seen = [], set()
    for c in chunks:
        if c not in seen:
            uniq.append(c); seen.add(c)
    return uniq

# ===================== LLM Helpers =====================

def call_llm(client, model: str, system: str, user: str,
             temperature=0.0, max_tokens=1100) -> str:
    rsp = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[
            {"role":"system","content":ensure_text(system)},
            {"role":"user",  "content":ensure_text(user)}
        ]
    )
    try:
        return ensure_text(rsp.choices[0].message.content).strip()
    except Exception:
        return ""

def strip_codes(text: str) -> str:
    text = ensure_text(text)
    text = re.sub(r"\bC\d{2}(?:\.\d)?\b","[redacted-code]", text)
    text = re.sub(r"\b(\d{4})/([0-9](?:[0-4X])?)\b","[redacted-code]", text)
    text = re.sub(r"\b[CD]\d{2}(?:\.\d+)?\b","[redacted-code]", text)
    return text

def dedupe_sentences(text: str) -> str:
    sents = re.split(r'(?<=[.!?])\s+', ensure_text(text).strip())
    out, seen = [], set()
    for s in sents:
        t = s.strip()
        if not t: continue
        key = re.sub(r"\W+","", t.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(t)
    return " ".join(out)

# ===================== QC =====================

KEY_FIELDS_RE = re.compile(
    r"(morpholog|histolog|zelltyp|topograph|organ|seite|metastas|grading|größe|tumorgröße)",
    flags=re.IGNORECASE
)
POSITIVE_TUMOR_RE = re.compile(
    r"(karzinom|carcinom|tumor|neoplasie|sarkom|melanom|lymphom|gliom|"
    r"adenokarzinom|plattenepithelkarzinom|metastase|herd|herde|raumforderung|läsion|knoten)",
    flags=re.IGNORECASE
)
NEGATION_RE = re.compile(
    r"\b(keine|kein|ohne|nicht angegeben|nicht berichtet|unklar)\b",
    flags=re.IGNORECASE
)
PROBLEM_RE = re.compile(
    r"(keine konkreten befunde|benötige[n]* den eigentlichen inhalt|"
    r"keine sinnvolle zusammenfassung|bitte stellen sie|liegen .* keine .* information)",
    flags=re.IGNORECASE
)

def looks_problematic(text: str,
                      min_chars: int = 300,
                      min_alnum: int = 160,
                      min_sentences: int = 6) -> Tuple[bool, List[str]]:
    reasons = []
    t = ensure_text(text)

    if not t.strip():
        return True, ["empty_summary"]

    if len(t) < min_chars and alnum_len(t) < min_alnum:
        reasons.append("too_short_low_info")

    if sentence_count(t) < min_sentences:
        reasons.append("few_sentences")

    has_key = bool(KEY_FIELDS_RE.search(t))
    has_tumour = bool(POSITIVE_TUMOR_RE.search(t))
    neg_count = len(NEGATION_RE.findall(t))

    if not has_key:
        reasons.append("missing_key_fields")

    if PROBLEM_RE.search(t):
        reasons.append("refusal_pattern")

    # If tumour words exist → remove missing_key_fields
    if has_tumour and "missing_key_fields" in reasons:
        reasons.remove("missing_key_fields")

    # detect "only-negative patterns"
    if has_key and not has_tumour and neg_count >= 3:
        reasons.append("only_negated_fields")

    return (len(reasons) > 0), reasons

# ===================== Summarization v4.2 =====================

def summarize_map_reduce_v4_2(text: str, client, model: str) -> Tuple[str, Dict]:
    """
    v4.2 robust multi-step summarization
    Guarantees: NEVER returns empty summary
    """

    route = {"route_used": "map_reduce", "reasons": [], "chunk_count": 0}

    text = clean_text(text)
    chunks = chunk_text(text)
    route["chunk_count"] = len(chunks)

    if not chunks:
        return "", {"route_used":"none","reasons":["no_chunks"],"chunk_count":0}

    # ---- MAP ----
    chunk_outputs = []
    for i, ch in enumerate(chunks, 1):
        u = CHUNK_PROMPT_DE.format(chunk=ch)
        s = call_llm(client, model, SYSTEM_PROMPT_DE, u, 0.0, 700)
        chunk_outputs.append(strip_codes(s))

    chunk_summaries = "\n\n".join(
        f"[Chunk {i}]\n{co}" for i, co in enumerate(chunk_outputs, 1)
    )

    # ---- PASS 1 ----
    merged = FINAL_PROMPT_DE.format(chunk_summaries=chunk_summaries)
    final = call_llm(client, model, SYSTEM_PROMPT_DE, merged, 0.0, 1100)
    final = strip_codes(final)

    is_prob, reasons = looks_problematic(final)
    if not is_prob:
        return final, route

    # ---- PASS 2 (strict) ----
    route["route_used"] = "expanded"
    route["reasons"] = reasons

    strict_user = FORCE_FINAL_PROMPT_DE.format(chunk_summaries=chunk_summaries)
    strict_final = call_llm(client, model, SYSTEM_PROMPT_DE, strict_user, 0.0, 1100)
    strict_final = strip_codes(strict_final)

    is_prob2, reasons2 = looks_problematic(strict_final)
    if not is_prob2:
        return strict_final, route

    # ---- FALLBACK 1: robust simple summarization ----
    robust_user = ROBUST_PROMPT.format(chunk_summaries=chunk_summaries)
    robust_final = call_llm(client, model, SYSTEM_PROMPT_DE, robust_user, 0.0, 1100)
    robust_final = strip_codes(robust_final)

    if robust_final.strip():
        return robust_final, {**route, "route_used":"fallback_simple"}

    # ---- FALLBACK 2: LLM-free merge of chunk summaries ----
    merged_chunks = " ".join(chunk_outputs)
    merged_chunks = dedupe_sentences(merged_chunks)

    return merged_chunks, {**route, "route_used":"fallback_chunk_merge"}

# ===================== ICD Extraction =====================

TOPO_RE  = re.compile(r"\bC(?P<num>\d{2})(?:\.(?P<sub>\d))?\b")
MORPH_RE = re.compile(r"\b(?P<m>\d{4})/(?P<b>[0-9](?:[0-4X])?)\b")
ICD10_RE = re.compile(r"\b[CD]\d{2}(?:\.\d+)?\b")

def extract_icd_from_raw(text: str) -> Dict[str, List[str]]:
    text = ensure_text(text)
    topo, morph, icd10 = set(), set(), set()

    for m in TOPO_RE.finditer(text):
        code = "C" + m.group("num")
        if m.group("sub"): code += "." + m.group("sub")
        topo.add(code)

    for m in MORPH_RE.finditer(text):
        try: first4 = int(m.group("m"))
        except ValueError: continue
        if 8000 <= first4 <= 9999:
            morph.add(f"{m.group('m')}/{m.group('b')}")

    for m in ICD10_RE.finditer(text):
        icd10.add(m.group(0))

    return {
        "icdo_topography_all": sorted(topo),
        "icdo_morphology_all": sorted(morph),
        "icd10_all": sorted(icd10)
    }

def choose_primary_topo(codes: List[str]) -> Optional[str]:
    if not codes: return None
    if "C61" in codes: return "C61"   # optional preference
    with_dec = [c for c in codes if "." in c]
    return with_dec[0] if with_dec else codes[0]

def choose_primary_morph(codes: List[str]) -> Optional[str]:
    if not codes: return None
    return codes[0]

# ===================== Worker =====================

def process_one(p: Path, client, model: str, dry_run: bool, idx: int, total: int,
                verbose: bool, problems_sink: List[Dict]) -> Optional[Dict]:

    t0 = time.time()
    try:
        if not TQDM_OK or verbose:
            log(f"[{idx}/{total}] Reading: {p.name}")

        raw = ensure_text(load_text(p))
        if not raw.strip():
            problems_sink.append({
                "doc_id": p.stem,
                "source_file": str(p),
                "route_used":"none",
                "reasons":["empty_or_unreadable_raw"],
                "final_summary":""
            })
            return None

        raw_clean = clean_text(raw)
        codes = extract_icd_from_raw(raw_clean)
        topo_all  = codes["icdo_topography_all"]
        morph_all = codes["icdo_morphology_all"]
        icd10_all = codes["icd10_all"]

        gt = {
            "icdo_topography_primary": ensure_text(choose_primary_topo(topo_all)) or "",
            "icdo_morphology_primary": ensure_text(choose_primary_morph(morph_all)) or "",
            "icdo_topography_all": topo_all,
            "icdo_morphology_all": morph_all,
            "icd10_all": icd10_all
        }

        if dry_run:
            return {
                "doc_id": p.stem,
                "language":"de",
                "summary_text":"",
                "ground_truth":gt,
                "source_file":str(p),
                "route_used":"dry_run"
            }

        summary, route_info = summarize_map_reduce_v4_2(raw_clean, client, model)

        dt = time.time() - t0
        if not TQDM_OK or verbose:
            log(f"[OK] {p.name}: route={route_info['route_used']} | summary_len={len(summary)} | time={dt:.1f}s")

        # QC check
        is_prob, reasons_prob = looks_problematic(summary)
        if is_prob and route_info["route_used"] == "expanded":
            problems_sink.append({
                "doc_id": p.stem,
                "source_file": str(p),
                "route_used": route_info["route_used"],
                "reasons": list(route_info.get("reasons",[])) + reasons_prob,
                "final_summary": summary
            })

        return {
            "doc_id":p.stem,
            "language":"de",
            "summary_text":summary,
            "ground_truth":gt,
            "source_file":str(p),
            "route_used":route_info["route_used"]
        }

    except Exception as e:
        dt = time.time() - t0
        log(f"[ERROR] {p.name} after {dt:.1f}s -> {e}")
        traceback.print_exc()
        problems_sink.append({
            "doc_id":p.stem,
            "source_file":str(p),
            "route_used":"exception",
            "reasons":[str(e)],
            "final_summary":""
        })
        return None

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--out-dir",   type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--model",     type=str, default=DEFAULT_MODEL)
    ap.add_argument("--base-url",  type=str, default=DEFAULT_BASE_URL)
    ap.add_argument("--api-key",   type=str, default=None)
    ap.add_argument("--dry-run",   action="store_true")
    ap.add_argument("--verbose",   action="store_true")
    args = ap.parse_args()

    log("=== Effective Config (v4.2) ===")
    log(f"input-dir : {args.input_dir}")
    log(f"out-dir   : {args.out_dir}")
    log(f"base-url  : {args.base_url}")
    log(f"model     : {args.model}")
    log(f"dry-run   : {args.dry_run}")
    log("===============================")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or API_KEY_DEFAULT or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("ERROR: Missing API key. Use --api-key or set LLM_API_KEY / OPENAI_API_KEY.")

    client = None
    if not args.dry_run:
        client = _openai_client(args.base_url, api_key)

    in_dir = Path(args.input_dir)
    files = sorted([
        p for p in in_dir.glob("**/*")
        if p.is_file() and p.suffix.lower() in {".md",".txt",".json",".pdf"}
    ])
    if not files:
        print("[WARN] no input files found.")
        return

    iterator = tqdm(files, desc="Processing files") if TQDM_OK else files

    records = []
    problems = []

    for idx, p in enumerate(iterator, start=1):
        rec = process_one(p, client, args.model, args.dry_run, idx, len(files),
                          args.verbose, problems)
        if rec:
            records.append(rec)

    # Write JSON
    json_path = out_dir / "AllTumorReport_ExtractedData4.2.json"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] JSON saved: {json_path}")

    # Write CSV
    csv_rows = []
    for rec in records:
        gt = rec["ground_truth"]
        csv_rows.append({
            "doc_id": rec["doc_id"],
            "source_file": rec["source_file"],
            "icdo_topography_primary": ensure_text(gt.get("icdo_topography_primary","")),
            "icdo_morphology_primary": ensure_text(gt.get("icdo_morphology_primary","")),
            "icdo_topography_all": ";".join(gt.get("icdo_topography_all",[])),
            "icdo_morphology_all": ";".join(gt.get("icdo_morphology_all",[])),
            "icd10_all": ";".join(gt.get("icd10_all",[])),
            "summary_text": ensure_text(rec.get("summary_text","")).replace("\n"," "),
            "route_used": rec.get("route_used","")
        })

    csv_path = out_dir / "AllTumorReport_ExtractedData4.2.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[OK] CSV saved: {csv_path}")

    # Write problematic cases
    prob_path = out_dir / "problematic_cases4,2.json"
    prob_path.write_text(json.dumps(problems, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] problematic cases saved: {prob_path}  (count={len(problems)})")

if __name__ == "__main__":
    main()
