# ICDO_RAG_FromSummary.py
# ===========================================================
# RAG-based ICD-O-3 Topography & Morphology suggestion from
# tumor summaries, with:
#   1) Organ-based filtering for Topography retrieval
#   2) Hybrid retrieval (semantic + lexical) for both Morph & Topo
#   3) More robust LLM JSON handling
#   4) Reduced prompt size
# ===========================================================

import os
import json
import random
import re
from typing import List, Dict, Any, Tuple, Set

import numpy as np
from openai import OpenAI

# ==========================
# Configuration
# ==========================

SUMMARY_EMB_JSON = "./embedding/TumorSummary_embeddings_bge_m3.json"
MORPH_EMB_JSON   = "./embedding/Morph_embeddings_bge_m3.json"
TOPO_EMB_JSON    = "./embedding/Topo_embeddings_bge_m3.json"

OUTPUT_JSON      = "./results/ICDO_RAG_predictions.json"

BASE_URL   = "Base-URL-Here" 
API_KEY    = "Api-Key-Here"   # 
MODEL_NAME = "Model-Name-Here" # e.g., "gpt-oss-120b"

TOP_K_FINAL    = 10
SEMI_K         = 200
LLM_TEMP       = 0.0
LLM_MAX_TOK    = 1000
RUN_SEED       = 42

ALPHA_SEMANTIC = 0.7
BETA_LEXICAL   = 0.3

MAX_DOCS       = None

random.seed(RUN_SEED)

# ==========================
# Helper utils
# ==========================

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_list_float(v) -> List[float]:
    if isinstance(v, list):
        return [float(x) for x in v]
    raise ValueError("Embedding is not a list")

def strip_code_suffix(raw: str) -> str:
    raw = str(raw or "").strip()
    return raw.split()[0].split("_")[0].strip()

def canonical_morph_code(raw: str) -> str:
    return strip_code_suffix(raw)


def canonical_topo_code(raw: str) -> str:
    return strip_code_suffix(raw)


def topo_family(code_full: str) -> str:
    code_full = canonical_topo_code(code_full)
    if len(code_full) >= 3:
        return code_full[:3]
    return code_full


def extract_json_from_text(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty LLM response")

    s = text.strip()

    start = s.find("{")
    if start == -1:
        raise ValueError("No '{' found in LLM response")

    depth = 0
    end = None
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        raise ValueError("No complete JSON object found in LLM response")

    snippet = s[start:end+1]
    return json.loads(snippet)


# ==========================
# Tokenizer & organ mapping
# ==========================

WORD_SPLIT_RE = re.compile(r"[^\wäöüÄÖÜß]+")

GERMAN_STOPWORDS: Set[str] = {
    "der", "die", "das", "und", "oder", "mit", "ohne", "im", "in", "am", "an",
    "zu", "von", "auf", "für", "bei", "ist", "sind", "war", "waren", "ein",
    "eine", "einer", "einem", "eines", "den", "dem", "des", "als", "auch",
    "nach", "vor", "aus"
}

def tokenize_text(text: str) -> List[str]:
    if not isinstance(text, str):
        text = str(text or "")
    text = text.lower()
    tokens = WORD_SPLIT_RE.split(text)
    tokens = [t for t in tokens if t and len(t) > 2 and t not in GERMAN_STOPWORDS]
    return tokens


ORG_KEYWORDS_TO_FAMILY = {
    "prostata": "C61",
    "prostate": "C61",
    "pankreas": "C25",
    "pancreas": "C25",
    "mamma": "C50",
    "brust": "C50",
    "mammakarzinom": "C50",
    "lunge": "C34",
    "pulmonal": "C34",
    "leber": "C22",
    "hepatisch": "C22",
    "colon": "C18",
    "kolon": "C18",
    "rektum": "C20",
    "rektal": "C20",
    "zervix": "C53",
    "cervix": "C53",
    "uterus": "C54",
    "endometrium": "C54",
    "ovar": "C56",
    "ovarial": "C56",
    "ovarium": "C56",
    "gehirn": "C71",
    "hirn": "C71",
    "zns": "C71",
    "haut": "C44",
    "cutan": "C44",
    "lymphknoten": "C77",
    "lymphknotenmetastase": "C77",
    "lymphom": "C77",
}

def detect_topo_families_from_summary(summary_text: str) -> Set[str]:
    text_lower = str(summary_text or "").lower()
    families: Set[str] = set()
    for keyword, fam in ORG_KEYWORDS_TO_FAMILY.items():
        if keyword in text_lower:
            families.add(fam)

    # مهم: metastatic / lymph node family را حذف می‌کنیم تا retrieval خراب نشود
    families.discard("C77")
    return families


# ==========================
# Loading data
# ==========================

def load_summary_embeddings(path: str) -> List[Dict[str, Any]]:
    log(f"Loading summary embeddings from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    for obj in data:
        if "doc_id" in obj and "summary_text" in obj and "embedding" in obj:
            out.append(obj)

    log(f"Loaded {len(out)} summaries with embeddings.")
    return out


def load_icdo_embedding_list(
    path: str,
    kind: str,
    is_topography: bool = False
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    log(f"Loading {kind} embeddings from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    embs = []

    for obj in data:
        code_raw = obj.get("code", "")
        term = obj.get("term", "")
        emb = ensure_list_float(obj.get("embedding", []))

        rec: Dict[str, Any] = {
            "code_raw": code_raw,
            "term": term,
            "term_tokens": tokenize_text(term),
        }

        if is_topography:
            fam = topo_family(canonical_topo_code(code_raw))
            rec["topo_family"] = fam

        items.append(rec)
        embs.append(emb)

    emb_matrix = np.asarray(embs, dtype="float32")

    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix = emb_matrix / norms

    log(f"Loaded {len(items)} {kind} terms with embeddings.")
    return items, emb_matrix


# ==========================
# Hybrid retrieval
# ==========================

def lexical_overlap_score(query_tokens: List[str], term_tokens: List[str]) -> float:
    if not term_tokens:
        return 0.0
    qs = set(query_tokens)
    ts = set(term_tokens)
    inter = qs.intersection(ts)
    return float(len(inter)) / float(len(ts))


def get_topk_candidates_hybrid(
    query_emb: np.ndarray,
    query_text: str,
    icdo_items: List[Dict[str, Any]],
    emb_matrix: np.ndarray,
    top_k_final: int,
    semi_k: int,
    organ_families: Set[str] = None,
    is_topography: bool = False,
    alpha_semantic: float = 0.7,
    beta_lexical: float = 0.3
) -> List[Dict[str, Any]]:
    if emb_matrix.size == 0:
        return []

    q = query_emb.astype("float32")
    q_norm = np.linalg.norm(q)
    if q_norm == 0.0:
        return []

    q = q / q_norm

    if is_topography and organ_families:
        valid_indices = [
            idx for idx, item in enumerate(icdo_items)
            if item.get("topo_family") in organ_families
        ]
        if not valid_indices:
            valid_indices = list(range(len(icdo_items)))
    else:
        valid_indices = list(range(len(icdo_items)))

    sub_embs = emb_matrix[valid_indices, :]
    sims_sem_all = sub_embs @ q

    m = len(valid_indices)
    if m == 0:
        return []

    semi_k = min(semi_k, m)
    idx_unsorted = np.argpartition(-sims_sem_all, semi_k - 1)[:semi_k]
    idx_sem_sorted = idx_unsorted[np.argsort(-sims_sem_all[idx_unsorted])]

    query_tokens = tokenize_text(query_text)

    candidates: List[Dict[str, Any]] = []
    for local_idx in idx_sem_sorted:
        global_idx = valid_indices[local_idx]
        item = icdo_items[global_idx]

        sem_score = float(sims_sem_all[local_idx])
        lex_score = lexical_overlap_score(query_tokens, item.get("term_tokens", []))
        combined = alpha_semantic * sem_score + beta_lexical * lex_score

        candidates.append({
            "code_raw": item["code_raw"],
            "term": item["term"],
            "semantic_score": sem_score,
            "lexical_score": lex_score,
            "score": combined
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_k_final]


# ==========================
# LLM prompt & call
# ==========================

def format_candidate_block(title: str, candidates: List[Dict[str, Any]]) -> str:
    lines = [title]
    for i, c in enumerate(candidates, 1):
        lines.append(f"{i}. {c['code_raw']} | {c['term']}")
    return "\n".join(lines)

def build_prompt(summary_text: str,
                 morph_candidates: List[Dict[str, Any]],
                 topo_candidates: List[Dict[str, Any]]) -> str:
    morph_block = format_candidate_block("Morphology candidates:", morph_candidates)
    topo_block  = format_candidate_block("Topography candidates:", topo_candidates)

    prompt = f"""
You are a medical coding assistant.

Select exactly ONE morphology code and ONE topography code from the candidate lists.

Important:
- Use only the provided candidates.
- For topography, prefer the PRIMARY tumor site, not metastatic sites.
- Return ONLY one valid JSON object.
- Keep the reason very short (maximum 1 short sentence).

Required JSON keys:
- morphology_description
- morphology_code
- topography_description
- topography_code
- reason

summary_text:
{summary_text}

{morph_block}

{topo_block}
"""
    return prompt.strip()


def call_llm_for_case(
    client: OpenAI,
    summary_text: str,
    morph_candidates: List[Dict[str, Any]],
    topo_candidates: List[Dict[str, Any]]
) -> Dict[str, Any]:
    prompt = build_prompt(summary_text, morph_candidates, topo_candidates)

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Return only a valid JSON object with the required keys."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=LLM_TEMP,
        max_tokens=LLM_MAX_TOK
    )

    msg = resp.choices[0].message
    text = msg.content

    # fallback: some providers return reasoning_content instead of content
    if (not text or not str(text).strip()) and hasattr(msg, "provider_specific_fields"):
        psf = getattr(msg, "provider_specific_fields", {}) or {}
        text = psf.get("reasoning_content") or psf.get("reasoning")

    log(f"RAW LLM OUTPUT: {repr(text)}")

    if not text or not str(text).strip():
        raise ValueError(f"Empty LLM response | full response: {resp}")

    parsed = extract_json_from_text(text)

    expected_keys = {
        "morphology_description",
        "morphology_code",
        "topography_description",
        "topography_code",
        "reason"
    }
    missing = [k for k in expected_keys if k not in parsed]
    if missing:
        raise ValueError(f"JSON missing keys: {missing} | parsed={parsed}")

    return parsed


# ==========================
# Main pipeline
# ==========================

def main():
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    summaries = load_summary_embeddings(SUMMARY_EMB_JSON)
    morph_items, morph_embs = load_icdo_embedding_list(MORPH_EMB_JSON, kind="Morphology", is_topography=False)
    topo_items,  topo_embs  = load_icdo_embedding_list(TOPO_EMB_JSON, kind="Topography", is_topography=True)

    summaries_use = summaries[:MAX_DOCS] if isinstance(MAX_DOCS, int) and MAX_DOCS > 0 else summaries

    results: List[Dict[str, Any]] = []

    log(f"Starting RAG v3 prediction for {len(summaries_use)} summaries...")

    for idx, rec in enumerate(summaries_use, 1):
        doc_id       = rec.get("doc_id")
        summary_text = rec.get("summary_text", "")
        emb_list     = rec.get("embedding", [])

        try:
            query_emb = np.array(ensure_list_float(emb_list), dtype="float32")
        except Exception as e_emb:
            log(f"[{idx}] doc_id={doc_id}: ERROR converting embedding -> {e_emb}")
            continue

        topo_fams_from_summary = detect_topo_families_from_summary(summary_text)

        morph_cands = get_topk_candidates_hybrid(
            query_emb=query_emb,
            query_text=summary_text,
            icdo_items=morph_items,
            emb_matrix=morph_embs,
            top_k_final=TOP_K_FINAL,
            semi_k=SEMI_K,
            organ_families=None,
            is_topography=False,
            alpha_semantic=ALPHA_SEMANTIC,
            beta_lexical=BETA_LEXICAL
        )

        topo_cands = get_topk_candidates_hybrid(
            query_emb=query_emb,
            query_text=summary_text,
            icdo_items=topo_items,
            emb_matrix=topo_embs,
            top_k_final=TOP_K_FINAL,
            semi_k=SEMI_K,
            organ_families=topo_fams_from_summary,
            is_topography=True,
            alpha_semantic=ALPHA_SEMANTIC,
            beta_lexical=BETA_LEXICAL
        )

        morph_codes = [canonical_morph_code(c["code_raw"]) for c in morph_cands]
        topo_codes  = [topo_family(canonical_topo_code(c["code_raw"])) for c in topo_cands]

        fam_str = ", ".join(sorted(topo_fams_from_summary)) if topo_fams_from_summary else "None"
        log(
            f"[{idx}] doc_id={doc_id}: organ_families={fam_str} | "
            f"Morph candidates={morph_codes[:3]}..., Topo families={topo_codes[:3]}..."
        )

        llm_output: Dict[str, Any] = {}
        parse_ok = False
        error_msg = None

        for attempt in range(2):
            try:
                llm_output = call_llm_for_case(client, summary_text, morph_cands, topo_cands)
                parse_ok = True
                break
            except Exception as e_llm:
                error_msg = str(e_llm)
                log(f"   LLM parse error (attempt {attempt+1}) for doc_id={doc_id}: {e_llm}")

        if parse_ok:
            morph_code_raw = str(llm_output.get("morphology_code", "")).strip()
            topo_code_raw  = str(llm_output.get("topography_code", "")).strip()

            final_rec = {
                "doc_id": doc_id,
                "summary_text": summary_text,
                "pred_morphology_description": llm_output.get("morphology_description", ""),
                "pred_ICD_morphology_code": canonical_morph_code(morph_code_raw) if morph_code_raw else "",
                "pred_topography_description": llm_output.get("topography_description", ""),
                "pred_ICD_topography_code": canonical_topo_code(topo_code_raw) if topo_code_raw else "",
                "selection_cause": llm_output.get("reason", ""),
                "candidate_morphology_codes": morph_codes,
                "candidate_topography_codes": topo_codes,
                "organ_families_detected": sorted(topo_fams_from_summary)
            }
        else:
            final_rec = {
                "doc_id": doc_id,
                "summary_text": summary_text,
                "pred_morphology_description": "",
                "pred_ICD_morphology_code": "",
                "pred_topography_description": "",
                "pred_ICD_topography_code": "",
                "selection_cause": "",
                "candidate_morphology_codes": morph_codes,
                "candidate_topography_codes": topo_codes,
                "organ_families_detected": sorted(topo_fams_from_summary),
                "error": f"LLM parsing failed: {error_msg}"
            }

        results.append(final_rec)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    log(f"[OK] Saved {len(results)} predictions to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()