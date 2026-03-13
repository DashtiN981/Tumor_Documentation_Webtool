# evaluate_icdo_retrieval_recall.py
# ============================================
# Measure pure retrieval quality (Recall@K)
# using:
#   - candidate_morphology_codes
#   - candidate_topography_codes
#
# For each patient (doc_id / real_id), we check:
#   - Does any GT Morph code appear in candidate_morphology_codes?
#   - Does any GT Topo FULL code appear in candidate_topography_codes?
#   - Does any GT Topo FAMILY (Cxx) appear in candidate_topography_codes?
# ============================================

import json
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

PRED_JSON_PATH = Path("/home/naghmedashti/NCT_ICDOTOPO_MORPHO/results/ICDO_RAG_predictions_v2.json")
GT_PATH        = Path("/home/naghmedashti/NCT_ICDOTOPO_MORPHO/data/NLP-Test_Medi-Daten_bearbeitet.xlsx")
GT_SHEET_NAME  = "Sheet1"


def log(msg: str) -> None:
    print(msg, flush=True)


def canon_morph(code: str) -> str:
    if not isinstance(code, str):
        code = str(code or "").strip()
    if not code:
        return ""
    token = code.split()[0]
    token = token.split("_")[0]
    return token.strip()


def canon_topo_full(code: str) -> str:
    if not isinstance(code, str):
        code = str(code or "").strip()
    if not code:
        return ""
    token = code.split()[0]
    return token.strip()


def canon_topo_family(code: str) -> str:
    full = canon_topo_full(code)
    if len(full) >= 3:
        return full[:3]
    return full


def load_predictions(path: Path) -> pd.DataFrame:
    log(f"[INFO] Loading predictions from: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    required = ["doc_id", "candidate_morphology_codes", "candidate_topography_codes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Predictions JSON missing columns: {missing}")

    # candidate_*_codes در اسکریپت RAG از قبل canonical شده‌اند،
    # ولی برای اطمینان می‌توانیم باز هم canonical کنیم.
    df["cand_morph"] = df["candidate_morphology_codes"].apply(
        lambda lst: [canon_morph(x) for x in (lst or [])]
    )
    df["cand_topo_full"] = df["candidate_topography_codes"].apply(
        lambda lst: [canon_topo_full(x) for x in (lst or [])]
    )
    df["cand_topo_family"] = df["candidate_topography_codes"].apply(
        lambda lst: [canon_topo_family(x) for x in (lst or [])]
    )

    return df


def load_ground_truth(path: Path, sheet_name="Sheet1") -> pd.DataFrame:
    log(f"[INFO] Loading ground truth from: {path}")
    df = pd.read_excel(path, sheet_name=sheet_name)

    needed = ["real_id", "cat_icdo3morph", "cat_icdo3topo"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"GT file missing columns: {missing}")

    grouped = df.groupby("real_id").agg({
        "cat_icdo3morph": list,
        "cat_icdo3topo": list
    }).reset_index()

    def uniq(lst: List[str]) -> List[str]:
        return sorted(set(lst))

    grouped["gt_morph_all_full"] = grouped["cat_icdo3morph"].apply(
        lambda lst: uniq([canon_morph(x) for x in lst if str(x).strip()])
    )
    grouped["gt_topo_all_full"] = grouped["cat_icdo3topo"].apply(
        lambda lst: uniq([canon_topo_full(x) for x in lst if str(x).strip()])
    )
    grouped["gt_topo_all_family"] = grouped["cat_icdo3topo"].apply(
        lambda lst: uniq([canon_topo_family(x) for x in lst if str(x).strip()])
    )

    return grouped


def main():
    df_pred = load_predictions(PRED_JSON_PATH)
    df_gt   = load_ground_truth(GT_PATH, sheet_name=GT_SHEET_NAME)

    df = pd.merge(df_pred, df_gt, left_on="doc_id", right_on="real_id", how="inner")
    log(f"[INFO] Merged rows: {len(df)}")

    morph_hit = []
    topo_hit_full = []
    topo_hit_family = []

    for _, row in df.iterrows():
        cand_morph = row["cand_morph"]
        cand_topo_full = row["cand_topo_full"]
        cand_topo_family = row["cand_topo_family"]

        gt_m_all = row["gt_morph_all_full"] or []
        gt_t_all_full = row["gt_topo_all_full"] or []
        gt_t_all_family = row["gt_topo_all_family"] or []

        # Morph: hit if ANY GT morph in candidate list
        morph_hit.append(any(g in cand_morph for g in gt_m_all))

        # Topo full: hit if ANY GT topo full in candidate list
        topo_hit_full.append(any(g in cand_topo_full for g in gt_t_all_full))

        # Topo family: hit if ANY GT topo family in candidate list
        topo_hit_family.append(any(g in cand_topo_family for g in gt_t_all_family))

    morph_recall = float(np.mean(morph_hit))
    topo_recall_full = float(np.mean(topo_hit_full))
    topo_recall_family = float(np.mean(topo_hit_family))

    print("\n===== ICD-O Retrieval Recall@K (K = len(candidate lists)) =====")
    print(f"N patients: {len(df)}")
    print("--------------------------------------------")
    print(f"Morphology Recall@K (ANY GT in candidates):       {morph_recall:.3f}")
    print(f"Topography Recall@K (full code in candidates):    {topo_recall_full:.3f}")
    print(f"Topography Recall@K (family code in candidates):  {topo_recall_family:.3f}")
    print("=================================================\n")


if __name__ == "__main__":
    main()
