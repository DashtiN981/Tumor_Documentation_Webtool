# evaluate_icdo_rag_anymatch.py
# ====================================================
# Evaluate ICD-O RAG predictions against ground truth
# using your current GT file:
#   NLP-Test_Medi-Daten_bearbeitet.xlsx
#
# Mapping:
#   - predictions JSON: has doc_id (string)
#   - GT Excel: has real_id (string)
#     + cat_icdo3morph  (one or more rows per real_id)
#     + cat_icdo3topo
#
# For each real_id, we aggregate ALL ICD-O codes and
# compute these metrics:
#
#   - Morphology Accuracy (primary exact)
#   - Morphology Accuracy (ANY-match)
#   - Topography Accuracy (primary exact, full code)
#   - Topography Accuracy (primary exact, family Cxx)
#   - Topography Accuracy (ANY-match, full & family)
# ====================================================

import json
from pathlib import Path
from typing import List

import pandas as pd


# ---------- CONFIGURE THESE PATHS ----------
PRED_JSON_PATH = Path("./results/ICDO_RAG_predictions.json")
GT_PATH        = Path("./data/Patient_ground_truth.xlsx")
GT_SHEET_NAME  = "Sheet1" 
# ------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------- Canonicalization helpers ----------

def canon_morph(code: str) -> str:
    """
    Canonicalize ICD-O Morph code.
    Examples:
      "8935/3_B3Y" -> "8935/3"
      "8000/0"     -> "8000/0"
    """
    if not isinstance(code, str):
        code = str(code or "").strip()
    if not code:
        return ""
    token = code.split()[0]      # "8935/3_B3Y" -> "8935/3_B3Y"
    token = token.split("_")[0]  # "8935/3_B3Y" -> "8935/3"
    return token.strip()


def canon_topo_full(code: str) -> str:
    """
    Canonicalize ICD-O Topography FULL code.
    Examples:
      "C61 Pr_VWO"  -> "C61"
      "C52.9 _I/b"  -> "C52.9"
    """
    if not isinstance(code, str):
        code = str(code or "").strip()
    if not code:
        return ""
    token = code.split()[0]      # "C52.9 _I/b" -> "C52.9"
    return token.strip()


def canon_topo_family(code: str) -> str:
    """
    Canonicalize ICD-O Topography to FAMILY level (first 3 characters).
    Examples:
      "C61.9" -> "C61"
      "C61"   -> "C61"
      "C25"   -> "C25"
    """
    full = canon_topo_full(code)
    if len(full) >= 3:
        return full[:3]
    return full


# ---------- Loading predictions ----------

def load_predictions(path: Path) -> pd.DataFrame:
    """
    Load RAG predictions JSON and canonicalize predicted codes.
    Expected keys in JSON:
      - doc_id
      - pred_ICD_morphology_code
      - pred_ICD_topography_code
    """
    log(f"[INFO] Loading predictions from: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    required = ["doc_id", "pred_ICD_morphology_code", "pred_ICD_topography_code"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Predictions JSON missing columns: {missing}")

    df["pred_ICD_morphology_code_full"] = df["pred_ICD_morphology_code"].apply(canon_morph)
    df["pred_ICD_topography_code_full"] = df["pred_ICD_topography_code"].apply(canon_topo_full)
    df["pred_ICD_topography_code_family"] = df["pred_ICD_topography_code"].apply(canon_topo_family)

    return df


# ---------- Loading & aggregating ground truth ----------

def load_ground_truth(path: Path, sheet_name="Sheet1") -> pd.DataFrame:
    """
    Load GT Excel and aggregate all ICD-O codes per real_id.

    Expected columns in GT file:
      - real_id
      - cat_icdo3morph
      - cat_icdo3topo

    For each real_id we build:
      - gt_morph_all_full:   list of unique canonical Morph codes
      - gt_topo_all_full:    list of unique canonical Topo codes (full)
      - gt_topo_all_family:  list of unique canonical Topo family codes
    """
    log(f"[INFO] Loading ground truth from: {path}")
    df = pd.read_excel(path, sheet_name=sheet_name)

    needed = ["real_id", "cat_icdo3morph", "cat_icdo3topo"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"GT file missing columns: {missing}")

    # aggregate per real_id (there may be multiple rows per patient)
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


# ---------- Metrics ----------

def compute_metrics(df_eval: pd.DataFrame) -> None:
    """
    Compute and print metrics:

      - Morph primary exact:  pred == first GT Morph code
      - Morph ANY-match:      pred in ALL GT Morph codes

      - Topo primary exact (full):   pred_full == first GT Topo full
      - Topo primary exact (family): pred_family == first GT Topo family
      - Topo ANY-match (full):       pred_full in ALL GT Topo full codes
      - Topo ANY-match (family):     pred_family in ALL GT Topo family codes
    """
    n = len(df_eval)
    if n == 0:
        print("[WARN] No rows to evaluate.")
        return

    morph_primary = []
    morph_any = []
    topo_primary_full = []
    topo_primary_family = []
    topo_any_full = []
    topo_any_family = []

    for _, row in df_eval.iterrows():
        pm = row["pred_ICD_morphology_code_full"]
        pt_full = row["pred_ICD_topography_code_full"]
        pt_fam = row["pred_ICD_topography_code_family"]

        gt_m_all = row["gt_morph_all_full"] or []
        gt_t_all_full = row["gt_topo_all_full"] or []
        gt_t_all_fam = row["gt_topo_all_family"] or []

        gt_m_primary = gt_m_all[0] if gt_m_all else ""
        gt_t_primary_full = gt_t_all_full[0] if gt_t_all_full else ""
        gt_t_primary_fam = gt_t_all_fam[0] if gt_t_all_fam else ""

        # Morphology
        morph_primary.append(bool(pm) and pm == gt_m_primary)
        morph_any.append(bool(pm) and (pm in gt_m_all))

        # Topography
        topo_primary_full.append(bool(pt_full) and pt_full == gt_t_primary_full)
        topo_primary_family.append(bool(pt_fam) and pt_fam == gt_t_primary_fam)
        topo_any_full.append(bool(pt_full) and (pt_full in gt_t_all_full))
        topo_any_family.append(bool(pt_fam) and (pt_fam in gt_t_all_fam))

    import numpy as np

    morph_primary_acc = float(np.mean(morph_primary))
    morph_any_acc = float(np.mean(morph_any))
    topo_primary_full_acc = float(np.mean(topo_primary_full))
    topo_primary_family_acc = float(np.mean(topo_primary_family))
    topo_any_full_acc = float(np.mean(topo_any_full))
    topo_any_family_acc = float(np.mean(topo_any_family))

    print("\n===== ICD-O RAG Evaluation (refined) =====")
    print(f"Total patients (merged): {n}")
    print("------------------------------------------")
    print(f"Morphology Accuracy (primary exact):       {morph_primary_acc:.3f}")
    print(f"Morphology Accuracy (ANY-match):           {morph_any_acc:.3f}")
    print(f"Topography Accuracy (primary exact, full): {topo_primary_full_acc:.3f}")
    print(f"Topography Accuracy (primary exact, fam):  {topo_primary_family_acc:.3f}")
    print(f"Topography Accuracy (ANY-match, full):     {topo_any_full_acc:.3f}")
    print(f"Topography Accuracy (ANY-match, fam):      {topo_any_family_acc:.3f}")
    print("==========================================\n")


# ---------- Main ----------

def main():
    df_pred = load_predictions(PRED_JSON_PATH)
    df_gt   = load_ground_truth(GT_PATH, sheet_name=GT_SHEET_NAME)

    # merge: doc_id (pred) <-> real_id (GT)
    df_eval = pd.merge(df_pred, df_gt, left_on="doc_id", right_on="real_id", how="inner")
    log(f"[INFO] Merged rows (predictions ∩ GT): {len(df_eval)}")

    compute_metrics(df_eval)


if __name__ == "__main__":
    main()
