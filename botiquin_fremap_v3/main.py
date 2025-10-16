
import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

def normalize_text(s: Any) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    return " ".join(s.strip().lower().replace("\n", " ").replace("\r", " ").split())

def build_catalog_text(row: pd.Series) -> str:
    parts = [
        row.get("Descripción Proveedor", ""),
        row.get("Descripción BS", ""),
        row.get("Caracteristicas técnicas", ""),
        row.get("Caracteristicas tecnicas", ""),
    ]
    joined = " | ".join([p for p in parts if isinstance(p, str)])
    return normalize_text(joined)

def build_query_text(row: pd.Series) -> str:
    parts = [
        row.get("Artículo", ""),
        row.get("Descripcion del articulo", ""),
        row.get("Producto", ""),
    ]
    joined = " | ".join([p for p in parts if isinstance(p, str)])
    return normalize_text(joined)

@dataclass
class MatchResult:
    idx_catalog: int
    tfidf_score: float

def topk_by_tfidf(queries: List[str], catalog: List[str], top_k: int = 5):
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    corpus = catalog + queries
    X = vectorizer.fit_transform(corpus)
    X_catalog = X[:len(catalog)]
    X_queries = X[len(catalog):]
    sims = cosine_similarity(X_queries, X_catalog)
    topk = []
    for i in range(sims.shape[0]):
        scores = sims[i]
        idxs = np.argsort(-scores)[:top_k]
        topk.append([MatchResult(int(idx), float(scores[idx])) for idx in idxs])
    return topk

LLM_SYSTEM = (
    "You help match medical/clinical supply items from a tender line to a catalog.\n"
    "You are given: (1) the tender item description; (2) up to K candidate catalog items.\n"
    "Pick the single best candidate or say 'none' if none are good enough.\n"
    "Return a compact JSON with fields: choice ('none' or index number starting at 1), "
    "confidence (0-1), and a short reason in Spanish.\n"
    "Penalize mismatched format/size/volume/units and incompatible purpose. Consider product nuances in Spanish."
)

def call_llm_rerank(client, model: str, query: str, candidates: List[Dict[str, str]], temperature: float = 0.0) -> Dict[str, Any]:
    numbered = []
    for i, c in enumerate(candidates, start=1):
        block = (
            f"""{i}. Descripción Proveedor: {c.get('Descripción Proveedor','')}
   Descripción BS: {c.get('Descripción BS','')}
   Caracteristicas técnicas: {c.get('Caracteristicas técnicas','') or c.get('Caracteristicas tecnicas','')}
   Codigo BS: {c.get('Codigo BS','')}
   Proveedor: {c.get('Proveedor','')}
   Nombre proveedor: {c.get('Nombre proveedor','')}"""
        )
        numbered.append(block)
    user = (
        f"""TENDER ITEM:
{query}

CANDIDATES:
{chr(10).join(numbered)}

Respond ONLY with JSON like:
{{"choice": 1, "confidence": 0.82, "reason": "..."}} or {{"choice": "none", "confidence": 0.35, "reason": "..."}}
"""
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role":"system","content": LLM_SYSTEM},
            {"role":"user","content": user}
        ]
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
        return data
    except Exception:
        return {"choice":"none","confidence":0.0,"reason":"Failed to parse JSON from model."}

def ensure_required_columns(df: pd.DataFrame, required_any_of: List[List[str]], label: str):
    cols_lower = {c.lower(): c for c in df.columns}
    for group in required_any_of:
        if all(col.lower() in cols_lower for col in group):
            return
    groups_str = " OR ".join([" + ".join(g) for g in required_any_of])
    raise ValueError(f"{label}: missing required columns. Needs at least one of these sets: {groups_str}. Found: {list(df.columns)}")

def main():
    parser = argparse.ArgumentParser(description="Botiquin Fremap - structured search and LLM rerank")
    parser.add_argument("--licitacion", required=True, help="Path to licitacionfremapsample.xls/xlsx")
    parser.add_argument("--bbdd", required=True, help="Path to bbdd_2023.xls/xlsx (or 2022)")
    parser.add_argument("--output", default="resultado_botiquin.xlsx", help="Output Excel path")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K TFIDF candidates to send to LLM")
    parser.add_argument("--threshold", type=float, default=0.7, help="Acceptance threshold for LLM confidence (0-1)")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL","gpt-4o-mini"), help="OpenAI model for reranking")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM reranking and rely on TFIDF only")
    args = parser.parse_args()

    lic = pd.read_excel(args.licitacion)
    cat = pd.read_excel(args.bbdd)

    ensure_required_columns(
        lic,
        [["Artículo"], ["Descripcion del articulo"], ["Producto"]],
        "Licitación"
    )
    ensure_required_columns(
        cat,
        [["Descripción Proveedor","Descripción BS","Caracteristicas técnicas"],
         ["Descripción Proveedor","Descripción BS","Caracteristicas tecnicas"]],
        "BBDD Catalog"
    )

    lic = lic.copy()
    cat = cat.copy()
    lic["__query_text__"] = lic.apply(build_query_text, axis=1)
    cat["__catalog_text__"] = cat.apply(build_catalog_text, axis=1)

    queries = lic["__query_text__"].tolist()
    catalog = cat["__catalog_text__"].tolist()
    topk = topk_by_tfidf(queries, catalog, top_k=args.top_k)

    client = None
    if not args.no_llm:
        if not OPENAI_AVAILABLE:
            print("OpenAI SDK not available. Install `openai>=1.0.0` or use --no-llm.")
            args.no_llm = True
        else:
            try:
                client = OpenAI()
            except Exception as e:
                print(f"OpenAI client init failed: {e}. Falling back to --no-llm.")
                args.no_llm = True

    results_rows = []
    for i, row in tqdm(lic.iterrows(), total=len(lic), desc="Matching"):
        query_text = row["__query_text__"]
        candidates_info = []
        for mr in topk[i]:
            c_row = cat.iloc[mr.idx_catalog]
            c_dict = {
                "Descripción Proveedor": c_row.get("Descripción Proveedor",""),
                "Descripción BS": c_row.get("Descripción BS",""),
                "Caracteristicas técnicas": c_row.get("Caracteristicas técnicas","") if "Caracteristicas técnicas" in c_row else c_row.get("Caracteristicas tecnicas",""),
                "Codigo BS": c_row.get("Codigo BS",""),
                "Proveedor": c_row.get("Proveedor",""),
                "Nombre proveedor": c_row.get("Nombre proveedor",""),
                "_tfidf_score": mr.tfidf_score,
                "_row_index": mr.idx_catalog,
            }
            candidates_info.append(c_dict)

        best_idx = 0
        best_conf = candidates_info[0]["_tfidf_score"] if candidates_info else 0.0
        best_reason = "Seleccionado por similitud TF-IDF."
        used_llm = False

        if client is not None and candidates_info:
            try:
                llm_resp = call_llm_rerank(client, args.model, query_text, candidates_info, temperature=0.0)
                used_llm = True
                if isinstance(llm_resp.get("choice"), int) and 1 <= llm_resp["choice"] <= len(candidates_info):
                    best_idx = llm_resp["choice"] - 1
                elif isinstance(llm_resp.get("choice"), str) and llm_resp["choice"].lower() == "none":
                    best_idx = None
                best_conf = float(llm_resp.get("confidence", 0.0))
                best_reason = str(llm_resp.get("reason", "")).strip() or "LLM decidió."
            except Exception as e:
                used_llm = False
                best_idx = 0
                best_conf = candidates_info[0]["_tfidf_score"]
                best_reason = f"Fallo LLM, uso TF-IDF. Error: {e}"

        passes = False
        if used_llm:
            passes = (best_idx is not None) and (best_conf >= args.threshold)
        else:
            passes = (best_conf >= args.threshold)

        if passes and best_idx is not None:
            chosen = candidates_info[best_idx]
            out = dict(row)
            out.update({
                "MATCH_found": True,
                "MATCH_method": "LLM" if used_llm else "TFIDF",
                "MATCH_confidence": round(best_conf, 4),
                "MATCH_reason": best_reason,
                "Codigo BS": chosen.get("Codigo BS",""),
                "Descripcion BS": chosen.get("Descripción BS",""),
                "Nombre proveedor": chosen.get("Nombre proveedor",""),
                "Match TFIDF": round(chosen.get("_tfidf_score", 0.0), 4),
                "Matched Descripción Proveedor": chosen.get("Descripción Proveedor",""),
                "Matched Caracteristicas": chosen.get("Caracteristicas técnicas",""),
            })
        else:
            out = dict(row)
            out.update({
                "MATCH_found": False,
                "MATCH_method": "LLM" if used_llm else "TFIDF",
                "MATCH_confidence": round(best_conf, 4),
                "MATCH_reason": best_reason or ("Por debajo del umbral" if used_llm else "Similitud TF-IDF por debajo del umbral"),
                "Codigo BS": "",
                "Descripcion BS": "",
                "Nombre proveedor": "",
                "Match TFIDF": round(candidates_info[0].get("_tfidf_score", 0.0), 4) if candidates_info else 0.0,
                "Matched Descripción Proveedor": "",
                "Matched Caracteristicas": "",
            })
        results_rows.append(out)

    result_df = pd.DataFrame(results_rows)
    total = len(result_df)
    matched = int(result_df["MATCH_found"].sum())
    summary = pd.DataFrame([{
        "Total líneas licitación": total,
        "Con match": matched,
        "Sin match": total - matched,
        "Umbral": args.threshold,
        "Top-K": args.top_k,
        "Modelo": args.model if client is not None else "N/A",
        "Método por defecto si LLM falla": "TF-IDF cosine (1-2 grams)"
    }])

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Resultados")
        summary.to_excel(writer, index=False, sheet_name="Resumen")

    print(f"✅ Hecho. Guardado en: {args.output}")

if __name__ == "__main__":
    main()
