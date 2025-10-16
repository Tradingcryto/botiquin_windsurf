
# Botiquin Fremap V3 (TF‑IDF + LLM Reranker)

Script de Python per fer **cerca estructurada** entre la licitació (`licitacionfremapsample.xls/xlsx`) i el catàleg (`bbdd_2023.xls/xlsx` o `bbdd_2022.xlsx`).

- **Recall primari**: TF‑IDF (cosine) n‑grams (1–2) per treure **top‑K candidats**.
- **Precisió**: *Reranking* opcional amb **OpenAI** (Chat Completions) per escollir **el millor** i estimar una `confidence` 0–1.
- **Llindar**: `--threshold` (ex. `0.70`) per acceptar/rebutjar el *match*.
- **Sortida**: Excel amb pestanyes **Resultados** i **Resumen**. Afegeix **Codigo BS / Descripcion BS / Nombre proveedor** quan hi ha *match*.

## Instal·lació

Recomanat Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

## Variables d'entorn (si uses LLM)

Defineix la teva clau:

```bash
export OPENAI_API_KEY="sk-..."
# (opcional)
export OPENAI_MODEL="gpt-4o-mini"
```

## Ús

```bash
python main.py       --licitacion /path/licitacionfremapsample.xlsx       --bbdd /path/bbdd_2023.xlsx       --output resultado_botiquin.xlsx       --top-k 5       --threshold 0.70
```

- Afegeix `--no-llm` si vols anar només amb TF‑IDF.
- Si el catàleg és `bbdd_2022.xlsx`, simplement apunta `--bbdd` al fitxer corresponent.

## Columnes esperades

**Licitació** (n'hi ha prou amb almenys una d'aquestes):  
- `Artículo` **o** `Descripcion del articulo` **o** `Producto`

**Catàleg (BBDD)** (necessàries):  
- `Descripción Proveedor`, `Descripción BS`, `Caracteristicas técnicas` *(o `Caracteristicas tecnicas`)*

## Resultats

La pestanya **Resultados** afegeix, si hi ha *match*:
- `Codigo BS`, `Descripcion BS`, `Nombre proveedor`
- `Match TFIDF`, `MATCH_method` (LLM/TFIDF), `MATCH_confidence`, `MATCH_reason`
- Camps de context: `Matched Descripción Proveedor`, `Matched Caracteristicas`

La pestanya **Resumen** aporta KPI del procés.

## Notes

- El *reranker* LLM retorna JSON amb `choice`, `confidence` i `reason` (ES).  
- Si el LLM falla o no està configurat, s'usa TF‑IDF pur.
- Ajusta `--top-k` i `--threshold` segons *recall/precision* desitjats.
