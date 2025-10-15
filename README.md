# botiquin_windsurf

Pipeline en Python per fer matching entre línies de licitació i el catàleg de productes. L'script llegeix els dos Excel, normalitza els textos, cerca candidats amb BM25, fa boost per codis/mides, calcula un percentatge de coincidència i exporta els millors resultats a Excel.

## ✨ Objectiu
Agilitzar la preparació d'ofertes: per a cada línia de la licitació, trobar els millors productes del catàleg amb una puntuació de similitud i un resum d'estadístiques.

## 📁 Estructura del projecte
```
botiquin_windsurf/
├── data/
│   ├── input/
│   │   ├── catalogo.xlsx        # catàleg de productes
│   │   └── Licitacion.xlsx      # fitxer de licitació
│   └── output/                  # resultats generats
├── src/
│   └── botiquin_windsurf/
│       ├── __init__.py
│       ├── batch.py             # processament batch i exportació
│       ├── fuse.py              # fusió/format de resultats
│       ├── index_bm25.py        # índex i cerca BM25
│       ├── io_excel.py          # lectura/escriptura Excel
│       └── search.py            # normalització, exact match, routing categories
├── run_batch.py                 # script CLI principal
├── pyproject.toml
└── README.md
```

## 📦 Requisits
Instal·la com a mínim:
```bash
pip install pandas rank_bm25 openpyxl xlsxwriter
```

Activa l'entorn virtual abans d'instal·lar (recomanat).

## 🛠️ Instal·lació (amb entorn virtual)

### Windows (PowerShell)
```powershell
git clone https://github.com/Tradingcryto/botiquin_windsurf
cd botiquin_windsurf
python -m venv .venv
\.venv\Scripts\Activate.ps1
pip install -e .
pip install -e ".[dev]"   # opcional per eines de desenvolupament
```

### macOS / Linux
```bash
git clone https://github.com/Tradingcryto/botiquin_windsurf
cd botiquin_windsurf
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[dev]"   # opcional per eines de desenvolupament
```

Si no vols instal·lar el paquet en mode editable, simplement instal·la els requisits mínims (veure Requisits).

## 📑 Format esperat dels fitxers d'entrada

### Catàleg (data/input/catalogo.xlsx)
Columnes típiques:
- `product_id` (o es crearà id intern si no existeix)
- `ref_fabricante`
- `supplier`
- `product_name`
- `description`
- `category` (opcional; si no hi és, es fa cerca global)

### Licitació (data/input/Licitacion.xlsx)
Columnes habituals (el codi tria la millor disponible):
- `id` (identificador de línia)
- `product_name` o `description` o `descripcion` o `descripcionproveedor`
  (també accepta variants: "Descripción Proveedor", "Articulo Proveedor", etc.)
- `ref_fabricante` (opcional)

El codi és tolerant i fa auto-mapeig de columnes si el nom no coincideix exactament.

## 🔎 Metodologia de cerca (resum)
1. Normalització de text (minúscules, sense accents, neteja).
2. Extracció de features:
   - codis (ex. 14G, 02031-101)
   - mides (ex. 30 ml, 2.5cmx5m)
3. Exact match si hi ha coincidència clara.
4. Routing per categories (si el catàleg en té); si no, cerca global.
5. BM25 per recuperar candidats rellevants.
6. Fusió i puntuació amb boost per codis/mides; càlcul d'un % de coincidència.
7. Filtratge per llindar (`--threshold`) i top-k resultats per línia.

## ▶️ Execució

### Windows (PowerShell)
```powershell
cd "C:\ruta\al\projecte\botiquin_windsurf"
\.venv\Scripts\Activate.ps1

python .\run_batch.py `
  --catalog "data/input/catalogo.xlsx" `
  --query "data/input/Licitacion.xlsx" `
  --out "data/output/resultat.xlsx" `
  --top-k 3 `
  --threshold 60.0 `
  --verbose
```

### macOS / Linux
```bash
cd /ruta/al/projecte/botiquin_windsurf
source .venv/bin/activate

python run_batch.py \
  --catalog "data/input/catalogo.xlsx" \
  --query "data/input/Licitacion.xlsx" \
  --out "data/output/resultat.xlsx" \
  --top-k 3 \
  --threshold 60.0 \
  --verbose
```

### Paràmetres principals
- `--catalog` Ruta a l'Excel del catàleg
- `--query` Ruta a l'Excel de la licitació
- `--out` Fitxer Excel de sortida
- `--top-k` Nº màxim de candidats per línia (per defecte 5)
- `--threshold` Llindar mínim de coincidència en %
- `--verbose` Activa logging detallat

## 📤 Sortida
Un Excel a `data/output/...` amb fins a `top-k` candidats per cada línia de licitació, incloent:
- `tender_id`, `tender_query`
- `coincidencia_pct`
- Dades del producte del catàleg (`product_id`/`id`, `product_name`, `supplier`, `ref_fabricante`, …)

A consola es mostren estadístiques: processades, amb matches, per sobre del llindar, etc.

## 🧪 Desenvolupament
Format amb Black:
```bash
black .
```

Tests (si n'hi ha):
```bash
pytest
```

## 🐛 Troubleshooting
### No module named 'xlsxwriter'
Instal·la'l:
```bash
pip install xlsxwriter
```

### Errors de longitud / índex en Pandas
El codi ja crea un id únic intern si cal. Assegura't que `catalogo.xlsx` no tingui files totalment buides a les columnes clau (`product_name`, `description`).

### PowerShell i salts de línia
Usa el caràcter `` ` `` (backtick) per trencar línies, com a l'exemple d'execució en Windows.

## 🗺️ Full de ruta (idees)
- Afegir índex vectorial (embeddings) per millorar sinònims.
- Diccionari de sinònims sectorials.
- Ajust fi de pesos/boosts per codis/mides.
- `config.yaml` per mapejar columnes i pesos sense tocar codi.

## 📄 Llicència
MIT
