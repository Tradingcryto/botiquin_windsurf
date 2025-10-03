# Installation Guide

## Quick Start

### 1. Install Dependencies

```powershell
# Navigate to project directory
cd "C:\Users\Miquel\OneDrive\Desktop\Empresa\AI\Windsurf Projects\Botiquin Sans\botiquin_windsurf"

# Install the package and dependencies
pip install -e .
```

### 2. Set OpenAI API Key (Optional - for vector search)

```powershell
# Set environment variable for current session
$env:OPENAI_API_KEY = "your-api-key-here"

# Or permanently (requires new terminal)
setx OPENAI_API_KEY "your-api-key-here"
```

### 3. Run Tests

```powershell
# Run all tests
pytest

# Run specific test file
pytest tests/test_normalize.py -v

# Run with coverage
pytest --cov=src/botiquin_windsurf
```

### 4. Run Batch Processing

```powershell
python run_batch.py --catalog data/input/catalogo.xlsx --query data/input/licitacion.xlsx --out data/output/resultado_matching.xlsx
```

## Project Structure

```
botiquin_windsurf/
├── src/
│   └── botiquin_windsurf/
│       ├── __init__.py
│       ├── config.py          # Configuration management
│       ├── io_excel.py        # Excel I/O
│       ├── normalize.py       # Text normalization
│       ├── taxonomy.py        # Category taxonomy
│       ├── router.py          # Category routing
│       ├── index_bm25.py      # BM25 indexing
│       ├── fuse.py            # Result fusion (RRF)
│       ├── search.py          # Search engine
│       └── batch.py           # Batch processing
├── tests/                     # Unit tests
├── data/
│   ├── input/                 # Input files
│   └── output/                # Output files
├── config.yaml                # Configuration file
├── pyproject.toml            # Project metadata
└── run_batch.py              # Batch processing script
```

## Dependencies

Core dependencies:
- pandas: Data manipulation
- openpyxl: Excel file handling
- unidecode: Text normalization
- rank-bm25: BM25 search
- sentence-transformers: Vector embeddings (optional)
- faiss-cpu: Vector search (optional)
- pyyaml: Configuration
- xlsxwriter: Excel output

## Troubleshooting

### ModuleNotFoundError

If you get `ModuleNotFoundError`, install dependencies:
```powershell
pip install pandas openpyxl unidecode langdetect rapidfuzz rank-bm25 pyyaml xlsxwriter
```

### File Not Found

Ensure you're in the correct directory:
```powershell
cd "C:\Users\Miquel\OneDrive\Desktop\Empresa\AI\Windsurf Projects\Botiquin Sans\botiquin_windsurf"
```
