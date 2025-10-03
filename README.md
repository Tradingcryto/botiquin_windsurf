# Botiquin Windsurf

A Python project for data processing and analysis.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/botiquin_windsurf.git
   cd botiquin_windsurf
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

   For development with additional tools:
   ```bash
   pip install -e ".[dev]"
   ```

## Project Structure

```
botiquin_windsurf/
├── src/
│   └── botiquin_windsurf/
│       └── __init__.py
├── tests/
├── data/
│   ├── input/
│   └── output/
├── pyproject.toml
└── README.md
```

## Usage

1. Place your input files in the `data/input/` directory.
2. Run your Python scripts from the project root.
3. Output files will be saved in the `data/output/` directory.

## Development

- Format code with Black:
  ```bash
  black .
  ```

- Run tests:
  ```bash
  pytest
  ```

## License

MIT
