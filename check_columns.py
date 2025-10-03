import pandas as pd

# Read the tender file
try:
    df = pd.read_excel('data/input/Licitacion.xlsx')
    print("Tender file columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col!r} (length: {len(col)})")
    
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    
except Exception as e:
    print(f"Error: {str(e)}")
