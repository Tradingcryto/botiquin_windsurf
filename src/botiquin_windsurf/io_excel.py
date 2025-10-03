"""
Excel I/O module for reading catalog and tender files with column mapping.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
from .config import get_config, ConfigError


class ExcelReadError(Exception):
    """Custom exception for Excel reading errors."""
    pass


def read_catalog(
    file_path: str,
    config_path: Optional[str] = None,
    sheet_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Read catalog Excel file and apply column mapping.
    
    Args:
        file_path: Path to the catalogo.xlsx file.
        config_path: Optional path to configuration file.
        sheet_name: Optional sheet name to read. If None, reads the first sheet.
    
    Returns:
        DataFrame with mapped column names (internal names as columns).
    
    Raises:
        ExcelReadError: If file cannot be read or columns are missing.
        ConfigError: If configuration is invalid.
    """
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        raise ExcelReadError(
            f"Catalog file not found: {file_path}\n"
            f"Please ensure the file exists at the specified path."
        )
    
    # Load configuration
    try:
        config = get_config(config_path)
        column_mapping = config.get_catalog_column_mapping()
    except ConfigError as e:
        print(f"ERROR: Configuration error\n{str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Read Excel file
    try:
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        else:
            df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        raise ExcelReadError(
            f"Error reading Excel file: {file_path}\n"
            f"Error: {str(e)}\n"
            f"Please ensure the file is a valid Excel file and not corrupted."
        )
    
    # Validate column mapping
    try:
        config.validate_column_mapping(
            column_mapping,
            df.columns.tolist(),
            "catalog file (catalogo.xlsx)"
        )
    except ConfigError as e:
        print(f"ERROR: Column mapping validation failed\n{str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Apply column mapping (rename columns from Excel names to internal names)
    reverse_mapping = {excel_name: internal_name 
                      for internal_name, excel_name in column_mapping.items()}
    
    # Only rename columns that exist in the mapping
    columns_to_rename = {col: reverse_mapping[col] 
                        for col in df.columns if col in reverse_mapping}
    df = df.rename(columns=columns_to_rename)
    
    return df


def read_tender(
    file_path: str,
    config_path: Optional[str] = None,
    sheet_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Read tender Excel file and apply column mapping.
    
    Args:
        file_path: Path to the licitacion.xlsx file.
        config_path: Optional path to configuration file.
        sheet_name: Optional sheet name to read. If None, reads the first sheet.
    
    Returns:
        DataFrame with mapped column names (internal names as columns).
    
    Raises:
        ExcelReadError: If file cannot be read or columns are missing.
        ConfigError: If configuration is invalid.
    """
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        raise ExcelReadError(
            f"Tender file not found: {file_path}\n"
            f"Please ensure the file exists at the specified path."
        )
    
    # Load configuration
    try:
        config = get_config(config_path)
        column_mapping = config.get_tender_column_mapping()
    except ConfigError as e:
        print(f"ERROR: Configuration error\n{str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Read Excel file
    try:
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        else:
            df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        raise ExcelReadError(
            f"Error reading Excel file: {file_path}\n"
            f"Error: {str(e)}\n"
            f"Please ensure the file is a valid Excel file and not corrupted."
        )
    
    # Validate column mapping
    try:
        config.validate_column_mapping(
            column_mapping,
            df.columns.tolist(),
            "tender file (licitacion.xlsx)"
        )
    except ConfigError as e:
        print(f"ERROR: Column mapping validation failed\n{str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Apply column mapping (rename columns from Excel names to internal names)
    reverse_mapping = {excel_name: internal_name 
                      for internal_name, excel_name in column_mapping.items()}
    
    # Only rename columns that exist in the mapping
    columns_to_rename = {col: reverse_mapping[col] 
                        for col in df.columns if col in reverse_mapping}
    df = df.rename(columns=columns_to_rename)
    
    return df


def write_excel(
    df: pd.DataFrame,
    file_path: str,
    sheet_name: str = "Sheet1",
    index: bool = False
) -> None:
    """
    Write DataFrame to Excel file.
    
    Args:
        df: DataFrame to write.
        file_path: Output file path.
        sheet_name: Name of the sheet.
        index: Whether to write row indices.
    
    Raises:
        ExcelReadError: If file cannot be written.
    """
    file_path = Path(file_path)
    
    # Create parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=index)
    except Exception as e:
        raise ExcelReadError(
            f"Error writing Excel file: {file_path}\n"
            f"Error: {str(e)}"
        )


def get_excel_sheets(file_path: str) -> List[str]:
    """
    Get list of sheet names in an Excel file.
    
    Args:
        file_path: Path to Excel file.
    
    Returns:
        List of sheet names.
    
    Raises:
        ExcelReadError: If file cannot be read.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ExcelReadError(f"File not found: {file_path}")
    
    try:
        xl_file = pd.ExcelFile(file_path, engine='openpyxl')
        return xl_file.sheet_names
    except Exception as e:
        raise ExcelReadError(
            f"Error reading Excel file: {file_path}\n"
            f"Error: {str(e)}"
        )


def preview_excel_columns(file_path: str, sheet_name: Optional[str] = None) -> List[str]:
    """
    Preview column names in an Excel file without loading all data.
    
    Args:
        file_path: Path to Excel file.
        sheet_name: Optional sheet name. If None, reads first sheet.
    
    Returns:
        List of column names.
    
    Raises:
        ExcelReadError: If file cannot be read.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ExcelReadError(f"File not found: {file_path}")
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0, engine='openpyxl')
        return df.columns.tolist()
    except Exception as e:
        raise ExcelReadError(
            f"Error reading Excel file: {file_path}\n"
            f"Error: {str(e)}"
        )
