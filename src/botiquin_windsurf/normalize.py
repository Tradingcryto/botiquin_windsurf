"""
Text normalization module for cleaning and standardizing product descriptions.
"""

import re
from typing import Dict, Optional, Tuple, List
from unidecode import unidecode


# Default unit mappings
DEFAULT_UNIT_MAP = {
    'μm': 'um',
    'µm': 'um',
    'ml': 'ml',
    'mL': 'ml',
    'ML': 'ml',
    'cm': 'cm',
    'CM': 'cm',
    'mm': 'mm',
    'MM': 'mm',
    'kg': 'kg',
    'KG': 'kg',
    'gr': 'g',
    'g': 'g',
    'G': 'g',
    'l': 'l',
    'L': 'l',
    'mg': 'mg',
    'MG': 'mg',
    'mcg': 'mcg',
    'MCG': 'mcg',
    'ui': 'ui',
    'UI': 'ui',
    'u.i.': 'ui',
    'U.I.': 'ui',
}

# Default abbreviation expansions
DEFAULT_ABBR_MAP = {
    'c/': 'con ',
    's/': 'sin ',
    'paq.': 'paquete',
    'paq': 'paquete',
    'env.': 'envase',
    'env': 'envase',
    'bot.': 'botella',
    'bot': 'botella',
    'amp.': 'ampolla',
    'amp': 'ampolla',
    'comp.': 'comprimido',
    'comp': 'comprimido',
    'caps.': 'capsula',
    'caps': 'capsula',
    'sol.': 'solucion',
    'sol': 'solucion',
    'susp.': 'suspension',
    'susp': 'suspension',
    'iny.': 'inyectable',
    'iny': 'inyectable',
    'tab.': 'tableta',
    'tab': 'tableta',
    'caj.': 'caja',
    'caj': 'caja',
    'und.': 'unidad',
    'und': 'unidad',
    'unid.': 'unidad',
    'unid': 'unidad',
    'pza.': 'pieza',
    'pza': 'pieza',
    'fte.': 'frasco',
    'fte': 'frasco',
    'fco.': 'frasco',
    'fco': 'frasco',
}


def normalize_text(
    text: str,
    remove_accents: bool = True,
    lowercase: bool = True,
    normalize_whitespace: bool = True,
    normalize_decimals: bool = True,
    decimal_separator: str = '.'
) -> str:
    """
    Normalize text by removing accents, converting to lowercase, and cleaning whitespace.
    
    Args:
        text: Input text to normalize.
        remove_accents: If True, removes accents and diacritics.
        lowercase: If True, converts text to lowercase.
        normalize_whitespace: If True, normalizes multiple spaces to single space.
        normalize_decimals: If True, converts decimal separators (comma to period).
        decimal_separator: Target decimal separator (default: '.').
    
    Returns:
        Normalized text.
    
    Examples:
        >>> normalize_text("Aguja 25G x 1½\"")
        'aguja 25g x 1.5"'
        >>> normalize_text("Solución 0,9%")
        'solucion 0.9%'
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    result = text
    
    # Remove accents
    if remove_accents:
        result = unidecode(result)
    
    # Convert to lowercase
    if lowercase:
        result = result.lower()
    
    # Normalize decimal separators (comma to period)
    if normalize_decimals:
        # Match numbers with comma as decimal separator (e.g., "0,9" or "1,5")
        result = re.sub(r'(\d+),(\d+)', rf'\1{decimal_separator}\2', result)
    
    # Normalize whitespace
    if normalize_whitespace:
        result = re.sub(r'\s+', ' ', result)
        result = result.strip()
    
    return result


def apply_unit_map(
    text: str,
    unit_map: Optional[Dict[str, str]] = None,
    preserve_case: bool = False
) -> str:
    """
    Apply unit mapping to standardize measurement units.
    
    Args:
        text: Input text containing units.
        unit_map: Dictionary mapping source units to target units.
                 If None, uses DEFAULT_UNIT_MAP.
        preserve_case: If True, preserves original case of surrounding text.
    
    Returns:
        Text with standardized units.
    
    Examples:
        >>> apply_unit_map("Filtro 0.22 μm")
        'Filtro 0.22 um'
        >>> apply_unit_map("Jeringa 10 mL")
        'Jeringa 10 ml'
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    if unit_map is None:
        unit_map = DEFAULT_UNIT_MAP
    
    result = text
    
    # Sort by length (longest first) to avoid partial replacements
    sorted_units = sorted(unit_map.items(), key=lambda x: len(x[0]), reverse=True)
    
    for source_unit, target_unit in sorted_units:
        # Use word boundaries for units that are letters only
        if source_unit.isalpha():
            pattern = r'\b' + re.escape(source_unit) + r'\b'
        else:
            pattern = re.escape(source_unit)
        
        result = re.sub(pattern, target_unit, result)
    
    return result


def expand_abbr(
    text: str,
    abbr_map: Optional[Dict[str, str]] = None,
    preserve_case: bool = False
) -> str:
    """
    Expand abbreviations to full words.
    
    Args:
        text: Input text containing abbreviations.
        abbr_map: Dictionary mapping abbreviations to full words.
                 If None, uses DEFAULT_ABBR_MAP.
        preserve_case: If True, attempts to preserve case of original abbreviation.
    
    Returns:
        Text with expanded abbreviations.
    
    Examples:
        >>> expand_abbr("Jeringa c/ aguja")
        'Jeringa con aguja'
        >>> expand_abbr("Sol. iny. 10 ml")
        'Solucion inyectable 10 ml'
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    if abbr_map is None:
        abbr_map = DEFAULT_ABBR_MAP
    
    result = text
    
    # Sort by length (longest first) to avoid partial replacements
    sorted_abbrs = sorted(abbr_map.items(), key=lambda x: len(x[0]), reverse=True)
    
    for abbr, expansion in sorted_abbrs:
        # Create pattern with word boundaries
        # Handle abbreviations with special characters like 'c/' or 's/'
        if re.search(r'[^\w\s]', abbr):
            pattern = re.escape(abbr)
        else:
            pattern = r'\b' + re.escape(abbr) + r'\b'
        
        if preserve_case:
            # Case-insensitive replacement preserving original case
            def replace_func(match):
                original = match.group(0)
                if original.isupper():
                    return expansion.upper()
                elif original[0].isupper():
                    return expansion.capitalize()
                return expansion
            
            result = re.sub(pattern, replace_func, result, flags=re.IGNORECASE)
        else:
            result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
    
    return result


def extract_codes_sizes(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract product codes and sizes from text.
    
    Args:
        text: Input text containing codes and sizes.
    
    Returns:
        Tuple of (codes, sizes) where:
        - codes: List of extracted product codes (alphanumeric patterns)
        - sizes: List of extracted sizes (numbers with units)
    
    Examples:
        >>> extract_codes_sizes("Aguja BD 25G x 1.5 pulgadas REF: 305122")
        (['305122', '25G'], ['25G', '1.5'])
        >>> extract_codes_sizes("Jeringa 10 ml Luer Lock")
        ([], ['10'])
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    codes = []
    sizes = []
    
    # Extract product codes (common patterns)
    # Pattern 1: REF:, Ref:, ref: followed by alphanumeric
    ref_pattern = r'(?:REF|Ref|ref)[:\s]+([A-Z0-9\-]+)'
    codes.extend(re.findall(ref_pattern, text, re.IGNORECASE))
    
    # Pattern 2: Standalone alphanumeric codes (e.g., "305122", "BD-123")
    code_pattern = r'\b([A-Z]{2,}[0-9]+|[0-9]{5,}|[A-Z0-9]+-[A-Z0-9]+)\b'
    codes.extend(re.findall(code_pattern, text))
    
    # Pattern 3: Gauge sizes (e.g., "25G", "18G")
    gauge_pattern = r'\b(\d{1,2}G)\b'
    gauge_codes = re.findall(gauge_pattern, text, re.IGNORECASE)
    codes.extend(gauge_codes)
    
    # Extract sizes (numbers with optional units)
    # Pattern 1: Numbers with units (e.g., "10 ml", "25G", "0.22 um")
    size_with_unit_pattern = r'\b(\d+(?:[.,]\d+)?)\s*([a-zA-Zμµ]+)\b'
    size_matches = re.findall(size_with_unit_pattern, text)
    sizes.extend([f"{num}{unit}" if not re.search(r'\s', text[text.find(num):text.find(num)+len(num)+len(unit)+2]) 
                  else f"{num} {unit}" 
                  for num, unit in size_matches])
    
    # Pattern 2: Standalone numbers that might be sizes
    standalone_number_pattern = r'\b(\d+(?:[.,]\d+)?)\b'
    standalone_numbers = re.findall(standalone_number_pattern, text)
    sizes.extend([num for num in standalone_numbers if num not in [s.split()[0] for s in sizes]])
    
    # Remove duplicates while preserving order
    codes = list(dict.fromkeys(codes))
    sizes = list(dict.fromkeys(sizes))
    
    return codes, sizes


def normalize_product_description(
    text: str,
    remove_accents: bool = True,
    expand_abbreviations: bool = True,
    standardize_units: bool = True,
    unit_map: Optional[Dict[str, str]] = None,
    abbr_map: Optional[Dict[str, str]] = None
) -> str:
    """
    Apply full normalization pipeline to product description.
    
    Args:
        text: Input product description.
        remove_accents: If True, removes accents.
        expand_abbreviations: If True, expands abbreviations.
        standardize_units: If True, standardizes measurement units.
        unit_map: Custom unit mapping dictionary.
        abbr_map: Custom abbreviation mapping dictionary.
    
    Returns:
        Fully normalized product description.
    
    Examples:
        >>> normalize_product_description("Sol. iny. 0,9% c/ 10 mL")
        'solucion inyectable 0.9% con 10 ml'
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    result = text
    
    # Step 1: Expand abbreviations (before lowercasing)
    if expand_abbreviations:
        result = expand_abbr(result, abbr_map)
    
    # Step 2: Standardize units
    if standardize_units:
        result = apply_unit_map(result, unit_map)
    
    # Step 3: Normalize text (accents, case, whitespace, decimals)
    result = normalize_text(result, remove_accents=remove_accents)
    
    return result
