"""
Unit tests for the normalize module.
"""

import pytest
from botiquin_windsurf.normalize import (
    normalize_text,
    apply_unit_map,
    expand_abbr,
    extract_codes_sizes,
    normalize_product_description
)


class TestNormalizeText:
    """Tests for normalize_text function."""
    
    def test_remove_accents(self):
        """Test accent removal."""
        assert normalize_text("Solución") == "solucion"
        assert normalize_text("Inyección") == "inyeccion"
        assert normalize_text("Estéril") == "esteril"
    
    def test_lowercase(self):
        """Test lowercase conversion."""
        assert normalize_text("AGUJA") == "aguja"
        assert normalize_text("Jeringa") == "jeringa"
        assert normalize_text("MiXeD CaSe") == "mixed case"
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        assert normalize_text("Aguja  25G   x  1.5") == "aguja 25g x 1.5"
        assert normalize_text("  Jeringa  ") == "jeringa"
        assert normalize_text("Multiple\t\ttabs") == "multiple tabs"
    
    def test_normalize_decimals(self):
        """Test decimal separator normalization."""
        assert normalize_text("0,9%") == "0.9%"
        assert normalize_text("1,5 pulgadas") == "1.5 pulgadas"
        assert normalize_text("10,25 ml") == "10.25 ml"
    
    def test_combined_normalization(self):
        """Test combined normalization."""
        assert normalize_text("Solución  0,9%") == "solucion 0.9%"
        assert normalize_text("AGUJA 25G x 1½\"") == "aguja 25g x 1.5\""
    
    def test_preserve_options(self):
        """Test with preservation options."""
        result = normalize_text("Solución", remove_accents=False, lowercase=False)
        assert result == "Solución"
        
        result = normalize_text("TEST", lowercase=False)
        assert result == "TEST"
    
    def test_none_and_empty(self):
        """Test handling of None and empty strings."""
        assert normalize_text(None) == ""
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""


class TestApplyUnitMap:
    """Tests for apply_unit_map function."""
    
    def test_micrometer_conversion(self):
        """Test micrometer unit conversions."""
        assert apply_unit_map("Filtro 0.22 μm") == "Filtro 0.22 um"
        assert apply_unit_map("Filtro 0.22 µm") == "Filtro 0.22 um"
    
    def test_volume_units(self):
        """Test volume unit conversions."""
        assert apply_unit_map("Jeringa 10 mL") == "Jeringa 10 ml"
        assert apply_unit_map("Botella 500 ML") == "Botella 500 ml"
    
    def test_weight_units(self):
        """Test weight unit conversions."""
        assert apply_unit_map("Comprimido 500 MG") == "Comprimido 500 mg"
        assert apply_unit_map("Tableta 1 gr") == "Tableta 1 g"
    
    def test_multiple_units(self):
        """Test text with multiple units."""
        result = apply_unit_map("Jeringa 10 mL con aguja 25G filtro 0.22 μm")
        assert "ml" in result
        assert "um" in result
    
    def test_custom_unit_map(self):
        """Test with custom unit mapping."""
        custom_map = {"kg": "kilogram", "g": "gram"}
        result = apply_unit_map("Peso 2 kg", unit_map=custom_map)
        assert "kilogram" in result
    
    def test_none_and_empty(self):
        """Test handling of None and empty strings."""
        assert apply_unit_map(None) == ""
        assert apply_unit_map("") == ""


class TestExpandAbbr:
    """Tests for expand_abbr function."""
    
    def test_basic_abbreviations(self):
        """Test basic abbreviation expansion."""
        assert "con" in expand_abbr("Jeringa c/ aguja")
        assert "sin" in expand_abbr("Envase s/ tapa")
    
    def test_medical_abbreviations(self):
        """Test medical abbreviation expansion."""
        result = expand_abbr("Sol. iny. 10 ml")
        assert "solucion" in result.lower()
        assert "inyectable" in result.lower()
    
    def test_packaging_abbreviations(self):
        """Test packaging abbreviation expansion."""
        assert "paquete" in expand_abbr("paq. de 10").lower()
        assert "caja" in expand_abbr("caj. x 100").lower()
    
    def test_multiple_abbreviations(self):
        """Test text with multiple abbreviations."""
        result = expand_abbr("Sol. iny. c/ amp. x 10 ml")
        assert "solucion" in result.lower()
        assert "inyectable" in result.lower()
        assert "con" in result.lower()
        assert "ampolla" in result.lower()
    
    def test_custom_abbr_map(self):
        """Test with custom abbreviation mapping."""
        custom_map = {"Dr.": "Doctor", "Sr.": "Señor"}
        result = expand_abbr("Dr. Smith", abbr_map=custom_map)
        assert "Doctor" in result
    
    def test_none_and_empty(self):
        """Test handling of None and empty strings."""
        assert expand_abbr(None) == ""
        assert expand_abbr("") == ""


class TestExtractCodesSizes:
    """Tests for extract_codes_sizes function."""
    
    def test_extract_reference_codes(self):
        """Test extraction of reference codes."""
        codes, _ = extract_codes_sizes("Aguja BD REF: 305122")
        assert "305122" in codes
    
    def test_extract_gauge_sizes(self):
        """Test extraction of gauge sizes."""
        codes, sizes = extract_codes_sizes("Aguja 25G x 1.5 pulgadas")
        assert "25G" in codes
        assert any("25" in s for s in sizes)
    
    def test_extract_numeric_sizes(self):
        """Test extraction of numeric sizes."""
        _, sizes = extract_codes_sizes("Jeringa 10 ml")
        assert any("10" in s for s in sizes)
    
    def test_extract_multiple_codes_sizes(self):
        """Test extraction of multiple codes and sizes."""
        codes, sizes = extract_codes_sizes("Aguja BD 25G x 1.5 pulgadas REF: 305122")
        assert len(codes) >= 1
        assert len(sizes) >= 1
    
    def test_no_codes_or_sizes(self):
        """Test text without codes or sizes."""
        codes, sizes = extract_codes_sizes("Producto generico")
        # Should return empty or minimal results
        assert isinstance(codes, list)
        assert isinstance(sizes, list)
    
    def test_none_and_empty(self):
        """Test handling of None and empty strings."""
        codes, sizes = extract_codes_sizes(None)
        assert codes == []
        assert sizes == []
        
        codes, sizes = extract_codes_sizes("")
        assert codes == []
        assert sizes == []


class TestNormalizeProductDescription:
    """Tests for normalize_product_description function."""
    
    def test_full_normalization_pipeline(self):
        """Test complete normalization pipeline."""
        input_text = "Sol. iny. 0,9% c/ 10 mL"
        result = normalize_product_description(input_text)
        
        # Should expand abbreviations
        assert "solucion" in result
        assert "inyectable" in result
        assert "con" in result
        
        # Should normalize units
        assert "ml" in result
        
        # Should normalize decimals
        assert "0.9" in result
    
    def test_with_accents_and_units(self):
        """Test normalization with accents and units."""
        input_text = "Solución estéril 500 mL"
        result = normalize_product_description(input_text)
        
        assert "solucion" in result
        assert "esteril" in result
        assert "ml" in result
    
    def test_complex_description(self):
        """Test complex product description."""
        input_text = "Aguja hipodérmica c/ bisel 25G x 1,5\" est. REF: 305122"
        result = normalize_product_description(input_text)
        
        assert "hipodermica" in result
        assert "con" in result
        assert "1.5" in result
    
    def test_selective_normalization(self):
        """Test with selective normalization options."""
        input_text = "Sol. 10 mL"
        
        # Without abbreviation expansion
        result = normalize_product_description(input_text, expand_abbreviations=False)
        assert "sol." in result.lower()
        
        # Without unit standardization
        result = normalize_product_description(input_text, standardize_units=False)
        assert "mL" in result or "ml" in result
    
    def test_none_and_empty(self):
        """Test handling of None and empty strings."""
        assert normalize_product_description(None) == ""
        assert normalize_product_description("") == ""


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_special_characters(self):
        """Test handling of special characters."""
        result = normalize_text("Aguja 25G x 1½\"")
        assert isinstance(result, str)
    
    def test_mixed_languages(self):
        """Test handling of mixed language text."""
        result = normalize_text("Producto médico - Medical product")
        assert "medico" in result
        assert "medical" in result
    
    def test_numbers_only(self):
        """Test handling of numeric-only text."""
        result = normalize_text("123456")
        assert result == "123456"
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        long_text = "Producto " * 100
        result = normalize_text(long_text)
        assert isinstance(result, str)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
