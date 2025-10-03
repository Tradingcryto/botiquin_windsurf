"""
Unit tests for the taxonomy module.
"""

import pytest
from botiquin_windsurf.taxonomy import Taxonomy, Category, get_taxonomy
from botiquin_windsurf.config import ConfigError


class TestCategory:
    """Tests for Category class."""
    
    def test_category_creation(self):
        """Test creating a category."""
        cat = Category(name="Test Category", keywords=["test", "sample"])
        assert cat.name == "Test Category"
        assert len(cat.keywords) == 2
    
    def test_keywords_normalized_to_lowercase(self):
        """Test that keywords are normalized to lowercase."""
        cat = Category(name="Test", keywords=["TEST", "Sample", "UPPER"])
        assert all(kw.islower() for kw in cat.keywords)
        assert "test" in cat.keywords
        assert "sample" in cat.keywords
        assert "upper" in cat.keywords


class TestTaxonomy:
    """Tests for Taxonomy class."""
    
    def test_taxonomy_loads_categories(self):
        """Test that taxonomy loads categories from config."""
        taxonomy = get_taxonomy()
        assert len(taxonomy.categories) > 0
    
    def test_get_categories(self):
        """Test getting all categories."""
        taxonomy = get_taxonomy()
        categories = taxonomy.get_categories()
        assert isinstance(categories, list)
        assert all(isinstance(cat, Category) for cat in categories)
    
    def test_get_category_names(self):
        """Test getting category names."""
        taxonomy = get_taxonomy()
        names = taxonomy.get_category_names()
        assert isinstance(names, list)
        assert all(isinstance(name, str) for name in names)
        assert len(names) == len(taxonomy.categories)
    
    def test_get_category_by_name(self):
        """Test getting category by name."""
        taxonomy = get_taxonomy()
        # Should find "Agujas y Jeringas" category
        cat = taxonomy.get_category_by_name("Agujas y Jeringas")
        assert cat is not None
        assert cat.name == "Agujas y Jeringas"
        assert "aguja" in cat.keywords
    
    def test_get_category_by_name_case_insensitive(self):
        """Test that category lookup is case-insensitive."""
        taxonomy = get_taxonomy()
        cat1 = taxonomy.get_category_by_name("agujas y jeringas")
        cat2 = taxonomy.get_category_by_name("AGUJAS Y JERINGAS")
        assert cat1 is not None
        assert cat2 is not None
        assert cat1.name == cat2.name
    
    def test_get_category_by_name_not_found(self):
        """Test getting non-existent category."""
        taxonomy = get_taxonomy()
        cat = taxonomy.get_category_by_name("Non-existent Category")
        assert cat is None
    
    def test_get_all_keywords(self):
        """Test getting all keywords organized by category."""
        taxonomy = get_taxonomy()
        keywords_dict = taxonomy.get_all_keywords()
        assert isinstance(keywords_dict, dict)
        assert len(keywords_dict) == len(taxonomy.categories)
        
        # Check that each category has keywords
        for cat_name, keywords in keywords_dict.items():
            assert isinstance(keywords, list)
            assert len(keywords) > 0
    
    def test_search_categories_by_keyword(self):
        """Test searching categories by keyword."""
        taxonomy = get_taxonomy()
        
        # Search for "aguja" keyword
        matches = taxonomy.search_categories_by_keyword("aguja")
        assert len(matches) > 0
        assert any(cat.name == "Agujas y Jeringas" for cat in matches)
    
    def test_search_categories_by_keyword_case_insensitive(self):
        """Test that keyword search is case-insensitive."""
        taxonomy = get_taxonomy()
        matches1 = taxonomy.search_categories_by_keyword("aguja")
        matches2 = taxonomy.search_categories_by_keyword("AGUJA")
        assert len(matches1) == len(matches2)
    
    def test_taxonomy_length(self):
        """Test taxonomy length."""
        taxonomy = get_taxonomy()
        assert len(taxonomy) == len(taxonomy.categories)
    
    def test_taxonomy_repr(self):
        """Test taxonomy string representation."""
        taxonomy = get_taxonomy()
        repr_str = repr(taxonomy)
        assert "Taxonomy" in repr_str
        assert "categories" in repr_str


class TestTaxonomyValidation:
    """Tests for taxonomy validation and error handling."""
    
    def test_categories_have_keywords(self):
        """Test that all categories have at least one keyword."""
        taxonomy = get_taxonomy()
        for category in taxonomy.categories:
            assert len(category.keywords) > 0, f"Category '{category.name}' has no keywords"
    
    def test_categories_have_names(self):
        """Test that all categories have names."""
        taxonomy = get_taxonomy()
        for category in taxonomy.categories:
            assert category.name, "Category has empty name"
            assert isinstance(category.name, str), "Category name is not a string"
    
    def test_keywords_are_normalized(self):
        """Test that all keywords are lowercase."""
        taxonomy = get_taxonomy()
        for category in taxonomy.categories:
            for keyword in category.keywords:
                assert keyword.islower(), f"Keyword '{keyword}' in category '{category.name}' is not lowercase"


class TestSpecificCategories:
    """Tests for specific categories in the taxonomy."""
    
    def test_agujas_y_jeringas_category(self):
        """Test Agujas y Jeringas category."""
        taxonomy = get_taxonomy()
        cat = taxonomy.get_category_by_name("Agujas y Jeringas")
        assert cat is not None
        assert "aguja" in cat.keywords
        assert "jeringa" in cat.keywords
    
    def test_material_curacion_category(self):
        """Test Material de Curación category."""
        taxonomy = get_taxonomy()
        cat = taxonomy.get_category_by_name("Material de Curación")
        assert cat is not None
        assert "gasa" in cat.keywords
        assert "venda" in cat.keywords
    
    def test_soluciones_category(self):
        """Test Soluciones y Líquidos category."""
        taxonomy = get_taxonomy()
        cat = taxonomy.get_category_by_name("Soluciones y Líquidos")
        assert cat is not None
        assert "solución" in cat.keywords or "solucion" in cat.keywords
        assert "suero" in cat.keywords


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
