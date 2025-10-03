"""
Unit tests for the router module.
"""

import pytest
from botiquin_windsurf.router import (
    Router,
    CategoryMatch,
    route_categories,
    get_best_category
)
from botiquin_windsurf.taxonomy import get_taxonomy


class TestCategoryMatch:
    """Tests for CategoryMatch dataclass."""
    
    def test_category_match_creation(self):
        """Test creating a CategoryMatch."""
        match = CategoryMatch(
            category_name="Test Category",
            score=5,
            matched_keywords=["test", "sample"]
        )
        assert match.category_name == "Test Category"
        assert match.score == 5
        assert len(match.matched_keywords) == 2


class TestRouter:
    """Tests for Router class."""
    
    def test_router_initialization(self):
        """Test router initialization."""
        router = Router()
        assert router.taxonomy is not None
    
    def test_route_categories_with_matches(self):
        """Test routing with matching keywords."""
        router = Router()
        matches = router.route_categories("aguja hipodermica")
        
        assert len(matches) > 0
        assert matches[0].score > 0
        assert any("Agujas" in match.category_name for match in matches)
    
    def test_route_categories_sorted_by_score(self):
        """Test that results are sorted by score (descending)."""
        router = Router()
        matches = router.route_categories("aguja jeringa hipodermica")
        
        if len(matches) > 1:
            for i in range(len(matches) - 1):
                assert matches[i].score >= matches[i + 1].score
    
    def test_route_categories_no_matches(self):
        """Test routing with no matching keywords."""
        router = Router()
        matches = router.route_categories("xyzabc123nonexistent")
        
        assert len(matches) == 0
    
    def test_route_categories_empty_query(self):
        """Test routing with empty query."""
        router = Router()
        matches = router.route_categories("")
        
        assert len(matches) == 0
    
    def test_route_categories_none_query(self):
        """Test routing with None query."""
        router = Router()
        matches = router.route_categories(None)
        
        assert len(matches) == 0
    
    def test_route_categories_min_score(self):
        """Test min_score parameter."""
        router = Router()
        
        # Get all matches
        all_matches = router.route_categories("aguja", min_score=1)
        
        # Get only high-scoring matches
        high_matches = router.route_categories("aguja", min_score=2)
        
        # High score matches should be subset of all matches
        assert len(high_matches) <= len(all_matches)
    
    def test_route_categories_max_results(self):
        """Test max_results parameter."""
        router = Router()
        matches = router.route_categories("aguja jeringa", max_results=2)
        
        assert len(matches) <= 2
    
    def test_get_best_category(self):
        """Test getting best matching category."""
        router = Router()
        best = router.get_best_category("aguja hipodermica")
        
        assert best is not None
        assert isinstance(best, CategoryMatch)
        assert best.score > 0
    
    def test_get_best_category_no_match(self):
        """Test getting best category with no matches."""
        router = Router()
        best = router.get_best_category("xyznonexistent")
        
        assert best is None
    
    def test_get_category_scores(self):
        """Test getting category scores as dictionary."""
        router = Router()
        scores = router.get_category_scores("aguja jeringa")
        
        assert isinstance(scores, dict)
        assert len(scores) > 0
        assert all(isinstance(score, int) for score in scores.values())


class TestRouterKeywordMatching:
    """Tests for keyword matching logic."""
    
    def test_single_keyword_match(self):
        """Test matching single keyword."""
        router = Router()
        matches = router.route_categories("aguja")
        
        assert len(matches) > 0
        assert matches[0].score >= 1
    
    def test_multiple_keyword_match(self):
        """Test matching multiple keywords from same category."""
        router = Router()
        matches = router.route_categories("aguja jeringa hipodermica")
        
        # Should find "Agujas y Jeringas" with high score
        agujas_match = next((m for m in matches if "Agujas" in m.category_name), None)
        assert agujas_match is not None
        assert agujas_match.score >= 2  # At least 2 keywords matched
    
    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        router = Router()
        matches1 = router.route_categories("aguja")
        matches2 = router.route_categories("AGUJA")
        matches3 = router.route_categories("Aguja")
        
        assert len(matches1) == len(matches2) == len(matches3)
    
    def test_partial_word_matching(self):
        """Test matching keywords within words."""
        router = Router()
        # "agujas" (plural) should match "aguja" keyword
        matches = router.route_categories("agujas")
        
        assert len(matches) > 0
    
    def test_multi_word_keyword_matching(self):
        """Test matching multi-word keywords."""
        router = Router()
        # "porta agujas" is a multi-word keyword
        matches = router.route_categories("porta agujas quirurgico")
        
        # Should match "Instrumental Quirúrgico" category
        assert len(matches) > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_route_categories_function(self):
        """Test route_categories convenience function."""
        matches = route_categories("aguja hipodermica")
        
        assert isinstance(matches, list)
        assert len(matches) > 0
    
    def test_get_best_category_function(self):
        """Test get_best_category convenience function."""
        best = get_best_category("aguja hipodermica")
        
        assert best is not None
        assert isinstance(best, CategoryMatch)


class TestRealWorldScenarios:
    """Tests with real-world product descriptions."""
    
    def test_syringe_description(self):
        """Test routing syringe description."""
        router = Router()
        matches = router.route_categories("jeringa 10 ml luer lock")
        
        assert len(matches) > 0
        assert any("Agujas" in m.category_name or "Jeringa" in m.category_name 
                  for m in matches)
    
    def test_needle_description(self):
        """Test routing needle description."""
        router = Router()
        matches = router.route_categories("aguja hipodermica 25g x 1.5 pulgadas")
        
        assert len(matches) > 0
        assert matches[0].score >= 1
    
    def test_solution_description(self):
        """Test routing solution description."""
        router = Router()
        matches = router.route_categories("solucion salina 0.9% 500 ml")
        
        assert len(matches) > 0
        # Should match "Soluciones y Líquidos"
        assert any("Solucion" in m.category_name or "Líquido" in m.category_name 
                  for m in matches)
    
    def test_glove_description(self):
        """Test routing glove description."""
        router = Router()
        matches = router.route_categories("guante latex sin polvo talla m")
        
        assert len(matches) > 0
        assert any("Guante" in m.category_name or "Protección" in m.category_name 
                  for m in matches)
    
    def test_filter_description(self):
        """Test routing filter description."""
        router = Router()
        matches = router.route_categories("filtro membrana 0.22 um esteril")
        
        assert len(matches) > 0
        assert any("Filtro" in m.category_name or "Membrana" in m.category_name 
                  for m in matches)
    
    def test_catheter_description(self):
        """Test routing catheter description."""
        router = Router()
        matches = router.route_categories("cateter venoso periferico")
        
        assert len(matches) > 0
        assert any("Catéter" in m.category_name or "Sonda" in m.category_name 
                  for m in matches)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_very_long_query(self):
        """Test with very long query."""
        router = Router()
        long_query = "aguja " * 100
        matches = router.route_categories(long_query)
        
        assert isinstance(matches, list)
    
    def test_query_with_numbers(self):
        """Test query with numbers."""
        router = Router()
        matches = router.route_categories("aguja 25g 1.5 pulgadas")
        
        assert len(matches) > 0
    
    def test_query_with_special_characters(self):
        """Test query with special characters."""
        router = Router()
        matches = router.route_categories("aguja 25g x 1.5\"")
        
        assert len(matches) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
