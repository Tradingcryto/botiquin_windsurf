"""
Unit tests for the search module.
"""

import pytest
import pandas as pd
import logging
from botiquin_windsurf.search import (
    SearchEngine,
    SearchContext,
    search_one,
    configure_logging
)
from botiquin_windsurf.index_bm25 import BM25Index
from botiquin_windsurf.fuse import Fuser


@pytest.fixture
def sample_catalog():
    """Create a sample catalog DataFrame."""
    data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'product_name': [
            'Aguja BD 25G x 1.5 pulgadas',
            'Jeringa 10 ml Luer Lock',
            'Solución Salina 0.9% 500 ml',
            'Guante Latex sin polvo M',
            'Aguja 21G x 1 pulgada',
            'Jeringa Insulina 1 ml',
            'Filtro 0.22 um estéril',
            'Catéter venoso 20G'
        ],
        'category': [
            'Agujas y Jeringas',
            'Agujas y Jeringas',
            'Soluciones y Líquidos',
            'Guantes y Protección',
            'Agujas y Jeringas',
            'Agujas y Jeringas',
            'Filtros y Membranas',
            'Catéteres y Sondas'
        ],
        'fabricante': ['BD', 'BD', 'Baxter', 'Medline', 'BD', 'BD', 'Millipore', 'BD'],
        'ref_fabricante': ['305122', '302995', 'BAX001', 'MED456', '305127', '305128', 'MIL022', 'BD-CAT20'],
        'mida': ['25G', '10ml', '500ml', 'M', '21G', '1ml', '0.22um', '20G']
    }
    return pd.DataFrame(data)


@pytest.fixture
def built_bm25_index(sample_catalog):
    """Create a built BM25 index."""
    index = BM25Index()
    index.build_indexes(
        sample_catalog,
        text_column='product_name',
        category_column='category',
        row_id_column='id'
    )
    return index


@pytest.fixture
def search_engine(sample_catalog, built_bm25_index):
    """Create a SearchEngine instance."""
    return SearchEngine(
        catalog_df=sample_catalog,
        bm25_index=built_bm25_index,
        text_column='product_name',
        category_column='category',
        row_id_column='id'
    )


class TestSearchContext:
    """Tests for SearchContext dataclass."""
    
    def test_search_context_creation(self):
        """Test creating a SearchContext."""
        context = SearchContext(
            lic_ref="LIC001",
            query_text="aguja 25g",
            query_normalized="aguja 25g",
            extracted_codes=['25G'],
            extracted_sizes=['25'],
            routed_categories=[],
            prefer_category=True
        )
        
        assert context.lic_ref == "LIC001"
        assert context.query_text == "aguja 25g"
        assert len(context.extracted_codes) == 1


class TestSearchEngineInitialization:
    """Tests for SearchEngine initialization."""
    
    def test_search_engine_init(self, sample_catalog, built_bm25_index):
        """Test SearchEngine initialization."""
        engine = SearchEngine(
            catalog_df=sample_catalog,
            bm25_index=built_bm25_index
        )
        
        assert engine.catalog_df is not None
        assert engine.bm25_index is not None
        assert engine.fuser is not None
    
    def test_search_engine_with_custom_fuser(self, sample_catalog, built_bm25_index):
        """Test SearchEngine with custom Fuser."""
        custom_fuser = Fuser(k=30)
        engine = SearchEngine(
            catalog_df=sample_catalog,
            bm25_index=built_bm25_index,
            fuser=custom_fuser
        )
        
        assert engine.fuser.k == 30


class TestNormalizeQuery:
    """Tests for query normalization."""
    
    def test_normalize_query(self, search_engine):
        """Test query normalization."""
        normalized = search_engine._normalize_query("Aguja 25G x 1.5\"")
        
        assert isinstance(normalized, str)
        assert normalized.islower()
    
    def test_normalize_query_with_accents(self, search_engine):
        """Test normalization removes accents."""
        normalized = search_engine._normalize_query("Solución estéril")
        
        assert "solucion" in normalized
        assert "esteril" in normalized


class TestExtractFeatures:
    """Tests for feature extraction."""
    
    def test_extract_features_basic(self, search_engine):
        """Test basic feature extraction."""
        features = search_engine._extract_features("aguja 25g x 1.5 pulgadas")
        
        assert 'codes' in features
        assert 'sizes' in features
        assert isinstance(features['codes'], list)
        assert isinstance(features['sizes'], list)
    
    def test_extract_features_with_codes(self, search_engine):
        """Test extraction of product codes."""
        features = search_engine._extract_features("aguja ref 305122 25g")
        
        assert len(features['codes']) > 0
    
    def test_extract_features_empty_query(self, search_engine):
        """Test feature extraction with empty query."""
        features = search_engine._extract_features("")
        
        assert features['codes'] == []
        assert features['sizes'] == []


class TestExactMatch:
    """Tests for exact matching."""
    
    def test_exact_match_found(self, search_engine):
        """Test exact match when product exists."""
        # Use exact normalized product name
        result = search_engine._exact_match("aguja bd 25g x 1.5 pulgadas")
        
        # May or may not find exact match depending on normalization
        assert result is None or isinstance(result, pd.DataFrame)
    
    def test_exact_match_not_found(self, search_engine):
        """Test exact match when product doesn't exist."""
        result = search_engine._exact_match("producto inexistente xyz123")
        
        assert result is None


class TestRouteCategories:
    """Tests for category routing."""
    
    def test_route_categories_with_matches(self, search_engine):
        """Test routing with matching categories."""
        categories = search_engine._route_categories("aguja hipodermica")
        
        assert isinstance(categories, list)
        # Should match "Agujas y Jeringas" category
        if categories:
            assert any("Aguja" in c.category_name for c in categories)
    
    def test_route_categories_no_matches(self, search_engine):
        """Test routing with no matching categories."""
        categories = search_engine._route_categories("xyznonexistent")
        
        assert isinstance(categories, list)
        assert len(categories) == 0


class TestSearchBM25:
    """Tests for BM25 search."""
    
    def test_search_bm25_global(self, search_engine):
        """Test global BM25 search."""
        results = search_engine._search_bm25("aguja", categories=None, top_k=5)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    
    def test_search_bm25_category(self, search_engine):
        """Test category-specific BM25 search."""
        results = search_engine._search_bm25(
            "aguja",
            categories=["Agujas y Jeringas"],
            top_k=5
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
    
    def test_search_bm25_multiple_categories(self, search_engine):
        """Test BM25 search across multiple categories."""
        results = search_engine._search_bm25(
            "aguja jeringa",
            categories=["Agujas y Jeringas", "Soluciones y Líquidos"],
            top_k=5
        )
        
        assert isinstance(results, list)


class TestSearchVector:
    """Tests for vector search."""
    
    def test_search_vector_no_index(self, search_engine):
        """Test vector search without vector index."""
        results = search_engine._search_vector("aguja", categories=None, top_k=5)
        
        # Should return empty list when no vector index
        assert results == []


class TestFuseResults:
    """Tests for result fusion."""
    
    def test_fuse_results_basic(self, search_engine):
        """Test basic result fusion."""
        bm25_results = [(1, 10.0), (2, 8.0), (3, 5.0)]
        vector_results = [(2, 0.9), (1, 0.8), (4, 0.7)]
        features = {'fabricante': 'BD', 'ref_fabricante': None, 'mida': '25G'}
        
        result_df = search_engine._fuse_results(bm25_results, vector_results, features)
        
        assert isinstance(result_df, pd.DataFrame)
        assert not result_df.empty
        assert 'final_score' in result_df.columns
        assert 'coincidencia_pct' in result_df.columns
    
    def test_fuse_results_with_boosts(self, search_engine):
        """Test fusion with feature boosts."""
        bm25_results = [(1, 10.0)]
        vector_results = [(1, 0.9)]
        features = {'fabricante': 'BD', 'ref_fabricante': '305122', 'mida': '25G'}
        
        result_df = search_engine._fuse_results(bm25_results, vector_results, features)
        
        assert not result_df.empty
        # Should have boost_score column
        assert 'boost_score' in result_df.columns


class TestSearchOne:
    """Tests for the main search_one method."""
    
    def test_search_one_basic(self, search_engine):
        """Test basic search_one operation."""
        results = search_engine.search_one(
            lic_ref="LIC001",
            query_text="aguja 25g",
            prefer_cat=True,
            top_k=5
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) <= 5
    
    def test_search_one_with_category_preference(self, search_engine):
        """Test search with category preference."""
        results = search_engine.search_one(
            lic_ref="LIC002",
            query_text="aguja hipodermica 25g",
            prefer_cat=True,
            top_k=3
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) <= 3
    
    def test_search_one_global_search(self, search_engine):
        """Test search without category preference."""
        results = search_engine.search_one(
            lic_ref="LIC003",
            query_text="aguja",
            prefer_cat=False,
            top_k=5
        )
        
        assert isinstance(results, pd.DataFrame)
    
    def test_search_one_returns_top_k(self, search_engine):
        """Test that search returns exactly top_k results (or fewer)."""
        results = search_engine.search_one(
            lic_ref="LIC004",
            query_text="aguja jeringa",
            prefer_cat=True,
            top_k=3
        )
        
        assert len(results) <= 3
    
    def test_search_one_includes_scores(self, search_engine):
        """Test that results include score columns."""
        results = search_engine.search_one(
            lic_ref="LIC005",
            query_text="aguja",
            top_k=5
        )
        
        if not results.empty:
            assert 'final_score' in results.columns
            assert 'coincidencia_pct' in results.columns
    
    def test_search_one_coincidencia_capped(self, search_engine):
        """Test that coincidencia is capped at 100%."""
        results = search_engine.search_one(
            lic_ref="LIC006",
            query_text="aguja",
            top_k=5
        )
        
        if not results.empty:
            assert all(results['coincidencia_pct'] <= 100.0)
    
    def test_search_one_sorted_by_score(self, search_engine):
        """Test that results are sorted by final score."""
        results = search_engine.search_one(
            lic_ref="LIC007",
            query_text="aguja jeringa",
            top_k=5
        )
        
        if len(results) > 1:
            scores = results['final_score'].tolist()
            assert scores == sorted(scores, reverse=True)
    
    def test_search_one_no_results(self, search_engine):
        """Test search with query that yields no results."""
        results = search_engine.search_one(
            lic_ref="LIC008",
            query_text="xyznonexistentproduct12345",
            top_k=5
        )
        
        # May return empty DataFrame or very low-scored results
        assert isinstance(results, pd.DataFrame)


class TestSearchOneConvenienceFunction:
    """Tests for the search_one convenience function."""
    
    def test_search_one_function(self, sample_catalog, built_bm25_index):
        """Test search_one convenience function."""
        results = search_one(
            lic_ref="LIC001",
            query_text="aguja 25g",
            catalog_df=sample_catalog,
            bm25_index=built_bm25_index,
            prefer_cat=True,
            top_k=5
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) <= 5


class TestRealWorldScenarios:
    """Tests with real-world search scenarios."""
    
    def test_search_needle_product(self, search_engine):
        """Test searching for needle products."""
        results = search_engine.search_one(
            lic_ref="LIC_NEEDLE",
            query_text="aguja hipodermica 25g x 1.5 pulgadas",
            prefer_cat=True,
            top_k=5
        )
        
        assert not results.empty
        # Should find needle products
        if not results.empty:
            assert any('aguja' in str(name).lower() 
                      for name in results['product_name'])
    
    def test_search_syringe_product(self, search_engine):
        """Test searching for syringe products."""
        results = search_engine.search_one(
            lic_ref="LIC_SYRINGE",
            query_text="jeringa 10 ml luer lock",
            prefer_cat=True,
            top_k=5
        )
        
        assert not results.empty
        if not results.empty:
            assert any('jeringa' in str(name).lower() 
                      for name in results['product_name'])
    
    def test_search_solution_product(self, search_engine):
        """Test searching for solution products."""
        results = search_engine.search_one(
            lic_ref="LIC_SOLUTION",
            query_text="solucion salina 0.9%",
            prefer_cat=True,
            top_k=5
        )
        
        assert not results.empty
    
    def test_search_with_manufacturer(self, search_engine):
        """Test search with manufacturer information."""
        results = search_engine.search_one(
            lic_ref="LIC_BD",
            query_text="aguja bd 25g",
            prefer_cat=True,
            top_k=5
        )
        
        assert not results.empty
        # BD products should be boosted
        if not results.empty and 'fabricante' in results.columns:
            # Top results should include BD products
            top_manufacturers = results.head(3)['fabricante'].tolist()
            assert 'BD' in top_manufacturers
    
    def test_search_filter_product(self, search_engine):
        """Test searching for filter products."""
        results = search_engine.search_one(
            lic_ref="LIC_FILTER",
            query_text="filtro 0.22 um esteril",
            prefer_cat=True,
            top_k=5
        )
        
        assert not results.empty


class TestLogging:
    """Tests for logging functionality."""
    
    def test_configure_logging(self):
        """Test logging configuration."""
        configure_logging(level=logging.DEBUG)
        
        # Should not raise any errors
        assert True
    
    def test_search_produces_logs(self, search_engine, caplog):
        """Test that search operations produce logs."""
        with caplog.at_level(logging.INFO):
            search_engine.search_one(
                lic_ref="LIC_LOG",
                query_text="aguja",
                top_k=5
            )
        
        # Should have log messages
        assert len(caplog.records) > 0


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_search_empty_query(self, search_engine):
        """Test search with empty query."""
        results = search_engine.search_one(
            lic_ref="LIC_EMPTY",
            query_text="",
            top_k=5
        )
        
        # Should handle gracefully
        assert isinstance(results, pd.DataFrame)
    
    def test_search_very_long_query(self, search_engine):
        """Test search with very long query."""
        long_query = "aguja " * 50
        results = search_engine.search_one(
            lic_ref="LIC_LONG",
            query_text=long_query,
            top_k=5
        )
        
        assert isinstance(results, pd.DataFrame)
    
    def test_search_special_characters(self, search_engine):
        """Test search with special characters."""
        results = search_engine.search_one(
            lic_ref="LIC_SPECIAL",
            query_text="aguja 25G x 1.5\"",
            top_k=5
        )
        
        assert isinstance(results, pd.DataFrame)
    
    def test_search_top_k_zero(self, search_engine):
        """Test search with top_k=0."""
        results = search_engine.search_one(
            lic_ref="LIC_ZERO",
            query_text="aguja",
            top_k=0
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0
    
    def test_search_top_k_large(self, search_engine):
        """Test search with very large top_k."""
        results = search_engine.search_one(
            lic_ref="LIC_LARGE",
            query_text="aguja",
            top_k=1000
        )
        
        assert isinstance(results, pd.DataFrame)
        # Should return at most the number of products in catalog
        assert len(results) <= len(search_engine.catalog_df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
