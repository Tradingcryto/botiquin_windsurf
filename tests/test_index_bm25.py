"""
Unit tests for the index_bm25 module.
"""

import pytest
import pandas as pd
from botiquin_windsurf.index_bm25 import BM25Index, SearchResult


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    data = {
        'id': [1, 2, 3, 4, 5, 6],
        'text': [
            'aguja hipodermica 25g x 1.5 pulgadas',
            'jeringa 10 ml luer lock',
            'solucion salina 0.9% 500 ml',
            'guante latex sin polvo talla m',
            'aguja 21g x 1 pulgada',
            'jeringa insulina 1 ml'
        ],
        'category': [
            'Agujas y Jeringas',
            'Agujas y Jeringas',
            'Soluciones',
            'Guantes',
            'Agujas y Jeringas',
            'Agujas y Jeringas'
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def built_index(sample_dataframe):
    """Create a built BM25 index for testing."""
    index = BM25Index()
    index.build_indexes(
        sample_dataframe,
        text_column='text',
        category_column='category',
        row_id_column='id'
    )
    return index


class TestSearchResult:
    """Tests for SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        result = SearchResult(row_id=1, score=10.5)
        assert result.row_id == 1
        assert result.score == 10.5
    
    def test_search_result_repr(self):
        """Test SearchResult string representation."""
        result = SearchResult(row_id=1, score=10.5)
        repr_str = repr(result)
        assert "SearchResult" in repr_str
        assert "row_id=1" in repr_str


class TestBM25IndexInitialization:
    """Tests for BM25Index initialization."""
    
    def test_index_initialization(self):
        """Test index initialization."""
        index = BM25Index()
        assert index.global_index is None
        assert index.global_corpus == []
        assert index.global_row_ids == []
        assert index.category_indexes == {}
        assert index.is_built is False
    
    def test_index_repr(self):
        """Test index string representation."""
        index = BM25Index()
        repr_str = repr(index)
        assert "BM25Index" in repr_str


class TestBuildGlobalIndex:
    """Tests for building global index."""
    
    def test_build_global_index(self, sample_dataframe):
        """Test building global index."""
        index = BM25Index()
        index.build_global_index(sample_dataframe, text_column='text', row_id_column='id')
        
        assert index.is_built is True
        assert index.global_index is not None
        assert len(index.global_corpus) == len(sample_dataframe)
        assert len(index.global_row_ids) == len(sample_dataframe)
    
    def test_build_global_index_with_default_row_ids(self, sample_dataframe):
        """Test building global index with default row IDs (DataFrame index)."""
        index = BM25Index()
        index.build_global_index(sample_dataframe, text_column='text')
        
        assert index.is_built is True
        assert index.global_row_ids == sample_dataframe.index.tolist()
    
    def test_build_global_index_invalid_column(self, sample_dataframe):
        """Test building global index with invalid column name."""
        index = BM25Index()
        
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            index.build_global_index(sample_dataframe, text_column='nonexistent')
    
    def test_build_global_index_with_nan_values(self):
        """Test building global index with NaN values."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'text': ['aguja 25g', None, 'jeringa 10 ml']
        })
        
        index = BM25Index()
        index.build_global_index(df, text_column='text', row_id_column='id')
        
        assert index.is_built is True
        assert len(index.global_corpus) == 3
        assert index.global_corpus[1] == []  # NaN becomes empty list


class TestBuildCategoryIndexes:
    """Tests for building category indexes."""
    
    def test_build_category_indexes(self, sample_dataframe):
        """Test building category indexes."""
        index = BM25Index()
        index.build_category_indexes(
            sample_dataframe,
            text_column='text',
            category_column='category',
            row_id_column='id'
        )
        
        assert len(index.category_indexes) > 0
        assert 'Agujas y Jeringas' in index.category_indexes
        assert 'Soluciones' in index.category_indexes
        assert 'Guantes' in index.category_indexes
    
    def test_build_category_indexes_invalid_text_column(self, sample_dataframe):
        """Test building category indexes with invalid text column."""
        index = BM25Index()
        
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            index.build_category_indexes(
                sample_dataframe,
                text_column='nonexistent',
                category_column='category'
            )
    
    def test_build_category_indexes_invalid_category_column(self, sample_dataframe):
        """Test building category indexes with invalid category column."""
        index = BM25Index()
        
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            index.build_category_indexes(
                sample_dataframe,
                text_column='text',
                category_column='nonexistent'
            )


class TestBuildIndexes:
    """Tests for building both global and category indexes."""
    
    def test_build_indexes(self, sample_dataframe):
        """Test building both indexes."""
        index = BM25Index()
        index.build_indexes(
            sample_dataframe,
            text_column='text',
            category_column='category',
            row_id_column='id'
        )
        
        assert index.is_built is True
        assert index.global_index is not None
        assert len(index.category_indexes) > 0
    
    def test_build_indexes_without_categories(self, sample_dataframe):
        """Test building only global index."""
        index = BM25Index()
        index.build_indexes(
            sample_dataframe,
            text_column='text',
            row_id_column='id'
        )
        
        assert index.is_built is True
        assert index.global_index is not None
        assert len(index.category_indexes) == 0


class TestSearchGlobal:
    """Tests for global search."""
    
    def test_search_global_basic(self, built_index):
        """Test basic global search."""
        results = built_index.search_global("aguja", top_k=5)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
    
    def test_search_global_sorted_by_score(self, built_index):
        """Test that results are sorted by score."""
        results = built_index.search_global("aguja hipodermica", top_k=5)
        
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score
    
    def test_search_global_top_k(self, built_index):
        """Test top_k parameter."""
        results = built_index.search_global("aguja", top_k=2)
        
        assert len(results) <= 2
    
    def test_search_global_empty_query(self, built_index):
        """Test search with empty query."""
        results = built_index.search_global("", top_k=5)
        
        assert len(results) == 0
    
    def test_search_global_none_query(self, built_index):
        """Test search with None query."""
        results = built_index.search_global(None, top_k=5)
        
        assert len(results) == 0
    
    def test_search_global_no_matches(self, built_index):
        """Test search with query that has no good matches."""
        results = built_index.search_global("xyznonexistent", top_k=5)
        
        # Should return empty or very low scores
        assert isinstance(results, list)
    
    def test_search_global_not_built(self):
        """Test search when index is not built."""
        index = BM25Index()
        
        with pytest.raises(RuntimeError, match="Global index has not been built"):
            index.search_global("aguja", top_k=5)
    
    def test_search_global_returns_correct_row_ids(self, built_index):
        """Test that search returns correct row IDs."""
        results = built_index.search_global("aguja", top_k=5)
        
        assert len(results) > 0
        # Row IDs should be from the original DataFrame
        assert all(r.row_id in [1, 2, 3, 4, 5, 6] for r in results)


class TestSearchCategory:
    """Tests for category search."""
    
    def test_search_category_basic(self, built_index):
        """Test basic category search."""
        results = built_index.search_category("aguja", "Agujas y Jeringas", top_k=5)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
    
    def test_search_category_sorted_by_score(self, built_index):
        """Test that results are sorted by score."""
        results = built_index.search_category("aguja", "Agujas y Jeringas", top_k=5)
        
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score
    
    def test_search_category_invalid_category(self, built_index):
        """Test search with invalid category."""
        with pytest.raises(ValueError, match="Category 'NonExistent' not found"):
            built_index.search_category("aguja", "NonExistent", top_k=5)
    
    def test_search_category_empty_query(self, built_index):
        """Test search with empty query."""
        results = built_index.search_category("", "Agujas y Jeringas", top_k=5)
        
        assert len(results) == 0
    
    def test_search_category_returns_correct_row_ids(self, built_index):
        """Test that category search returns correct row IDs."""
        results = built_index.search_category("aguja", "Agujas y Jeringas", top_k=5)
        
        assert len(results) > 0
        # Row IDs should be from the "Agujas y Jeringas" category (1, 2, 5, 6)
        assert all(r.row_id in [1, 2, 5, 6] for r in results)


class TestSearchMultipleCategories:
    """Tests for searching multiple categories."""
    
    def test_search_multiple_categories(self, built_index):
        """Test searching across multiple categories."""
        results = built_index.search_multiple_categories(
            "aguja jeringa",
            ["Agujas y Jeringas", "Soluciones"],
            top_k=5
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
    
    def test_search_multiple_categories_sorted(self, built_index):
        """Test that results are sorted by score."""
        results = built_index.search_multiple_categories(
            "aguja",
            ["Agujas y Jeringas", "Guantes"],
            top_k=5
        )
        
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score
    
    def test_search_multiple_categories_with_invalid(self, built_index):
        """Test searching with some invalid categories."""
        results = built_index.search_multiple_categories(
            "aguja",
            ["Agujas y Jeringas", "NonExistent"],
            top_k=5
        )
        
        # Should still return results from valid categories
        assert isinstance(results, list)


class TestIndexUtilities:
    """Tests for index utility methods."""
    
    def test_get_categories(self, built_index):
        """Test getting list of categories."""
        categories = built_index.get_categories()
        
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert 'Agujas y Jeringas' in categories
    
    def test_get_global_size(self, built_index):
        """Test getting global index size."""
        size = built_index.get_global_size()
        
        assert size == 6  # Sample DataFrame has 6 rows
    
    def test_get_category_size(self, built_index):
        """Test getting category size."""
        size = built_index.get_category_size("Agujas y Jeringas")
        
        assert size == 4  # 4 items in "Agujas y Jeringas" category
    
    def test_get_category_size_invalid(self, built_index):
        """Test getting size of invalid category."""
        with pytest.raises(ValueError, match="Category 'NonExistent' not found"):
            built_index.get_category_size("NonExistent")
    
    def test_get_stats(self, built_index):
        """Test getting index statistics."""
        stats = built_index.get_stats()
        
        assert isinstance(stats, dict)
        assert stats['is_built'] is True
        assert stats['global_documents'] == 6
        assert stats['num_categories'] == 3
        assert 'categories' in stats
        assert isinstance(stats['categories'], dict)


class TestRealWorldScenarios:
    """Tests with real-world scenarios."""
    
    def test_search_needle_products(self, built_index):
        """Test searching for needle products."""
        results = built_index.search_global("aguja hipodermica 25g", top_k=3)
        
        assert len(results) > 0
        # Should find needle products with high scores
        assert results[0].score > 0
    
    def test_search_syringe_products(self, built_index):
        """Test searching for syringe products."""
        results = built_index.search_global("jeringa 10 ml", top_k=3)
        
        assert len(results) > 0
        assert results[0].score > 0
    
    def test_category_specific_search(self, built_index):
        """Test category-specific search."""
        # Search only in "Agujas y Jeringas" category
        results = built_index.search_category("jeringa", "Agujas y Jeringas", top_k=5)
        
        assert len(results) > 0
        # All results should be from "Agujas y Jeringas" category
        assert all(r.row_id in [1, 2, 5, 6] for r in results)
    
    def test_multi_word_query(self, built_index):
        """Test multi-word query."""
        results = built_index.search_global("aguja 25g pulgadas", top_k=3)
        
        assert len(results) > 0
        assert results[0].score > 0


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({'id': [], 'text': [], 'category': []})
        index = BM25Index()
        index.build_indexes(df, text_column='text', category_column='category', row_id_column='id')
        
        assert index.get_global_size() == 0
        assert len(index.get_categories()) == 0
    
    def test_single_document(self):
        """Test with single document."""
        df = pd.DataFrame({
            'id': [1],
            'text': ['aguja 25g'],
            'category': ['Agujas']
        })
        index = BM25Index()
        index.build_indexes(df, text_column='text', category_column='category', row_id_column='id')
        
        results = index.search_global("aguja", top_k=5)
        assert len(results) == 1
    
    def test_special_characters_in_text(self):
        """Test with special characters in text."""
        df = pd.DataFrame({
            'id': [1, 2],
            'text': ['aguja 25G x 1.5"', 'jeringa 10 ml (luer lock)'],
            'category': ['Agujas', 'Jeringas']
        })
        index = BM25Index()
        index.build_indexes(df, text_column='text', category_column='category', row_id_column='id')
        
        results = index.search_global("aguja", top_k=5)
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
