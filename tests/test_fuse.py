"""
Unit tests for the fuse module.
"""

import pytest
import pandas as pd
from botiquin_windsurf.fuse import (
    FusedResult,
    reciprocal_rank_fusion,
    apply_boosts,
    calculate_coincidencia,
    fuse_and_boost,
    fused_results_to_dataframe,
    Fuser
)


@pytest.fixture
def sample_bm25_results():
    """Sample BM25 search results."""
    return [
        (1, 10.5),
        (2, 8.3),
        (3, 5.1),
        (4, 3.2)
    ]


@pytest.fixture
def sample_vector_results():
    """Sample vector search results."""
    return [
        (2, 0.95),
        (1, 0.87),
        (4, 0.75),
        (5, 0.68)
    ]


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame with product data."""
    data = {
        'id': [1, 2, 3, 4, 5],
        'product_name': [
            'Aguja BD 25G x 1.5"',
            'Jeringa 10 ml Luer Lock',
            'Solución Salina 0.9%',
            'Guante Latex M',
            'Aguja 21G x 1"'
        ],
        'fabricante': ['BD', 'BD', 'Baxter', 'Medline', 'BD'],
        'ref_fabricante': ['305122', '302995', 'BAX001', 'MED456', '305127'],
        'mida': ['25G', '10ml', '500ml', 'M', '21G']
    }
    return pd.DataFrame(data)


class TestFusedResult:
    """Tests for FusedResult dataclass."""
    
    def test_fused_result_creation(self):
        """Test creating a FusedResult."""
        result = FusedResult(
            row_id=1,
            rrf_score=0.5,
            bm25_score=10.0,
            vector_score=0.9,
            final_score=0.5,
            coincidencia_pct=85.0
        )
        assert result.row_id == 1
        assert result.rrf_score == 0.5
        assert result.final_score == 0.5
        assert result.coincidencia_pct == 85.0
    
    def test_fused_result_repr(self):
        """Test FusedResult string representation."""
        result = FusedResult(
            row_id=1,
            rrf_score=0.5,
            bm25_score=10.0,
            vector_score=0.9
        )
        repr_str = repr(result)
        assert "FusedResult" in repr_str
        assert "row_id=1" in repr_str


class TestReciprocalRankFusion:
    """Tests for reciprocal_rank_fusion function."""
    
    def test_rrf_basic(self, sample_bm25_results, sample_vector_results):
        """Test basic RRF fusion."""
        results = reciprocal_rank_fusion(sample_bm25_results, sample_vector_results)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, FusedResult) for r in results)
    
    def test_rrf_sorted_by_score(self, sample_bm25_results, sample_vector_results):
        """Test that results are sorted by RRF score."""
        results = reciprocal_rank_fusion(sample_bm25_results, sample_vector_results)
        
        for i in range(len(results) - 1):
            assert results[i].rrf_score >= results[i + 1].rrf_score
    
    def test_rrf_combines_all_results(self, sample_bm25_results, sample_vector_results):
        """Test that RRF combines results from both sources."""
        results = reciprocal_rank_fusion(sample_bm25_results, sample_vector_results)
        
        # Should have results from both BM25 and vector (union of row IDs)
        bm25_ids = {row_id for row_id, _ in sample_bm25_results}
        vector_ids = {row_id for row_id, _ in sample_vector_results}
        all_ids = bm25_ids | vector_ids
        
        result_ids = {r.row_id for r in results}
        assert result_ids == all_ids
    
    def test_rrf_top_ranked_gets_higher_score(self):
        """Test that top-ranked items get higher RRF scores."""
        bm25_results = [(1, 100.0), (2, 50.0)]
        vector_results = [(1, 0.99), (2, 0.50)]
        
        results = reciprocal_rank_fusion(bm25_results, vector_results, k=60)
        
        # Item 1 is ranked #1 in both, should have highest score
        assert results[0].row_id == 1
        assert results[0].rrf_score > results[1].rrf_score
    
    def test_rrf_with_different_k_values(self, sample_bm25_results, sample_vector_results):
        """Test RRF with different k values."""
        results_k60 = reciprocal_rank_fusion(sample_bm25_results, sample_vector_results, k=60)
        results_k10 = reciprocal_rank_fusion(sample_bm25_results, sample_vector_results, k=10)
        
        # Different k values should produce different scores
        assert results_k60[0].rrf_score != results_k10[0].rrf_score
    
    def test_rrf_empty_bm25(self, sample_vector_results):
        """Test RRF with empty BM25 results."""
        results = reciprocal_rank_fusion([], sample_vector_results)
        
        assert len(results) == len(sample_vector_results)
        assert all(r.bm25_score == 0.0 for r in results)
    
    def test_rrf_empty_vector(self, sample_bm25_results):
        """Test RRF with empty vector results."""
        results = reciprocal_rank_fusion(sample_bm25_results, [])
        
        assert len(results) == len(sample_bm25_results)
        assert all(r.vector_score == 0.0 for r in results)
    
    def test_rrf_both_empty(self):
        """Test RRF with both empty."""
        results = reciprocal_rank_fusion([], [])
        
        assert len(results) == 0
    
    def test_rrf_preserves_original_scores(self, sample_bm25_results, sample_vector_results):
        """Test that RRF preserves original BM25 and vector scores."""
        results = reciprocal_rank_fusion(sample_bm25_results, sample_vector_results)
        
        # Check that original scores are preserved
        for result in results:
            if result.row_id == 1:
                assert result.bm25_score == 10.5
                assert result.vector_score == 0.87
            elif result.row_id == 2:
                assert result.bm25_score == 8.3
                assert result.vector_score == 0.95


class TestApplyBoosts:
    """Tests for apply_boosts function."""
    
    def test_boost_fabricante_match(self):
        """Test boost for matching manufacturer."""
        row = pd.Series({'fabricante': 'BD', 'ref_fabricante': 'ABC', 'mida': '25G'})
        features = {'fabricante': 'BD'}
        
        boost = apply_boosts(row, features)
        assert boost == 10.0
    
    def test_boost_ref_fabricante_match(self):
        """Test boost for matching manufacturer reference."""
        row = pd.Series({'fabricante': 'BD', 'ref_fabricante': '305122', 'mida': '25G'})
        features = {'ref_fabricante': '305122'}
        
        boost = apply_boosts(row, features)
        assert boost == 15.0
    
    def test_boost_mida_match(self):
        """Test boost for matching size."""
        row = pd.Series({'fabricante': 'BD', 'ref_fabricante': 'ABC', 'mida': '25G'})
        features = {'mida': '25G'}
        
        boost = apply_boosts(row, features)
        assert boost == 5.0
    
    def test_boost_all_match(self):
        """Test boost when all features match."""
        row = pd.Series({'fabricante': 'BD', 'ref_fabricante': '305122', 'mida': '25G'})
        features = {'fabricante': 'BD', 'ref_fabricante': '305122', 'mida': '25G'}
        
        boost = apply_boosts(row, features)
        assert boost == 30.0  # 10 + 15 + 5
    
    def test_boost_no_match(self):
        """Test boost when nothing matches."""
        row = pd.Series({'fabricante': 'BD', 'ref_fabricante': '305122', 'mida': '25G'})
        features = {'fabricante': 'Baxter', 'ref_fabricante': 'XYZ', 'mida': '21G'}
        
        boost = apply_boosts(row, features)
        assert boost == 0.0
    
    def test_boost_case_insensitive(self):
        """Test that boost matching is case-insensitive."""
        row = pd.Series({'fabricante': 'BD', 'ref_fabricante': 'ABC', 'mida': '25G'})
        features = {'fabricante': 'bd'}
        
        boost = apply_boosts(row, features)
        assert boost == 10.0
    
    def test_boost_custom_config(self):
        """Test boost with custom configuration."""
        row = pd.Series({'fabricante': 'BD', 'ref_fabricante': 'ABC', 'mida': '25G'})
        features = {'fabricante': 'BD'}
        custom_config = {'fabricante': 20.0, 'ref_fabricante': 30.0, 'mida': 10.0}
        
        boost = apply_boosts(row, features, boost_config=custom_config)
        assert boost == 20.0
    
    def test_boost_empty_features(self):
        """Test boost with empty features."""
        row = pd.Series({'fabricante': 'BD', 'ref_fabricante': 'ABC', 'mida': '25G'})
        features = {}
        
        boost = apply_boosts(row, features)
        assert boost == 0.0
    
    def test_boost_partial_size_match(self):
        """Test partial size matching."""
        row = pd.Series({'fabricante': 'BD', 'ref_fabricante': 'ABC', 'mida': '25G x 1.5"'})
        features = {'mida': '25G'}
        
        boost = apply_boosts(row, features)
        assert boost == 2.5  # 50% of 5.0 for partial match
    
    def test_boost_with_missing_columns(self):
        """Test boost when row is missing some columns."""
        row = pd.Series({'fabricante': 'BD'})
        features = {'fabricante': 'BD', 'ref_fabricante': '305122', 'mida': '25G'}
        
        boost = apply_boosts(row, features)
        assert boost == 10.0  # Only fabricante matches


class TestCalculateCoincidencia:
    """Tests for calculate_coincidencia function."""
    
    def test_coincidencia_max_score(self):
        """Test coincidencia at max score."""
        result = calculate_coincidencia(1.0, 1.0, 0.0)
        assert result == 100.0
    
    def test_coincidencia_min_score(self):
        """Test coincidencia at min score."""
        result = calculate_coincidencia(0.0, 1.0, 0.0)
        assert result == 0.0
    
    def test_coincidencia_mid_score(self):
        """Test coincidencia at mid score."""
        result = calculate_coincidencia(0.5, 1.0, 0.0)
        assert result == 50.0
    
    def test_coincidencia_capped_at_100(self):
        """Test that coincidencia is capped at 100%."""
        result = calculate_coincidencia(1.5, 1.0, 0.0)
        assert result == 100.0
    
    def test_coincidencia_with_range(self):
        """Test coincidencia with non-zero min score."""
        result = calculate_coincidencia(0.75, 1.0, 0.5)
        assert result == 50.0  # (0.75 - 0.5) / (1.0 - 0.5) * 100
    
    def test_coincidencia_same_min_max(self):
        """Test coincidencia when min equals max."""
        result = calculate_coincidencia(0.5, 0.5, 0.5)
        assert result == 100.0


class TestFuseAndBoost:
    """Tests for fuse_and_boost function."""
    
    def test_fuse_and_boost_basic(self, sample_bm25_results, sample_vector_results, sample_dataframe):
        """Test basic fuse and boost."""
        results = fuse_and_boost(
            sample_bm25_results,
            sample_vector_results,
            sample_dataframe,
            features={'fabricante': 'BD'},
            row_id_column='id'
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, FusedResult) for r in results)
    
    def test_fuse_and_boost_applies_boosts(self, sample_bm25_results, sample_vector_results, sample_dataframe):
        """Test that boosts are applied."""
        features = {'fabricante': 'BD', 'ref_fabricante': '305122', 'mida': '25G'}
        results = fuse_and_boost(
            sample_bm25_results,
            sample_vector_results,
            sample_dataframe,
            features=features,
            row_id_column='id'
        )
        
        # Row 1 should have boosts (BD, 305122, 25G all match)
        row_1_result = next((r for r in results if r.row_id == 1), None)
        assert row_1_result is not None
        assert row_1_result.boost_score > 0
    
    def test_fuse_and_boost_calculates_coincidencia(self, sample_bm25_results, sample_vector_results, sample_dataframe):
        """Test that coincidencia is calculated."""
        results = fuse_and_boost(
            sample_bm25_results,
            sample_vector_results,
            sample_dataframe,
            features={'fabricante': 'BD'},
            row_id_column='id'
        )
        
        assert all(0 <= r.coincidencia_pct <= 100 for r in results)
        # Top result should have 100% coincidencia
        assert results[0].coincidencia_pct == 100.0
    
    def test_fuse_and_boost_without_features(self, sample_bm25_results, sample_vector_results, sample_dataframe):
        """Test fuse and boost without features."""
        results = fuse_and_boost(
            sample_bm25_results,
            sample_vector_results,
            sample_dataframe,
            features=None,
            row_id_column='id'
        )
        
        # Should still work, but no boosts applied
        assert all(r.boost_score == 0.0 for r in results)
        assert all(r.final_score == r.rrf_score for r in results)
    
    def test_fuse_and_boost_sorted_by_final_score(self, sample_bm25_results, sample_vector_results, sample_dataframe):
        """Test that results are sorted by final score."""
        results = fuse_and_boost(
            sample_bm25_results,
            sample_vector_results,
            sample_dataframe,
            features={'fabricante': 'BD'},
            row_id_column='id'
        )
        
        for i in range(len(results) - 1):
            assert results[i].final_score >= results[i + 1].final_score


class TestFusedResultsToDataFrame:
    """Tests for fused_results_to_dataframe function."""
    
    def test_to_dataframe_basic(self, sample_bm25_results, sample_vector_results, sample_dataframe):
        """Test converting fused results to DataFrame."""
        fused = reciprocal_rank_fusion(sample_bm25_results, sample_vector_results)
        df = fused_results_to_dataframe(fused, sample_dataframe, row_id_column='id')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(fused)
    
    def test_to_dataframe_includes_scores(self, sample_bm25_results, sample_vector_results, sample_dataframe):
        """Test that score columns are included."""
        fused = reciprocal_rank_fusion(sample_bm25_results, sample_vector_results)
        df = fused_results_to_dataframe(fused, sample_dataframe, row_id_column='id', include_scores=True)
        
        assert 'final_score' in df.columns
        assert 'coincidencia_pct' in df.columns
        assert 'rrf_score' in df.columns
        assert 'bm25_score' in df.columns
        assert 'vector_score' in df.columns
    
    def test_to_dataframe_without_scores(self, sample_bm25_results, sample_vector_results, sample_dataframe):
        """Test DataFrame without score columns."""
        fused = reciprocal_rank_fusion(sample_bm25_results, sample_vector_results)
        df = fused_results_to_dataframe(fused, sample_dataframe, row_id_column='id', include_scores=False)
        
        assert 'final_score' not in df.columns
        assert 'rrf_score' not in df.columns
    
    def test_to_dataframe_preserves_order(self, sample_bm25_results, sample_vector_results, sample_dataframe):
        """Test that DataFrame preserves result order."""
        fused = reciprocal_rank_fusion(sample_bm25_results, sample_vector_results)
        df = fused_results_to_dataframe(fused, sample_dataframe, row_id_column='id')
        
        # First row should correspond to first fused result
        assert df.iloc[0]['id'] == fused[0].row_id
    
    def test_to_dataframe_empty_results(self, sample_dataframe):
        """Test with empty fused results."""
        df = fused_results_to_dataframe([], sample_dataframe, row_id_column='id')
        
        assert df.empty


class TestFuserClass:
    """Tests for Fuser class."""
    
    def test_fuser_initialization(self):
        """Test Fuser initialization."""
        fuser = Fuser(k=60)
        assert fuser.k == 60
        assert fuser.boost_config is not None
    
    def test_fuser_custom_boost_config(self):
        """Test Fuser with custom boost config."""
        custom_config = {'fabricante': 20.0, 'ref_fabricante': 30.0, 'mida': 10.0}
        fuser = Fuser(boost_config=custom_config)
        
        assert fuser.boost_config == custom_config
    
    def test_fuser_fuse_method(self, sample_bm25_results, sample_vector_results, sample_dataframe):
        """Test Fuser.fuse method."""
        fuser = Fuser()
        results = fuser.fuse(
            sample_bm25_results,
            sample_vector_results,
            df=sample_dataframe,
            features={'fabricante': 'BD'},
            row_id_column='id'
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
    
    def test_fuser_fuse_without_df(self, sample_bm25_results, sample_vector_results):
        """Test Fuser.fuse without DataFrame."""
        fuser = Fuser()
        results = fuser.fuse(sample_bm25_results, sample_vector_results)
        
        assert isinstance(results, list)
        assert all(r.boost_score == 0.0 for r in results)
    
    def test_fuser_set_boost_config(self):
        """Test setting boost configuration."""
        fuser = Fuser()
        new_config = {'fabricante': 25.0, 'ref_fabricante': 35.0, 'mida': 12.0}
        fuser.set_boost_config(new_config)
        
        assert fuser.boost_config == new_config
    
    def test_fuser_get_boost_config(self):
        """Test getting boost configuration."""
        custom_config = {'fabricante': 20.0, 'ref_fabricante': 30.0, 'mida': 10.0}
        fuser = Fuser(boost_config=custom_config)
        
        config = fuser.get_boost_config()
        assert config == custom_config
        # Should return a copy
        config['fabricante'] = 999.0
        assert fuser.boost_config['fabricante'] == 20.0


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_result_each(self):
        """Test with single result from each source."""
        bm25 = [(1, 10.0)]
        vector = [(1, 0.9)]
        
        results = reciprocal_rank_fusion(bm25, vector)
        assert len(results) == 1
        assert results[0].row_id == 1
    
    def test_no_overlap_in_results(self):
        """Test when BM25 and vector have no overlapping results."""
        bm25 = [(1, 10.0), (2, 8.0)]
        vector = [(3, 0.9), (4, 0.8)]
        
        results = reciprocal_rank_fusion(bm25, vector)
        assert len(results) == 4
    
    def test_complete_overlap_in_results(self):
        """Test when BM25 and vector have complete overlap."""
        bm25 = [(1, 10.0), (2, 8.0)]
        vector = [(1, 0.9), (2, 0.8)]
        
        results = reciprocal_rank_fusion(bm25, vector)
        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
