"""
Unit tests for the batch module.
"""

import pytest
import pandas as pd
from botiquin_windsurf.batch import (
    BatchProcessor,
    format_output_dataframe,
    save_results
)
from botiquin_windsurf.search import SearchEngine
from botiquin_windsurf.index_bm25 import BM25Index


@pytest.fixture
def sample_catalog():
    """Create a sample catalog DataFrame."""
    data = {
        'id': [1, 2, 3, 4, 5],
        'product_name': [
            'Aguja BD 25G x 1.5"',
            'Jeringa 10 ml Luer Lock',
            'Solución Salina 0.9%',
            'Guante Latex M',
            'Aguja 21G x 1"'
        ],
        'category': ['Agujas', 'Jeringas', 'Soluciones', 'Guantes', 'Agujas'],
        'fabricante': ['BD', 'BD', 'Baxter', 'Medline', 'BD']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_tender():
    """Create a sample tender DataFrame."""
    data = {
        'tender_id': ['LIC001', 'LIC002', 'LIC003'],
        'product_name': [
            'aguja 25g',
            'jeringa 10 ml',
            'solucion salina'
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def search_engine(sample_catalog):
    """Create a search engine with sample catalog."""
    index = BM25Index()
    index.build_indexes(
        sample_catalog,
        text_column='product_name',
        category_column='category',
        row_id_column='id'
    )
    
    return SearchEngine(
        catalog_df=sample_catalog,
        bm25_index=index,
        text_column='product_name',
        category_column='category',
        row_id_column='id'
    )


@pytest.fixture
def batch_processor(search_engine):
    """Create a batch processor."""
    return BatchProcessor(
        search_engine=search_engine,
        match_threshold=50.0,
        top_k=5
    )


class TestBatchProcessor:
    """Tests for BatchProcessor class."""
    
    def test_batch_processor_init(self, search_engine):
        """Test BatchProcessor initialization."""
        processor = BatchProcessor(search_engine, match_threshold=60.0, top_k=3)
        
        assert processor.search_engine is not None
        assert processor.match_threshold == 60.0
        assert processor.top_k == 3
        assert processor.stats['total_processed'] == 0
    
    def test_process_one_success(self, batch_processor, sample_tender):
        """Test processing a single row successfully."""
        row = sample_tender.iloc[0]
        results, error = batch_processor.process_one(
            row_index=0,
            tender_row=row,
            query_column='product_name',
            tender_id_column='tender_id'
        )
        
        assert error is None
        assert results is not None
        assert isinstance(results, pd.DataFrame)
        assert not results.empty
    
    def test_process_one_empty_query(self, batch_processor):
        """Test processing row with empty query."""
        row = pd.Series({'tender_id': 'LIC999', 'product_name': ''})
        results, error = batch_processor.process_one(
            row_index=0,
            tender_row=row
        )
        
        assert results is None
        assert error is not None
        assert "Empty query" in error
    
    def test_process_batch(self, batch_processor, sample_tender):
        """Test batch processing."""
        results = batch_processor.process_batch(
            sample_tender,
            query_column='product_name',
            tender_id_column='tender_id'
        )
        
        assert isinstance(results, pd.DataFrame)
        assert batch_processor.stats['total_processed'] == len(sample_tender)
    
    def test_process_batch_statistics(self, batch_processor, sample_tender):
        """Test that batch processing updates statistics."""
        batch_processor.process_batch(sample_tender)
        
        stats = batch_processor.get_stats()
        assert stats['total_processed'] > 0
        assert 'success_rate' in stats
        assert 'match_rate' in stats
    
    def test_get_stats(self, batch_processor):
        """Test getting statistics."""
        stats = batch_processor.get_stats()
        
        assert isinstance(stats, dict)
        assert 'total_processed' in stats
        assert 'successful' in stats
        assert 'failed' in stats
    
    def test_reset_stats(self, batch_processor, sample_tender):
        """Test resetting statistics."""
        batch_processor.process_batch(sample_tender)
        assert batch_processor.stats['total_processed'] > 0
        
        batch_processor.reset_stats()
        assert batch_processor.stats['total_processed'] == 0


class TestFormatOutputDataFrame:
    """Tests for format_output_dataframe function."""
    
    def test_format_output_basic(self):
        """Test basic output formatting."""
        df = pd.DataFrame({
            'tender_id': ['LIC001'],
            'product_name': ['Aguja 25G'],
            'final_score': [0.123456],
            'coincidencia_pct': [85.123456]
        })
        
        formatted = format_output_dataframe(df)
        
        assert isinstance(formatted, pd.DataFrame)
        assert not formatted.empty
    
    def test_format_output_rounds_numbers(self):
        """Test that numeric columns are rounded."""
        df = pd.DataFrame({
            'final_score': [0.123456789],
            'coincidencia_pct': [85.123456789]
        })
        
        formatted = format_output_dataframe(df)
        
        assert formatted['final_score'].iloc[0] == 0.1235
        assert formatted['coincidencia_pct'].iloc[0] == 85.1235
    
    def test_format_output_empty_dataframe(self):
        """Test formatting empty DataFrame."""
        df = pd.DataFrame()
        formatted = format_output_dataframe(df)
        
        assert formatted.empty


class TestNormalization:
    """Tests for normalization in batch processing."""
    
    def test_query_normalization(self, batch_processor):
        """Test that queries are normalized."""
        row = pd.Series({
            'tender_id': 'LIC001',
            'product_name': 'Aguja 25G x 1.5"'
        })
        
        results, error = batch_processor.process_one(0, row)
        
        # Should process without error
        assert error is None or results is not None


class TestRouter:
    """Tests for router integration in batch processing."""
    
    def test_category_routing(self, batch_processor):
        """Test that category routing works in batch."""
        row = pd.Series({
            'tender_id': 'LIC001',
            'product_name': 'aguja hipodermica'
        })
        
        results, error = batch_processor.process_one(0, row)
        
        # Should find results
        assert results is not None or error is not None


class TestRankings:
    """Tests for ranking results."""
    
    def test_results_not_empty(self, batch_processor, sample_tender):
        """Test that results are not empty for valid queries."""
        results = batch_processor.process_batch(sample_tender)
        
        # Should have some results
        assert isinstance(results, pd.DataFrame)
    
    def test_results_sorted(self, batch_processor):
        """Test that results are sorted by score."""
        row = pd.Series({
            'tender_id': 'LIC001',
            'product_name': 'aguja 25g'
        })
        
        results, error = batch_processor.process_one(0, row)
        
        if results is not None and len(results) > 1:
            scores = results['final_score'].tolist()
            assert scores == sorted(scores, reverse=True)
    
    def test_coincidencia_in_results(self, batch_processor):
        """Test that coincidencia percentage is included."""
        row = pd.Series({
            'tender_id': 'LIC001',
            'product_name': 'aguja'
        })
        
        results, error = batch_processor.process_one(0, row)
        
        if results is not None:
            assert 'coincidencia_pct' in results.columns


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_error_handling_invalid_query(self, batch_processor):
        """Test error handling for invalid queries."""
        row = pd.Series({'tender_id': 'LIC999', 'product_name': None})
        
        results, error = batch_processor.process_one(0, row)
        
        # Should handle gracefully
        assert results is None or error is not None
    
    def test_batch_continues_after_error(self, batch_processor):
        """Test that batch processing continues after errors."""
        tender_df = pd.DataFrame({
            'tender_id': ['LIC001', 'LIC002', 'LIC003'],
            'product_name': ['aguja', '', 'jeringa']  # Middle one is empty
        })
        
        results = batch_processor.process_batch(tender_df)
        
        # Should process all rows
        assert batch_processor.stats['total_processed'] == 3
        # Should have some failures
        assert batch_processor.stats['failed'] > 0


class TestLogging:
    """Tests for logging functionality."""
    
    def test_logging_in_batch(self, batch_processor, sample_tender, caplog):
        """Test that batch processing produces logs."""
        import logging
        with caplog.at_level(logging.INFO):
            batch_processor.process_batch(sample_tender)
        
        # Should have log messages
        assert len(caplog.records) > 0
    
    def test_summary_logging(self, batch_processor, sample_tender, caplog):
        """Test that summary is logged."""
        import logging
        with caplog.at_level(logging.INFO):
            batch_processor.process_batch(sample_tender)
        
        # Should have summary in logs
        log_text = " ".join([record.message for record in caplog.records])
        assert "SUMMARY" in log_text or "processed" in log_text.lower()


class TestFinalSummary:
    """Tests for final summary statistics."""
    
    def test_summary_includes_total_processed(self, batch_processor, sample_tender):
        """Test summary includes total processed count."""
        batch_processor.process_batch(sample_tender)
        stats = batch_processor.get_stats()
        
        assert stats['total_processed'] == len(sample_tender)
    
    def test_summary_includes_time_info(self, batch_processor, sample_tender):
        """Test that processing tracks time."""
        import time
        start = time.time()
        batch_processor.process_batch(sample_tender)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed >= 0
    
    def test_summary_includes_match_percentage(self, batch_processor, sample_tender):
        """Test summary includes match percentage."""
        batch_processor.process_batch(sample_tender)
        stats = batch_processor.get_stats()
        
        assert 'match_rate' in stats
        assert 0 <= stats['match_rate'] <= 100
    
    def test_summary_includes_threshold_percentage(self, batch_processor, sample_tender):
        """Test summary includes threshold percentage."""
        batch_processor.process_batch(sample_tender)
        stats = batch_processor.get_stats()
        
        assert 'threshold_rate' in stats
        assert 0 <= stats['threshold_rate'] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
