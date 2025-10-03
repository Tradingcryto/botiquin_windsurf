"""
Batch processing module for processing multiple queries from tender files.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd

from .search import SearchEngine
from .index_bm25 import BM25Index
from .io_excel import read_catalog, read_tender, write_excel


logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Batch processor for processing tender queries against catalog.
    """
    
    def __init__(
        self,
        search_engine: SearchEngine,
        match_threshold: float = 50.0,
        top_k: int = 5
    ):
        """
        Initialize the batch processor.
        
        Args:
            search_engine: SearchEngine instance for performing searches.
            match_threshold: Minimum coincidencia percentage for a valid match.
            top_k: Number of top results to return per query.
        """
        self.search_engine = search_engine
        self.match_threshold = match_threshold
        self.top_k = top_k
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'with_match': 0,
            'without_match': 0,
            'above_threshold': 0,
            'errors': []
        }
    
    def process_one(
        self,
        row_index: int,
        tender_row: pd.Series,
        query_column: str = 'product_name',
        tender_id_column: str = 'tender_id'
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Process a single tender row.
        
        Args:
            row_index: Index of the row being processed.
            tender_row: Series containing tender data.
            query_column: Name of the column containing the query text.
            tender_id_column: Name of the column containing tender ID.
        
        Returns:
            Tuple of (results_df, error_message).
            If successful, error_message is None.
            If failed, results_df contains the tender info with no matches.
        """
        try:
            # Get tender ID and query text
            tender_id = tender_row.get(tender_id_column, f"ROW_{row_index}")
            query_text = tender_row.get(query_column, "")
            
            if not query_text or pd.isna(query_text):
                logger.warning(f"Row {row_index}: Empty query text for tender {tender_id}")
                # Create a result row with error info
                result = pd.DataFrame([{
                    'tender_id': tender_id,
                    'tender_query': '',
                    'row_index': row_index,
                    'error': 'Empty query text',
                    'coincidencia_pct': 0.0
                }])
                return result, "Empty query text"
            
            logger.info(f"Processing row {row_index}: tender_id={tender_id}, query='{query_text}'")
            
            # Perform search
            results = self.search_engine.search_one(
                lic_ref=str(tender_id),
                query_text=str(query_text),
                prefer_cat=True,
                top_k=self.top_k
            )
            
            if results.empty:
                logger.warning(f"Row {row_index}: No results found for tender {tender_id}")
                # Create a result row with no matches
                result = pd.DataFrame([{
                    'tender_id': tender_id,
                    'tender_query': query_text,
                    'row_index': row_index,
                    'error': 'No results found',
                    'coincidencia_pct': 0.0
                }])
                # Add other tender columns
                for col in tender_row.index:
                    if col not in [query_column, tender_id_column]:
                        result[f'tender_{col}'] = tender_row[col]
                return result, None
            
            # Add tender information to results
            results['tender_id'] = tender_id
            results['tender_query'] = query_text
            results['row_index'] = row_index
            results['error'] = None  # No error
            
            # Add other tender columns if available
            for col in tender_row.index:
                if col not in results.columns and col not in [query_column, tender_id_column]:
                    results[f'tender_{col}'] = tender_row[col]
            
            logger.info(f"Row {row_index}: Found {len(results)} results, top match: {results.iloc[0]['coincidencia_pct']:.1f}%")
            
            return results, None
            
        except Exception as e:
            error_msg = f"Error processing row {row_index}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Create a result row with error info
            result = pd.DataFrame([{
                'tender_id': tender_row.get(tender_id_column, f"ROW_{row_index}"),
                'tender_query': tender_row.get(query_column, ""),
                'row_index': row_index,
                'error': error_msg,
                'coincidencia_pct': 0.0
            }])
            return result, error_msg
    
    def process_batch(
        self,
        tender_df: pd.DataFrame,
        query_column: str = 'product_name',
        tender_id_column: str = 'tender_id'
    ) -> pd.DataFrame:
        """
        Process all rows in a tender DataFrame.
        
        Args:
            tender_df: DataFrame containing tender queries.
            query_column: Name of the column containing query text.
            tender_id_column: Name of the column containing tender IDs.
        
        Returns:
            DataFrame with all results combined, including rows with no matches.
        """
        logger.info(f"Starting batch processing of {len(tender_df)} rows")
        start_time = time.time()
        
        all_results = []
        
        for idx, row in tender_df.iterrows():
            self.stats['total_processed'] += 1
            
            results, error = self.process_one(
                row_index=idx,
                tender_row=row,
                query_column=query_column,
                tender_id_column=tender_id_column
            )
            
            if error is not None:
                self.stats['failed'] += 1
                self.stats['errors'].append({
                    'row_index': idx,
                    'tender_id': row.get(tender_id_column, f"ROW_{idx}"),
                    'error': error
                })
                # Even if there was an error, we still want to include the row in the output
                if results is not None:
                    all_results.append(results)
            else:
                self.stats['successful'] += 1
                
                if results is not None:
                    if not results.empty and 'coincidencia_pct' in results.columns and not results['coincidencia_pct'].isna().all():
                        self.stats['with_match'] += 1
                        
                        # Check if top result is above threshold
                        if results.iloc[0]['coincidencia_pct'] >= self.match_threshold:
                            self.stats['above_threshold'] += 1
                    else:
                        self.stats['without_match'] += 1
                    
                    all_results.append(results)
                else:
                    self.stats['without_match'] += 1
                    # Create a default result for this case
                    result = pd.DataFrame([{
                        'tender_id': row.get(tender_id_column, f"ROW_{idx}"),
                        'tender_query': row.get(query_column, ""),
                        'row_index': idx,
                        'error': 'No results generated',
                        'coincidencia_pct': 0.0
                    }])
                    all_results.append(result)
        
        # Combine all results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
        else:
            combined_results = pd.DataFrame(columns=[
                'tender_id', 'tender_query', 'row_index', 'error', 'coincidencia_pct'
            ])
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"Batch processing completed in {elapsed_time:.2f} seconds")
        self._log_summary()
        
        return combined_results
    
    def _log_summary(self):
        """Log processing summary."""
        logger.info("=" * 80)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total rows processed: {self.stats['total_processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"With matches: {self.stats['with_match']}")
        logger.info(f"Without matches: {self.stats['without_match']}")
        logger.info(f"Above threshold ({self.match_threshold}%): {self.stats['above_threshold']}")
        
        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
            match_rate = (self.stats['with_match'] / self.stats['total_processed']) * 100
            threshold_rate = (self.stats['above_threshold'] / self.stats['total_processed']) * 100
            
            logger.info(f"Success rate: {success_rate:.1f}%")
            logger.info(f"Match rate: {match_rate:.1f}%")
            logger.info(f"Above threshold rate: {threshold_rate:.1f}%")
        
        if self.stats['errors']:
            logger.info(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error_info in self.stats['errors'][:5]:  # Show first 5 errors
                logger.info(f"  Row {error_info['row_index']}: {error_info['error']}")
            if len(self.stats['errors']) > 5:
                logger.info(f"  ... and {len(self.stats['errors']) - 5} more errors")
        
        logger.info("=" * 80)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics.
        """
        stats = self.stats.copy()
        
        if stats['total_processed'] > 0:
            stats['success_rate'] = (stats['successful'] / stats['total_processed']) * 100
            stats['match_rate'] = (stats['with_match'] / stats['total_processed']) * 100
            stats['threshold_rate'] = (stats['above_threshold'] / stats['total_processed']) * 100
        else:
            stats['success_rate'] = 0.0
            stats['match_rate'] = 0.0
            stats['threshold_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'with_match': 0,
            'without_match': 0,
            'above_threshold': 0,
            'errors': []
        }


def format_output_dataframe(
    results_df: pd.DataFrame,
    include_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Format results DataFrame for output.
    
    Args:
        results_df: Raw results DataFrame.
        include_columns: Optional list of columns to include.
                        If None, includes all columns.
    
    Returns:
        Formatted DataFrame.
    """
    if results_df.empty:
        return results_df
    
    # Define standard output columns
    standard_columns = [
        'tender_id',
        'tender_query',
        'row_index',
        'id',
        'product_name',
        'category',
        'fabricante',
        'ref_fabricante',
        'mida',
        'final_score',
        'coincidencia_pct',
        'rrf_score',
        'bm25_score',
        'vector_score',
        'boost_score'
    ]
    
    # Select columns that exist in the DataFrame
    if include_columns is None:
        output_columns = [col for col in standard_columns if col in results_df.columns]
        # Add any remaining columns not in standard list
        for col in results_df.columns:
            if col not in output_columns:
                output_columns.append(col)
    else:
        output_columns = [col for col in include_columns if col in results_df.columns]
    
    formatted_df = results_df[output_columns].copy()
    
    # Round numeric columns
    numeric_columns = ['final_score', 'coincidencia_pct', 'rrf_score', 'bm25_score', 'vector_score', 'boost_score']
    for col in numeric_columns:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].round(4)
    
    return formatted_df


def save_results(
    results_df: pd.DataFrame,
    output_path: str,
    sheet_name: str = "Matching Results"
) -> None:
    """
    Save results to Excel file.
    
    Args:
        results_df: Results DataFrame to save.
        output_path: Path to output Excel file.
        sheet_name: Name of the Excel sheet.
    """
    logger.info(f"Saving {len(results_df)} results to {output_path}")
    
    # Format output
    formatted_df = format_output_dataframe(results_df)
    
    # Save to Excel
    write_excel(formatted_df, output_path, sheet_name=sheet_name, index=False)
    
    logger.info(f"Results saved successfully to {output_path}")
