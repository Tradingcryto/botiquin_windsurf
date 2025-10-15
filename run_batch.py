"""
Batch processing script for matching tender queries against catalog.

Usage:
    python run_batch.py --catalog data/input/catalogo.xlsx --query data/input/licitacion.xlsx --out data/output/resultado_matching.xlsx
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src.botiquin_windsurf.io_excel import read_catalog, read_tender
from src.botiquin_windsurf.index_bm25 import BM25Index
from src.botiquin_windsurf.search import SearchEngine, configure_logging
from src.botiquin_windsurf.batch import BatchProcessor, save_results


def setup_logging(log_file: str = None, verbose: bool = False):
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional path to log file.
        verbose: If True, set logging level to DEBUG.
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    handlers = [console_handler]
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=level,
        handlers=handlers
    )


def main():
    """Main batch processing function."""
    parser = argparse.ArgumentParser(
        description='Batch process tender queries against product catalog'
    )
    
    parser.add_argument(
        '--catalog',
        type=str,
        required=True,
        help='Path to catalog Excel file (catalogo.xlsx)'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Path to tender/query Excel file (licitacion.xlsx)'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Path to output Excel file (resultado_matching.xlsx)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file (default: config.yaml in project root)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top results per query (default: 5)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=50.0,
        help='Minimum coincidencia percentage for valid match (default: 50.0)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file=args.log_file, verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("BATCH MATCHING PROCESS STARTED")
    logger.info("=" * 80)
    logger.info(f"Catalog file: {args.catalog}")
    logger.info(f"Query file: {args.query}")
    logger.info(f"Output file: {args.out}")
    logger.info(f"Top-k results: {args.top_k}")
    logger.info(f"Match threshold: {args.threshold}%")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # Step 1: Load catalog
        logger.info("Step 1: Loading catalog file...")
        catalog_df = read_catalog(args.catalog, config_path=args.config)
        logger.info(f"Loaded {len(catalog_df)} products from catalog")
        logger.info(f"Catalog columns: {list(catalog_df.columns)}")
        
        # Step 2: Load tender/query file
        logger.info("\nStep 2: Loading tender/query file...")
        tender_df = read_tender(args.query, config_path=args.config)
        logger.info(f"Loaded {len(tender_df)} queries from tender file")
        logger.info(f"Tender columns: {list(tender_df.columns)}")
        
        # Step 3: Build BM25 index
        logger.info("\nStep 3: Building BM25 index...")
        bm25_index = BM25Index()
        
        # Determine column names (use mapped names from config)
        text_column = 'product_name'
        category_column = 'category'
        row_id_column = 'product_id'
        
        # Fallback to first available columns if mapped names don't exist
        if text_column not in catalog_df.columns and len(catalog_df.columns) > 0:
            text_column = catalog_df.columns[1] if len(catalog_df.columns) > 1 else catalog_df.columns[0]
            logger.warning(f"Using '{text_column}' as text column")
        
        if category_column not in catalog_df.columns:
            category_column = None
            logger.warning("Category column not found, will use global index only")
        
        if row_id_column not in catalog_df.columns:
            row_id_column = None
            logger.warning("Row ID column not found, will use DataFrame index")
        
        bm25_index.build_indexes(
            catalog_df,
            text_column=text_column,
            category_column=category_column,
            row_id_column=row_id_column
        )
        logger.info(f"BM25 index built: {bm25_index.get_global_size()} documents")
        if category_column:
            logger.info(f"Categories indexed: {len(bm25_index.get_categories())}")
        
        # Step 4: Create search engine
        logger.info("\nStep 4: Initializing search engine...")
        search_engine = SearchEngine(
            catalog_df=catalog_df,
            bm25_index=bm25_index,
            text_column=text_column,
            category_column=category_column if category_column else 'category',
            row_id_column=row_id_column if row_id_column else 'id'
        )
        logger.info("Search engine initialized")
        
        # Step 5: Create batch processor
        logger.info("\nStep 5: Initializing batch processor...")
        batch_processor = BatchProcessor(
            search_engine=search_engine,
            match_threshold=args.threshold,
            top_k=args.top_k
        )
        logger.info(f"Batch processor initialized (threshold: {args.threshold}%, top-k: {args.top_k})")
        
        # Step 6: Process batch
        logger.info("\nStep 6: Processing batch queries...")
        logger.info("-" * 80)
        
        # Determine query column name
        query_column = 'product_name'
        tender_id_column = 'tender_id'
        
        if query_column not in tender_df.columns and len(tender_df.columns) > 0:
            query_column = tender_df.columns[1] if len(tender_df.columns) > 1 else tender_df.columns[0]
            logger.warning(f"Using '{query_column}' as query column")
        
        if tender_id_column not in tender_df.columns and len(tender_df.columns) > 0:
            tender_id_column = tender_df.columns[0]
            logger.warning(f"Using '{tender_id_column}' as tender ID column")
        
        results_df = batch_processor.process_batch(
            tender_df,
            query_column=query_column,
            tender_id_column=tender_id_column
        )
        
        logger.info("-" * 80)
        logger.info(f"Batch processing completed: {len(results_df)} total results")
        
        # Step 7: Save results
        logger.info("\nStep 7: Saving results...")
        
        # Create output directory if it doesn't exist
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_results(results_df, str(output_path), sheet_name="Matching Results")
        logger.info(f"Results saved to: {output_path}")
        
        # Step 8: Print final summary
        elapsed_time = time.time() - start_time
        stats = batch_processor.get_stats()
        
        logger.info("\n" + "=" * 80)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total queries processed: {stats['total_processed']}")
        logger.info(f"Successful: {stats['successful']} ({stats['success_rate']:.1f}%)")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"With matches: {stats['with_match']} ({stats['match_rate']:.1f}%)")
        logger.info(f"Above threshold (≥{args.threshold}%): {stats['above_threshold']} ({stats['threshold_rate']:.1f}%)")
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        logger.info(f"Average time per query: {elapsed_time / max(stats['total_processed'], 1):.2f} seconds")
        logger.info("=" * 80)
        logger.info("BATCH MATCHING PROCESS COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"FATAL ERROR: {str(e)}", exc_info=True)
        logger.error("=" * 80)
        logger.error("BATCH MATCHING PROCESS FAILED")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())