import re

"""
Search module with comprehensive search pipeline for product matching.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
from dataclasses import dataclass

from .normalize import normalize_product_description, extract_codes_sizes
from .router import route_categories, CategoryMatch
from .index_bm25 import BM25Index
from .fuse import fuse_and_boost, fused_results_to_dataframe, Fuser
from .taxonomy import get_taxonomy


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SearchContext:
    """
    Context information for a search operation.
    """
    lic_ref: str
    query_text: str
    query_normalized: str
    extracted_codes: List[str]
    extracted_sizes: List[str]
    routed_categories: List[CategoryMatch]
    prefer_category: bool
    
    def __repr__(self) -> str:
        return f"SearchContext(lic_ref='{self.lic_ref}', categories={len(self.routed_categories)})"


class SearchEngine:
    """
    Main search engine for product matching.
    """
    
    def __init__(
        self,
        catalog_df: pd.DataFrame,
        bm25_index: BM25Index,
        vector_index: Optional[Any] = None,
        fuser: Optional[Fuser] = None,
        text_column: str = 'product_name',
        category_column: str = 'category',
        row_id_column: str = 'id'
    ):
        """
        Initialize the search engine.
        
        Args:
            catalog_df: Catalog DataFrame with product data.
            bm25_index: Built BM25 index.
            vector_index: Optional vector index (e.g., FAISS, sentence-transformers).
            fuser: Optional Fuser instance for result fusion.
            text_column: Name of the text column for searching.
            category_column: Name of the category column.
            row_id_column: Name of the row ID column.
        """
        self.catalog_df = catalog_df
        self.bm25_index = bm25_index
        self.vector_index = vector_index
        self.fuser = fuser or Fuser()
        self.text_column = text_column
        self.category_column = category_column
        self.row_id_column = row_id_column
        self.code_cols = [c for c in ("product_id", "ref_fabricante") if c in self.catalog_df.columns]

        logger.info(f"SearchEngine initialized with {len(catalog_df)} products")
    
    def _normalize_query(self, query_text: str) -> str:
        """
        Normalize query text.
        
        Args:
            query_text: Raw query text.
        
        Returns:
            Normalized query text.
        """
        if not query_text or not isinstance(query_text, str):
            logger.warning("Received empty or invalid query text")
            return ""
            
        logger.debug(f"Original query: '{query_text}'")
        
        try:
            # First, preserve the original for logging
            original = query_text.strip()
            
            # Apply normalization
            normalized = normalize_product_description(query_text)
            
            # Log the transformation
            if original != normalized:
                logger.debug(f"Normalized query: '{original}' -> '{normalized}'")
            else:
                logger.debug("No normalization changes applied")
                
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing query '{query_text}': {str(e)}")
            # Return the original text if normalization fails
            return query_text.strip()
    
    def _extract_features(self, query_text: str) -> Dict[str, Any]:
        """
        Extract features from query text.
        
        Args:
            query_text: Query text (preferably normalized).
        
        Returns:
            Dictionary with extracted features.
        """
        logger.debug(f"Extracting features from: '{query_text}'")
        codes, sizes = extract_codes_sizes(query_text)
        
        features = {
            'codes': codes,
            'sizes': sizes,
            'fabricante': None,
            'ref_fabricante': None,
            'mida': sizes[0] if sizes else None
        }
        
        logger.debug(f"Extracted features: codes={codes}, sizes={sizes}")
        return features
    
    def _exact_match(self, query_normalized: str) -> Optional[pd.DataFrame]:
        """
        Attempt exact match on product names or references.
        
        Args:
            query_normalized: Normalized query text.
        
        Returns:
            DataFrame with exact matches or None.
        """
        logger.debug(f"Attempting exact match for: '{query_normalized}'")
        
        # Try exact match on normalized product names
        if self.text_column in self.catalog_df.columns:
            # Normalize catalog text column for comparison
            catalog_normalized = self.catalog_df[self.text_column].apply(
                lambda x: normalize_product_description(str(x)) if pd.notna(x) else ""
            )
            exact_matches = self.catalog_df[catalog_normalized == query_normalized]
            
            if not exact_matches.empty:
                logger.info(f"Found {len(exact_matches)} exact matches")
                return exact_matches
        
        logger.debug("No exact matches found")
        return None
    
    def _route_categories(self, query_normalized: str) -> List[CategoryMatch]:
        """
        Route query to relevant categories.
        
        Args:
            query_normalized: Normalized query text.
        
        Returns:
            List of CategoryMatch objects.
        """
        logger.debug(f"Routing categories for: '{query_normalized}'")
        categories = route_categories(query_normalized, min_score=1)
        
        if categories:
            logger.info(f"Routed to {len(categories)} categories: {[c.category_name for c in categories]}")
        else:
            logger.info("No categories matched, will search globally")
        
        return categories
    
    def _search_bm25(
        self,
        query_normalized: str,
        categories: Optional[List[str]] = None,
        top_k: int = 20
    ) -> List[Tuple[int, float]]:
        """
        Perform BM25 search.
        
        Args:
            query_normalized: Normalized query text.
            categories: Optional list of categories to search within.
            top_k: Number of top results to retrieve.
        
        Returns:
            List of (row_id, score) tuples.
        """
        if not query_normalized:
            logger.warning("Empty query provided to BM25 search")
            return []
            
        try:
            if categories:
                logger.info(f"BM25 search in categories: {categories} for query: '{query_normalized}'")
                results = self.bm25_index.search_multiple_categories(
                    query_normalized,
                    categories,
                    top_k=top_k * 2  # Get more results for better filtering
                )
                bm25_results = [(r.row_id, r.score) for r in results]
                
                # Log top results for debugging
                if bm25_results:
                    top_scores = ", ".join(f"{score:.2f}" for _, score in bm25_results[:3])
                    logger.debug(f"Top BM25 scores in categories: {top_scores}...")
                else:
                    logger.debug("No BM25 results found in specified categories")
                    
            else:
                logger.info(f"BM25 global search for query: '{query_normalized}'")
                results = self.bm25_index.search_global(query_normalized, top_k=top_k * 2)
                bm25_results = [(r.row_id, r.score) for r in results]
                
                # Log top results for debugging
                if bm25_results:
                    top_scores = ", ".join(f"{score:.2f}" for _, score in bm25_results[:3])
                    logger.debug(f"Top BM25 global scores: {top_scores}...")
                else:
                    logger.debug("No BM25 results found in global search")
            
            logger.info(f"BM25 search returned {len(bm25_results)} results")
            return bm25_results
            
        except Exception as e:
            logger.error(f"Error in BM25 search for query '{query_normalized}': {str(e)}", exc_info=True)
            return []
    
    def _search_vector(
        self,
        query_text: str,
        categories: Optional[List[str]] = None,
        top_k: int = 20
    ) -> List[Tuple[int, float]]:
        """
        Perform vector search (placeholder for future implementation).
        
        Args:
            query_text: Query text.
            categories: Optional list of categories to search within.
            top_k: Number of top results to retrieve.
        
        Returns:
            List of (row_id, score) tuples.
        """
        if self.vector_index is None:
            logger.debug("Vector search not available (no vector index)")
            return []
        
        logger.debug("Vector search (placeholder)")
        # TODO: Implement vector search when vector_index is available
        return []
    
    def _fuse_results(
        self,
        bm25_results: List[Tuple[int, float]],
        vector_results: List[Tuple[int, float]],
        features: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Fuse BM25 and vector results with boosting.
        
        Args:
            bm25_results: BM25 search results.
            vector_results: Vector search results.
            features: Extracted features for boosting.
        
        Returns:
            DataFrame with fused and boosted results.
        """
        logger.debug(f"Fusing {len(bm25_results)} BM25 + {len(vector_results)} vector results")
        
        fused = fuse_and_boost(
            bm25_results,
            vector_results,
            self.catalog_df,
            features=features,
            row_id_column=self.row_id_column
        )
        
        logger.info(f"Fusion produced {len(fused)} results")
        
        # Convert to DataFrame
        result_df = fused_results_to_dataframe(
            fused,
            self.catalog_df,
            row_id_column=self.row_id_column,
            include_scores=True
        )
        
        return result_df
    
    def search_one(
        self,
        lic_ref: str,
        query_text: str,
        prefer_cat: bool = True,
        top_k: int = 5,
        exact_match_first: bool = True,
        debug: bool = True  # New parameter for debug mode
    ) -> pd.DataFrame:
        """
        Perform comprehensive search for a single query.
        
        Search pipeline:
        1. Normalize query text
        2. Extract features (codes, sizes)
        3. Attempt exact match (optional)
        4. Route to categories
        5. Search BM25 (category shards or global)
        6. Search vector (category shards or global)
        7. Fuse results with RRF
        8. Apply feature boosts
        9. Calculate coincidencia percentage
        10. Return top-k results
        
        Args:
            lic_ref: Licitación reference ID for logging.
            query_text: Query text to search.
            prefer_cat: If True, prefer category-specific search.
            top_k: Number of top results to return (default: 5).
            exact_match_first: If True, attempt exact match before full search.
        
        Returns:
            DataFrame with top-k results and scores.
        
        Examples:
            >>> engine = SearchEngine(catalog_df, bm25_index)
            >>> results = engine.search_one("LIC001", "aguja hipodermica 25g", prefer_cat=True)
            >>> print(results[['product_name', 'coincidencia_pct']].head())
        """
        logger.info(f"=" * 80)
        logger.info(f"Starting search for lic_ref='{lic_ref}', query='{query_text}'")
        logger.info(f"Parameters: prefer_cat={prefer_cat}, top_k={top_k}")
        
        # Enable debug logging for this search if requested
        if debug:
            import logging
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)
            
        logger.info(f"\n{'='*80}\nStarting search for lic_ref='{lic_ref}'") 
        logger.info(f"Original query: '{query_text}'")
        
        # Step 1: Normalize query
        query_normalized = self._normalize_query(query_text)
        if not query_normalized:
            logger.warning("Query normalized to empty string")
            return pd.DataFrame()
            
        logger.info(f"Normalized query: '{query_normalized}'")
        
        # Step 2: Extract features
        features = self._extract_features(query_normalized)
        codes = features['codes']
        sizes = features['sizes']
        
        logger.info(f"Extracted features - Codes: {codes}, Sizes: {sizes}")
        
        # Step 3: Exact match (optional)
        if exact_match_first:
            exact_matches = self._exact_match(query_normalized)
            if exact_matches is not None and not exact_matches.empty:
                logger.info(f"Returning {len(exact_matches)} exact matches")
                result = exact_matches.head(top_k).copy()
                result['coincidencia_pct'] = 100.0
                result['final_score'] = 1.0
                return result
        
        # Step 4: Route to categories
        routed_categories = self._route_categories(query_normalized)
        
        # Log category routing
        if routed_categories:
            logger.info(f"Routed to {len(routed_categories)} categories:")
            for cat in routed_categories[:3]:  # Log top 3 categories
                logger.info(f"  - {cat.category_name} (score: {cat.score:.2f})")
            if len(routed_categories) > 3:
                logger.info(f"  ... and {len(routed_categories) - 3} more")
        else:
            logger.info("No categories matched, will use global search")
        
        # Create search context
        context = SearchContext(
            lic_ref=lic_ref,
            query_text=query_text,
            query_normalized=query_normalized,
            extracted_codes=codes,
            extracted_sizes=sizes,
            routed_categories=routed_categories,
            prefer_category=prefer_cat
        )
        logger.debug(f"Search context: {context}")
        
        # Determine search strategy
        if prefer_cat and routed_categories:
            # Search in category shards
            category_names = [c.category_name for c in routed_categories[:3]]  # Top 3 categories
            logger.info(f"Category-preferred search in: {category_names}")
            
            # Step 5: BM25 search in categories
            bm25_results = self._search_bm25(query_normalized, categories=category_names, top_k=top_k * 4)
            
            # Step 6: Vector search in categories
            vector_results = self._search_vector(query_text, categories=category_names, top_k=top_k * 4)
            
            # If category search yields few results, fall back to global
            if len(bm25_results) < top_k:
                logger.info(f"Category search yielded only {len(bm25_results)} results, adding global search")
                global_bm25 = self._search_bm25(query_normalized, categories=None, top_k=top_k * 2)
                
                # Merge results (deduplicate by row_id)
                seen_ids = {row_id for row_id, _ in bm25_results}
                for row_id, score in global_bm25:
                    if row_id not in seen_ids:
                        bm25_results.append((row_id, score))
                        seen_ids.add(row_id)
        else:
            # Global search
            logger.info("Global search (no category preference or no categories matched)")
            
            # Step 5: BM25 global search
            bm25_results = self._search_bm25(query_normalized, categories=None, top_k=top_k * 4)
            
            # Step 6: Vector global search
            vector_results = self._search_vector(query_text, categories=None, top_k=top_k * 4)
        
        # Step 7-9: Fuse results, apply boosts, calculate coincidencia
        if not bm25_results and not vector_results:
            logger.warning("No results found from BM25 or vector search")
            return pd.DataFrame()
        
        result_df = self._fuse_results(bm25_results, vector_results, features)
        
        # Step 10: Return top-k results
        final_results = result_df.head(top_k)
        logger.info(f"Returning {len(final_results)} final results")
        logger.info(f"Top result: {final_results.iloc[0][self.text_column] if not final_results.empty else 'N/A'}")
        logger.info(f"=" * 80)
        
        return final_results


def search_one(
    lic_ref: str,
    query_text: str,
    catalog_df: pd.DataFrame,
    bm25_index: BM25Index,
    prefer_cat: bool = True,
    top_k: int = 5,
    vector_index: Optional[Any] = None,
    text_column: str = 'product_name',
    category_column: str = 'category',
    row_id_column: str = 'product_id',  # Canviat de 'id' a 'product_id' per assegurar unicitat
    debug: bool = False
) -> pd.DataFrame:
    """
    Convenience function to perform a single search without creating SearchEngine instance.
    
    Args:
        lic_ref: Licitación reference ID.
        query_text: Query text to search.
        catalog_df: Catalog DataFrame.
        bm25_index: Built BM25 index.
        prefer_cat: If True, prefer category-specific search.
        top_k: Number of top results to return.
        vector_index: Optional vector index.
        text_column: Name of the text column.
        category_column: Name of the category column.
        row_id_column: Name of the row ID column.
        debug: If True, enable debug logging.
    
    Returns:
        DataFrame with search results, including 'coincidencia_pct' and other relevant columns.
    """
    try:
        engine = SearchEngine(
            catalog_df=catalog_df,
            bm25_index=bm25_index,
            vector_index=vector_index,
            text_column=text_column,
            category_column=category_column,
            row_id_column=row_id_column
        )
        
        # Enable debug mode if requested
        if debug:
            import logging
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        
        results = engine.search_one(
            lic_ref=lic_ref, 
            query_text=query_text, 
            prefer_cat=prefer_cat, 
            top_k=top_k,
            debug=debug
        )
        
        # Log final results summary
        if results is not None and not results.empty:
            logger.info(f"Search completed with {len(results)} results")
            if 'coincidencia_pct' in results.columns:
                logger.info(f"Top match score: {results.iloc[0]['coincidencia_pct']:.1f}%")
        else:
            logger.warning("Search returned no results")
            
        return results if results is not None else pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error in search_one: {str(e)}", exc_info=True)
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[text_column, 'coincidencia_pct', 'score'])


def configure_logging(level: int = logging.INFO, format_string: Optional[str] = None):
    """
    Configure logging for the search module.
{{ ... }}
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        format_string: Optional custom format string.
    
    Examples:
        >>> configure_logging(level=logging.DEBUG)
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.setLevel(level)
