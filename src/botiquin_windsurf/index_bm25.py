"""
BM25 indexing module for efficient text-based search.
"""

from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
from rank_bm25 import BM25Okapi
from dataclasses import dataclass


@dataclass
class SearchResult:
    """
    Represents a search result with row ID and score.
    """
    row_id: int
    score: float
    
    def __repr__(self) -> str:
        return f"SearchResult(row_id={self.row_id}, score={self.score:.4f})"


class BM25Index:
    """
    BM25 index for efficient text-based search with support for global and per-category indexing.
    """
    
    def __init__(self):
        """Initialize the BM25 index."""
        self.global_index: Optional[BM25Okapi] = None
        self.global_corpus: List[List[str]] = []
        self.global_row_ids: List[int] = []
        
        self.category_indexes: Dict[str, BM25Okapi] = {}
        self.category_corpus: Dict[str, List[List[str]]] = {}
        self.category_row_ids: Dict[str, List[int]] = {}
        
        self.is_built = False
    
    def build_global_index(
        self,
        df: pd.DataFrame,
        text_column: str,
        row_id_column: Optional[str] = None
    ) -> None:
        """
        Build a global BM25 index from a DataFrame.
        
        Args:
            df: DataFrame containing the documents.
            text_column: Name of the column containing text to index.
            row_id_column: Name of the column containing row IDs. 
                          If None, uses DataFrame index.
        
        Raises:
            ValueError: If text_column doesn't exist in DataFrame.
        """
        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found in DataFrame.\n"
                f"Available columns: {', '.join(df.columns)}"
            )
        
        # Get row IDs
        if row_id_column is not None:
            if row_id_column not in df.columns:
                raise ValueError(
                    f"Column '{row_id_column}' not found in DataFrame.\n"
                    f"Available columns: {', '.join(df.columns)}"
                )
            self.global_row_ids = df[row_id_column].tolist()
        else:
            self.global_row_ids = df.index.tolist()
        
        # Tokenize documents (split by whitespace)
        self.global_corpus = []
        for text in df[text_column]:
            if pd.isna(text):
                tokens = []
            else:
                tokens = str(text).lower().split()
            self.global_corpus.append(tokens)
        
        # Build BM25 index
        if self.global_corpus:
            self.global_index = BM25Okapi(self.global_corpus)
        
        self.is_built = True
    
    def build_category_indexes(
        self,
        df: pd.DataFrame,
        text_column: str,
        category_column: str,
        row_id_column: Optional[str] = None
    ) -> None:
        """
        Build per-category BM25 indexes from a DataFrame.
        
        Args:
            df: DataFrame containing the documents.
            text_column: Name of the column containing text to index.
            category_column: Name of the column containing categories.
            row_id_column: Name of the column containing row IDs.
                          If None, uses DataFrame index.
        
        Raises:
            ValueError: If required columns don't exist in DataFrame.
        """
        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found in DataFrame.\n"
                f"Available columns: {', '.join(df.columns)}"
            )
        
        if category_column not in df.columns:
            raise ValueError(
                f"Column '{category_column}' not found in DataFrame.\n"
                f"Available columns: {', '.join(df.columns)}"
            )
        
        # Group by category
        for category, group_df in df.groupby(category_column):
            if pd.isna(category):
                category = "uncategorized"
            
            category_str = str(category)
            
            # Get row IDs for this category
            if row_id_column is not None:
                if row_id_column not in df.columns:
                    raise ValueError(
                        f"Column '{row_id_column}' not found in DataFrame.\n"
                        f"Available columns: {', '.join(df.columns)}"
                    )
                row_ids = group_df[row_id_column].tolist()
            else:
                row_ids = group_df.index.tolist()
            
            # Tokenize documents for this category
            corpus = []
            for text in group_df[text_column]:
                if pd.isna(text):
                    tokens = []
                else:
                    tokens = str(text).lower().split()
                corpus.append(tokens)
            
            # Build BM25 index for this category
            if corpus:
                self.category_indexes[category_str] = BM25Okapi(corpus)
                self.category_corpus[category_str] = corpus
                self.category_row_ids[category_str] = row_ids
    
    def build_indexes(
        self,
        df: pd.DataFrame,
        text_column: str,
        category_column: Optional[str] = None,
        row_id_column: Optional[str] = None
    ) -> None:
        """
        Build both global and category indexes in one call.
        
        Args:
            df: DataFrame containing the documents.
            text_column: Name of the column containing text to index.
            category_column: Optional name of the column containing categories.
                           If provided, builds per-category indexes.
            row_id_column: Name of the column containing row IDs.
                          If None, uses DataFrame index.
        """
        # Build global index
        self.build_global_index(df, text_column, row_id_column)
        
        # Build category indexes if category column is provided
        if category_column is not None:
            self.build_category_indexes(df, text_column, category_column, row_id_column)
    
    def search_global(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search the global index.
        
        Args:
            query: Search query text.
            top_k: Number of top results to return.
        
        Returns:
            List of SearchResult objects sorted by score (descending).
        
        Raises:
            RuntimeError: If global index has not been built.
        """
        if self.global_index is None:
            raise RuntimeError(
                "Global index has not been built. "
                "Call build_global_index() or build_indexes() first."
            )
        
        if not query or not isinstance(query, str):
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.global_index.get_scores(query_tokens)
        
        # Get top-k results, including some with lower scores
        top_indices = scores.argsort()[-top_k*2:][::-1]  # Get more results for filtering
        
        results = []
        for idx in top_indices:
            # Include results with score > 0 or within 50% of the top score
            if len(results) == 0 or (scores[idx] > 0 and (scores[idx] / scores[top_indices[0]]) > 0.3):
                results.append(SearchResult(
                    row_id=self.global_row_ids[idx],
                    score=float(max(scores[idx], 0.01))  # Ensure minimum score of 0.01
                ))
                
            # Stop if we have enough results
            if len(results) >= top_k:
                break
        
        return results
    
    def search_category(
        self,
        query: str,
        category: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search within a specific category.
        
        Args:
            query: Search query text.
            category: Category name to search within.
            top_k: Number of top results to return.
        
        Returns:
            List of SearchResult objects sorted by score (descending).
        
        Raises:
            ValueError: If category doesn't exist in the index.
        """
        if category not in self.category_indexes:
            available_categories = list(self.category_indexes.keys())
            raise ValueError(
                f"Category '{category}' not found in index.\n"
                f"Available categories: {', '.join(available_categories)}"
            )
        
        if not query or not isinstance(query, str):
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        if not query_tokens:
            return []
        
        # Get BM25 scores for this category
        category_index = self.category_indexes[category]
        scores = category_index.get_scores(query_tokens)
        
        # Get top-k results, including some with lower scores
        top_indices = scores.argsort()[-top_k*2:][::-1]  # Get more results for filtering
        
        results = []
        for idx in top_indices:
            # Include results with score > 0 or within 50% of the top score
            if len(results) == 0 or (scores[idx] > 0 and (scores[idx] / scores[top_indices[0]]) > 0.3):
                results.append(SearchResult(
                    row_id=self.category_row_ids[category][idx],
                    score=float(max(scores[idx], 0.01))  # Ensure minimum score of 0.01
                ))
                
            # Stop if we have enough results
            if len(results) >= top_k:
                break
        
        return results
    
    def search_multiple_categories(
        self,
        query: str,
        categories: List[str],
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search across multiple categories and merge results.
        
        Args:
            query: Search query text.
            categories: List of category names to search within.
            top_k: Number of top results to return (total across all categories).
        
        Returns:
            List of SearchResult objects sorted by score (descending).
        """
        all_results = []
        
        for category in categories:
            if category in self.category_indexes:
                results = self.search_category(query, category, top_k=top_k)
                all_results.extend(results)
        
        # Sort by score and return top-k
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
    
    def get_categories(self) -> List[str]:
        """
        Get list of available categories in the index.
        
        Returns:
            List of category names.
        """
        return list(self.category_indexes.keys())
    
    def get_global_size(self) -> int:
        """
        Get the number of documents in the global index.
        
        Returns:
            Number of documents.
        """
        return len(self.global_corpus)
    
    def get_category_size(self, category: str) -> int:
        """
        Get the number of documents in a category index.
        
        Args:
            category: Category name.
        
        Returns:
            Number of documents in the category.
        
        Raises:
            ValueError: If category doesn't exist.
        """
        if category not in self.category_corpus:
            raise ValueError(f"Category '{category}' not found in index.")
        return len(self.category_corpus[category])
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics.
        """
        stats = {
            "is_built": self.is_built,
            "global_documents": self.get_global_size(),
            "num_categories": len(self.category_indexes),
            "categories": {}
        }
        
        for category in self.category_indexes.keys():
            stats["categories"][category] = self.get_category_size(category)
        
        return stats
    
    def __repr__(self) -> str:
        return (
            f"BM25Index(global_docs={self.get_global_size()}, "
            f"categories={len(self.category_indexes)})"
        )
