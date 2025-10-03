"""
Fusion module for combining BM25 and vector search results using Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class FusedResult:
    """
    Represents a fused search result with combined score.
    """
    row_id: int
    rrf_score: float
    bm25_score: float
    vector_score: float
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    boost_score: float = 0.0
    final_score: float = 0.0
    coincidencia_pct: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"FusedResult(row_id={self.row_id}, "
            f"final_score={self.final_score:.4f}, "
            f"coincidencia={self.coincidencia_pct:.1f}%)"
        )


def reciprocal_rank_fusion(
    bm25_results: List[Tuple[int, float]],
    vector_results: List[Tuple[int, float]],
    k: int = 60
) -> List[FusedResult]:
    """
    Combine BM25 and vector search results using Reciprocal Rank Fusion (RRF).
    
    RRF formula: score(d) = sum(1 / (k + rank(d)))
    
    Args:
        bm25_results: List of (row_id, score) tuples from BM25 search.
        vector_results: List of (row_id, score) tuples from vector search.
        k: RRF constant (default: 60, commonly used value).
    
    Returns:
        List of FusedResult objects sorted by RRF score (descending).
    
    Examples:
        >>> bm25_results = [(1, 10.5), (2, 8.3), (3, 5.1)]
        >>> vector_results = [(2, 0.95), (1, 0.87), (4, 0.75)]
        >>> fused = reciprocal_rank_fusion(bm25_results, vector_results)
        >>> fused[0].row_id  # Best combined result
    """
    # Create rank mappings
    bm25_ranks = {row_id: rank + 1 for rank, (row_id, _) in enumerate(bm25_results)}
    vector_ranks = {row_id: rank + 1 for rank, (row_id, _) in enumerate(vector_results)}
    
    # Create score mappings
    bm25_scores = {row_id: score for row_id, score in bm25_results}
    vector_scores = {row_id: score for row_id, score in vector_results}
    
    # Get all unique row IDs
    all_row_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())
    
    # Calculate RRF scores
    fused_results = []
    for row_id in all_row_ids:
        rrf_score = 0.0
        
        # Add BM25 contribution
        if row_id in bm25_ranks:
            rrf_score += 1.0 / (k + bm25_ranks[row_id])
        
        # Add vector contribution
        if row_id in vector_ranks:
            rrf_score += 1.0 / (k + vector_ranks[row_id])
        
        fused_results.append(FusedResult(
            row_id=row_id,
            rrf_score=rrf_score,
            bm25_score=bm25_scores.get(row_id, 0.0),
            vector_score=vector_scores.get(row_id, 0.0),
            bm25_rank=bm25_ranks.get(row_id),
            vector_rank=vector_ranks.get(row_id),
            final_score=rrf_score
        ))
    
    # Sort by RRF score (descending)
    fused_results.sort(key=lambda x: x.rrf_score, reverse=True)
    
    return fused_results


def apply_boosts(
    row: pd.Series,
    features: Dict[str, Any],
    boost_config: Optional[Dict[str, float]] = None
) -> float:
    """
    Apply feature-based boosts to a search result.
    
    Boosts are applied for matching:
    - Manufacturer (fabricante)
    - Manufacturer reference (ref_fabricante)
    - Size/dimensions (mida/tamaño)
    
    Args:
        row: DataFrame row containing product information.
        features: Dictionary with query features to match against.
                 Expected keys: 'fabricante', 'ref_fabricante', 'mida'
        boost_config: Optional dictionary with boost values.
                     Default: {'fabricante': 10.0, 'ref_fabricante': 15.0, 'mida': 5.0}
    
    Returns:
        Total boost score (sum of all matching boosts).
    
    Examples:
        >>> row = pd.Series({'fabricante': 'BD', 'ref_fabricante': '305122', 'mida': '25G'})
        >>> features = {'fabricante': 'BD', 'ref_fabricante': '305122', 'mida': '25G'}
        >>> boost = apply_boosts(row, features)
        >>> boost
        30.0  # 10 + 15 + 5
    """
    if boost_config is None:
        boost_config = {
            'fabricante': 10.0,
            'ref_fabricante': 15.0,
            'mida': 5.0
        }
    
    boost_score = 0.0
    
    # Boost for matching manufacturer
    if 'fabricante' in features and features['fabricante']:
        row_fabricante = row.get('fabricante', '') or row.get('manufacturer', '')
        if row_fabricante and str(row_fabricante).lower().strip() == str(features['fabricante']).lower().strip():
            boost_score += boost_config.get('fabricante', 10.0)
    
    # Boost for matching manufacturer reference
    if 'ref_fabricante' in features and features['ref_fabricante']:
        row_ref = row.get('ref_fabricante', '') or row.get('manufacturer_ref', '') or row.get('reference', '')
        if row_ref and str(row_ref).lower().strip() == str(features['ref_fabricante']).lower().strip():
            boost_score += boost_config.get('ref_fabricante', 15.0)
    
    # Boost for matching size
    if 'mida' in features and features['mida']:
        row_mida = row.get('mida', '') or row.get('size', '') or row.get('tamaño', '')
        query_mida = str(features['mida']).lower().strip()
        row_mida_str = str(row_mida).lower().strip()
        
        # Exact match
        if row_mida_str == query_mida:
            boost_score += boost_config.get('mida', 5.0)
        # Partial match (size contained in product description)
        elif query_mida and query_mida in row_mida_str:
            boost_score += boost_config.get('mida', 5.0) * 0.5
    
    return boost_score


def calculate_coincidencia(
    final_score: float,
    max_score: float,
    min_score: float = 0.0
) -> float:
    """
    Calculate coincidencia percentage (match percentage) capped at 100%.
    
    Args:
        final_score: The final score of the result.
        max_score: Maximum score in the result set.
        min_score: Minimum score in the result set (default: 0.0).
    
    Returns:
        Coincidencia percentage (0-100).
    
    Examples:
        >>> calculate_coincidencia(0.8, 1.0)
        80.0
        >>> calculate_coincidencia(1.2, 1.0)
        100.0
    """
    if max_score == min_score:
        return 100.0 if final_score > 0 else 0.0
    
    # Normalize to 0-100 range
    normalized = ((final_score - min_score) / (max_score - min_score)) * 100.0
    
    # Cap at 100%
    return min(normalized, 100.0)


def fuse_and_boost(
    bm25_results: List[Tuple[int, float]],
    vector_results: List[Tuple[int, float]],
    df: pd.DataFrame,
    features: Optional[Dict[str, Any]] = None,
    k: int = 60,
    boost_config: Optional[Dict[str, float]] = None,
    row_id_column: str = 'id'
) -> List[FusedResult]:
    """
    Complete fusion pipeline: RRF + feature boosts + coincidencia calculation.
    
    Args:
        bm25_results: List of (row_id, score) tuples from BM25 search.
        vector_results: List of (row_id, score) tuples from vector search.
        df: DataFrame containing product information.
        features: Optional dictionary with query features for boosting.
        k: RRF constant (default: 60).
        boost_config: Optional boost configuration.
        row_id_column: Name of the row ID column in DataFrame.
    
    Returns:
        List of FusedResult objects with final scores and coincidencia percentages.
    
    Examples:
        >>> bm25_results = [(1, 10.5), (2, 8.3)]
        >>> vector_results = [(2, 0.95), (1, 0.87)]
        >>> results = fuse_and_boost(bm25_results, vector_results, df, features={'fabricante': 'BD'})
    """
    # Step 1: Apply RRF
    fused_results = reciprocal_rank_fusion(bm25_results, vector_results, k=k)
    
    if not fused_results:
        return []
    
    # Step 2: Apply boosts if features are provided
    if features and not df.empty:
        # Create a mapping from row_id to DataFrame index
        if row_id_column in df.columns:
            row_id_to_idx = df.set_index(row_id_column).to_dict('index')
        else:
            row_id_to_idx = {idx: df.loc[idx].to_dict() for idx in df.index}
        
        for result in fused_results:
            if result.row_id in row_id_to_idx:
                row_data = pd.Series(row_id_to_idx[result.row_id])
                result.boost_score = apply_boosts(row_data, features, boost_config)
                result.final_score = result.rrf_score + result.boost_score
            else:
                result.final_score = result.rrf_score
    else:
        # No boosts, final score = RRF score
        for result in fused_results:
            result.final_score = result.rrf_score
    
    # Re-sort by final score
    fused_results.sort(key=lambda x: x.final_score, reverse=True)
    
    # Step 3: Calculate coincidencia percentages
    if fused_results:
        max_score = max(r.final_score for r in fused_results)
        min_score = min(r.final_score for r in fused_results)
        
        for result in fused_results:
            result.coincidencia_pct = calculate_coincidencia(
                result.final_score,
                max_score,
                min_score
            )
    
    return fused_results


def fused_results_to_dataframe(
    fused_results: List[FusedResult],
    df: pd.DataFrame,
    row_id_column: str = 'id',
    include_scores: bool = True
) -> pd.DataFrame:
    """
    Convert fused results to a DataFrame with original product data.
    
    Args:
        fused_results: List of FusedResult objects.
        df: Original DataFrame with product data.
        row_id_column: Name of the row ID column.
        include_scores: Whether to include score columns.
    
    Returns:
        DataFrame with results and scores.
    """
    if not fused_results:
        return pd.DataFrame()
    
    # Get row IDs
    row_ids = [r.row_id for r in fused_results]
    
    # Filter DataFrame
    if row_id_column in df.columns:
        result_df = df[df[row_id_column].isin(row_ids)].copy()
        # Reorder to match fused_results order
        result_df['_sort_key'] = result_df[row_id_column].map({rid: i for i, rid in enumerate(row_ids)})
        result_df = result_df.sort_values('_sort_key').drop('_sort_key', axis=1)
    else:
        result_df = df.loc[row_ids].copy()
    
    # Add score columns if requested
    if include_scores:
        score_data = {
            'final_score': [r.final_score for r in fused_results],
            'coincidencia_pct': [r.coincidencia_pct for r in fused_results],
            'rrf_score': [r.rrf_score for r in fused_results],
            'bm25_score': [r.bm25_score for r in fused_results],
            'vector_score': [r.vector_score for r in fused_results],
            'boost_score': [r.boost_score for r in fused_results]
        }
        
        for col, values in score_data.items():
            result_df[col] = values
    
    return result_df


class Fuser:
    """
    Fuser class for managing fusion operations.
    """
    
    def __init__(
        self,
        k: int = 60,
        boost_config: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the Fuser.
        
        Args:
            k: RRF constant (default: 60).
            boost_config: Optional boost configuration.
        """
        self.k = k
        self.boost_config = boost_config or {
            'fabricante': 10.0,
            'ref_fabricante': 15.0,
            'mida': 5.0
        }
    
    def fuse(
        self,
        bm25_results: List[Tuple[int, float]],
        vector_results: List[Tuple[int, float]],
        df: Optional[pd.DataFrame] = None,
        features: Optional[Dict[str, Any]] = None,
        row_id_column: str = 'id'
    ) -> List[FusedResult]:
        """
        Fuse BM25 and vector results with optional boosting.
        
        Args:
            bm25_results: BM25 search results.
            vector_results: Vector search results.
            df: Optional DataFrame for feature boosting.
            features: Optional features for boosting.
            row_id_column: Row ID column name.
        
        Returns:
            List of FusedResult objects.
        """
        if df is not None and features is not None:
            return fuse_and_boost(
                bm25_results,
                vector_results,
                df,
                features,
                k=self.k,
                boost_config=self.boost_config,
                row_id_column=row_id_column
            )
        else:
            return reciprocal_rank_fusion(bm25_results, vector_results, k=self.k)
    
    def set_boost_config(self, boost_config: Dict[str, float]) -> None:
        """Update boost configuration."""
        self.boost_config = boost_config
    
    def get_boost_config(self) -> Dict[str, float]:
        """Get current boost configuration."""
        return self.boost_config.copy()
