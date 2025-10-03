"""
Router module for routing queries to appropriate product categories.
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from .taxonomy import get_taxonomy, Taxonomy, Category


@dataclass
class CategoryMatch:
    """
    Represents a category match with its score.
    """
    category_name: str
    score: int
    matched_keywords: List[str]
    
    def __repr__(self) -> str:
        return f"CategoryMatch(category='{self.category_name}', score={self.score}, keywords={self.matched_keywords})"


class Router:
    """
    Router for matching queries to product categories based on keyword matching.
    """
    
    def __init__(self, taxonomy: Optional[Taxonomy] = None):
        """
        Initialize the router.
        
        Args:
            taxonomy: Taxonomy instance. If None, uses global taxonomy.
        """
        self.taxonomy = taxonomy if taxonomy is not None else get_taxonomy()
    
    def route_categories(
        self,
        query_norm: str,
        min_score: int = 1,
        max_results: Optional[int] = None
    ) -> List[CategoryMatch]:
        """
        Route a normalized query to candidate categories.
        
        Args:
            query_norm: Normalized query text (should be lowercase, without accents).
            min_score: Minimum score threshold for results (default: 1).
            max_results: Maximum number of results to return. If None, returns all matches.
        
        Returns:
            List of CategoryMatch objects sorted by score (descending).
            Returns empty list if no categories match.
        
        Examples:
            >>> router = Router()
            >>> matches = router.route_categories("aguja hipodermica 25g")
            >>> matches[0].category_name
            'Agujas y Jeringas'
            >>> matches[0].score
            2
        """
        if not query_norm or not isinstance(query_norm, str):
            return []
        
        query_lower = query_norm.lower()
        category_matches: List[CategoryMatch] = []
        
        # Check each category for keyword matches
        for category in self.taxonomy.get_categories():
            matched_keywords = []
            score = 0
            
            # Count how many keywords from this category appear in the query
            for keyword in category.keywords:
                # Use word boundary matching to avoid partial matches
                # For multi-word keywords, check if the entire phrase is present
                if ' ' in keyword:
                    # Multi-word keyword - check for exact phrase
                    if keyword in query_lower:
                        matched_keywords.append(keyword)
                        score += 1
                else:
                    # Single-word keyword - check with word boundaries
                    # Simple approach: check if keyword appears as a separate word
                    words = query_lower.split()
                    if keyword in words:
                        matched_keywords.append(keyword)
                        score += 1
                    # Also check if keyword is part of a compound word
                    elif any(keyword in word for word in words):
                        matched_keywords.append(keyword)
                        score += 1
            
            # Add category if it has matches above threshold
            if score >= min_score:
                category_matches.append(CategoryMatch(
                    category_name=category.name,
                    score=score,
                    matched_keywords=matched_keywords
                ))
        
        # Sort by score (descending), then by category name (ascending)
        category_matches.sort(key=lambda x: (-x.score, x.category_name))
        
        # Limit results if max_results is specified
        if max_results is not None and max_results > 0:
            category_matches = category_matches[:max_results]
        
        return category_matches
    
    def get_best_category(self, query_norm: str) -> Optional[CategoryMatch]:
        """
        Get the best matching category for a query.
        
        Args:
            query_norm: Normalized query text.
        
        Returns:
            Best CategoryMatch or None if no matches found.
        """
        matches = self.route_categories(query_norm, max_results=1)
        return matches[0] if matches else None
    
    def get_category_scores(self, query_norm: str) -> Dict[str, int]:
        """
        Get scores for all categories as a dictionary.
        
        Args:
            query_norm: Normalized query text.
        
        Returns:
            Dictionary mapping category names to scores.
        """
        matches = self.route_categories(query_norm, min_score=0)
        return {match.category_name: match.score for match in matches}


def route_categories(
    query_norm: str,
    taxonomy: Optional[Taxonomy] = None,
    min_score: int = 1,
    max_results: Optional[int] = None
) -> List[CategoryMatch]:
    """
    Convenience function to route categories without creating a Router instance.
    
    Args:
        query_norm: Normalized query text (should be lowercase, without accents).
        taxonomy: Optional Taxonomy instance. If None, uses global taxonomy.
        min_score: Minimum score threshold for results (default: 1).
        max_results: Maximum number of results to return.
    
    Returns:
        List of CategoryMatch objects sorted by score (descending).
        Returns empty list if no categories match.
    
    Examples:
        >>> from botiquin_windsurf.router import route_categories
        >>> matches = route_categories("aguja hipodermica 25g")
        >>> if matches:
        ...     print(f"Best match: {matches[0].category_name} (score: {matches[0].score})")
    """
    router = Router(taxonomy)
    return router.route_categories(query_norm, min_score, max_results)


def get_best_category(
    query_norm: str,
    taxonomy: Optional[Taxonomy] = None
) -> Optional[CategoryMatch]:
    """
    Convenience function to get the best matching category.
    
    Args:
        query_norm: Normalized query text.
        taxonomy: Optional Taxonomy instance.
    
    Returns:
        Best CategoryMatch or None if no matches found.
    """
    router = Router(taxonomy)
    return router.get_best_category(query_norm)
