"""
Taxonomy module for loading and managing product categories.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .config import get_config, ConfigError


@dataclass
class Category:
    """
    Represents a product category with associated keywords.
    """
    name: str
    keywords: List[str]
    
    def __post_init__(self):
        """Normalize keywords to lowercase for matching."""
        self.keywords = [kw.lower() for kw in self.keywords]
    
    def __repr__(self) -> str:
        return f"Category(name='{self.name}', keywords={len(self.keywords)} items)"


class Taxonomy:
    """
    Taxonomy manager for loading and accessing product categories.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the taxonomy manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default location.
        
        Raises:
            ConfigError: If taxonomy section is missing or invalid.
        """
        self.config = get_config(config_path)
        self.categories: List[Category] = []
        self._load_taxonomy()
    
    def _load_taxonomy(self) -> None:
        """
        Load taxonomy from configuration.
        
        Raises:
            ConfigError: If taxonomy section is missing or malformed.
        """
        config_data = self.config.config_data
        
        if 'taxonomy' not in config_data:
            raise ConfigError(
                "Missing 'taxonomy' section in configuration file.\n"
                "Please define the taxonomy with categories and keywords."
            )
        
        taxonomy_data = config_data['taxonomy']
        
        if 'categories' not in taxonomy_data:
            raise ConfigError(
                "Missing 'categories' in taxonomy section.\n"
                "Please define at least one category with keywords."
            )
        
        categories_data = taxonomy_data['categories']
        
        if not isinstance(categories_data, list):
            raise ConfigError(
                "Taxonomy 'categories' must be a list.\n"
                "Please check the configuration file format."
            )
        
        for idx, cat_data in enumerate(categories_data):
            if not isinstance(cat_data, dict):
                raise ConfigError(
                    f"Category at index {idx} must be a dictionary.\n"
                    f"Please check the configuration file format."
                )
            
            if 'name' not in cat_data:
                raise ConfigError(
                    f"Category at index {idx} is missing 'name' field.\n"
                    f"Please provide a name for each category."
                )
            
            if 'keywords' not in cat_data:
                raise ConfigError(
                    f"Category '{cat_data.get('name', 'unknown')}' is missing 'keywords' field.\n"
                    f"Please provide keywords for each category."
                )
            
            keywords = cat_data['keywords']
            if not isinstance(keywords, list):
                raise ConfigError(
                    f"Keywords for category '{cat_data['name']}' must be a list.\n"
                    f"Please check the configuration file format."
                )
            
            if not keywords:
                raise ConfigError(
                    f"Category '{cat_data['name']}' has no keywords.\n"
                    f"Please provide at least one keyword for each category."
                )
            
            category = Category(
                name=cat_data['name'],
                keywords=keywords
            )
            self.categories.append(category)
    
    def get_categories(self) -> List[Category]:
        """
        Get all categories.
        
        Returns:
            List of Category objects.
        """
        return self.categories
    
    def get_category_by_name(self, name: str) -> Optional[Category]:
        """
        Get a category by name.
        
        Args:
            name: Category name to search for.
        
        Returns:
            Category object if found, None otherwise.
        """
        for category in self.categories:
            if category.name.lower() == name.lower():
                return category
        return None
    
    def get_category_names(self) -> List[str]:
        """
        Get list of all category names.
        
        Returns:
            List of category names.
        """
        return [cat.name for cat in self.categories]
    
    def get_all_keywords(self) -> Dict[str, List[str]]:
        """
        Get all keywords organized by category.
        
        Returns:
            Dictionary mapping category names to their keywords.
        """
        return {cat.name: cat.keywords for cat in self.categories}
    
    def search_categories_by_keyword(self, keyword: str) -> List[Category]:
        """
        Find categories that contain a specific keyword.
        
        Args:
            keyword: Keyword to search for (case-insensitive).
        
        Returns:
            List of categories containing the keyword.
        """
        keyword_lower = keyword.lower()
        matching_categories = []
        
        for category in self.categories:
            if keyword_lower in category.keywords:
                matching_categories.append(category)
        
        return matching_categories
    
    def __len__(self) -> int:
        """Return number of categories."""
        return len(self.categories)
    
    def __repr__(self) -> str:
        return f"Taxonomy(categories={len(self.categories)})"


# Global taxonomy instance
_taxonomy_instance: Optional[Taxonomy] = None


def get_taxonomy(config_path: Optional[str] = None) -> Taxonomy:
    """
    Get or create the global taxonomy instance.
    
    Args:
        config_path: Path to configuration file (only used on first call).
    
    Returns:
        Taxonomy instance.
    """
    global _taxonomy_instance
    if _taxonomy_instance is None:
        _taxonomy_instance = Taxonomy(config_path)
    return _taxonomy_instance


def reload_taxonomy(config_path: Optional[str] = None) -> Taxonomy:
    """
    Reload the taxonomy from configuration file.
    
    Args:
        config_path: Path to configuration file.
    
    Returns:
        New Taxonomy instance.
    """
    global _taxonomy_instance
    _taxonomy_instance = Taxonomy(config_path)
    return _taxonomy_instance
