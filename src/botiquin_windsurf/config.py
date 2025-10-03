"""
Configuration module for loading and managing configuration settings.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


class Config:
    """
    Configuration manager for loading and accessing configuration settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default location.
        """
        if config_path is None:
            # Default to config.yaml in the project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config_data: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """
        Load configuration from YAML file.
        
        Raises:
            ConfigError: If the configuration file cannot be loaded.
        """
        if not self.config_path.exists():
            raise ConfigError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please ensure config.yaml exists in the project root."
            )
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(
                f"Error parsing configuration file: {self.config_path}\n"
                f"YAML error: {str(e)}"
            )
        except Exception as e:
            raise ConfigError(
                f"Error reading configuration file: {self.config_path}\n"
                f"Error: {str(e)}"
            )
    
    def get_catalog_column_mapping(self) -> Dict[str, str]:
        """
        Get the column mapping for catalog file.
        
        Returns:
            Dictionary mapping internal column names to Excel column names.
        
        Raises:
            ConfigError: If catalog_columns is not defined in config.
        """
        if 'catalog_columns' not in self.config_data:
            raise ConfigError(
                "Missing 'catalog_columns' section in configuration file.\n"
                "Please define the column mapping for catalog file."
            )
        return self.config_data['catalog_columns']
    
    def get_tender_column_mapping(self) -> Dict[str, str]:
        """
        Get the column mapping for tender file.
        
        Returns:
            Dictionary mapping internal column names to Excel column names.
        
        Raises:
            ConfigError: If tender_columns is not defined in config.
        """
        if 'tender_columns' not in self.config_data:
            raise ConfigError(
                "Missing 'tender_columns' section in configuration file.\n"
                "Please define the column mapping for tender file."
            )
        return self.config_data['tender_columns']
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a general setting value.
        
        Args:
            key: Setting key to retrieve.
            default: Default value if key is not found.
        
        Returns:
            Setting value or default.
        """
        settings = self.config_data.get('settings', {})
        return settings.get(key, default)
    
    def get_all_settings(self) -> Dict[str, Any]:
        """
        Get all general settings.
        
        Returns:
            Dictionary of all settings.
        """
        return self.config_data.get('settings', {})
    
    def validate_column_mapping(
        self, 
        mapping: Dict[str, str], 
        available_columns: list, 
        file_type: str
    ) -> None:
        """
        Validate that all mapped columns exist in the Excel file.
        
        Args:
            mapping: Column mapping dictionary.
            available_columns: List of available columns in the Excel file.
            file_type: Type of file (for error messages).
        
        Raises:
            ConfigError: If any mapped column is not found in the file.
        """
        missing_columns = []
        for internal_name, excel_name in mapping.items():
            if excel_name not in available_columns:
                missing_columns.append(f"  - '{excel_name}' (mapped from '{internal_name}')")
        
        if missing_columns:
            raise ConfigError(
                f"Column mapping error for {file_type}:\n"
                f"The following columns are defined in config.yaml but not found in the Excel file:\n"
                + "\n".join(missing_columns) + "\n\n"
                f"Available columns in the file:\n"
                + "\n".join(f"  - '{col}'" for col in available_columns) + "\n\n"
                f"Please update config.yaml to match the actual column names in your Excel file."
            )


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get or create the global configuration instance.
    
    Args:
        config_path: Path to configuration file (only used on first call).
    
    Returns:
        Config instance.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


def reload_config(config_path: Optional[str] = None) -> Config:
    """
    Reload the configuration from file.
    
    Args:
        config_path: Path to configuration file.
    
    Returns:
        New Config instance.
    """
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance
