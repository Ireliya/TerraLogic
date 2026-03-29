"""
Configuration loader for spatial reasoning tools.
Provides centralized configuration management with environment variable overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import logging

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ToolConfig:
    """Data class representing a tool configuration."""
    tool_id: str
    name: str
    description: str
    category: str
    keywords: List[str]
    use_cases: List[str]
    parameters: Dict[str, List[str]]
    parameter_schema: Dict[str, Any]
    default_prompts: Dict[str, Any]


class ConfigurationLoader:
    """
    Centralized configuration loader with environment variable overrides and validation.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to YAML configuration file. If None, uses default location.
        """
        self.config_path = config_path or self._get_default_config_path()
        self._config_cache: Optional[Dict[str, Any]] = None
        self._load_config()

    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        # Look for config file relative to this module
        current_dir = Path(__file__).parent
        config_file = current_dir / "tools_config.yaml"

        if config_file.exists():
            return str(config_file)

        # Fallback to environment variable or default
        return os.getenv('SPATIAL_TOOLS_CONFIG', 'spatialreason/config/tools_config.yaml')

    def _load_config(self) -> None:
        """Load configuration from YAML file with error handling."""
        try:
            config_path = Path(self.config_path)
            if not config_path.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                self._config_cache = self._get_fallback_config()
                return

            with open(config_path, 'r', encoding='utf-8') as f:
                self._config_cache = yaml.safe_load(f)

            # Apply environment variable overrides
            self._apply_environment_overrides()

            # Validate configuration
            self._validate_config()

            logger.info(f"Configuration loaded successfully from {config_path}")

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {self.config_path}: {e}")
            self._config_cache = self._get_fallback_config()
        except Exception as e:
            logger.error(f"Error loading configuration from {self.config_path}: {e}")
            self._config_cache = self._get_fallback_config()

    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        if not self._config_cache or 'environment_overrides' not in self._config_cache:
            return

        overrides = self._config_cache['environment_overrides']
        for env_var, config_path in overrides.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Convert string values to appropriate types
                    if env_value.lower() in ('true', 'false'):
                        env_value = env_value.lower() == 'true'
                    elif env_value.replace('.', '').replace('-', '').isdigit():
                        env_value = float(env_value) if '.' in env_value else int(env_value)

                    # Set the value in configuration using dot notation
                    self._set_nested_value(self._config_cache, config_path, env_value)
                    logger.info(f"Applied environment override: {env_var} = {env_value}")

                except Exception as e:
                    logger.warning(f"Failed to apply environment override {env_var}: {e}")

    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested configuration value using dot notation."""
        keys = path.split('.')
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _validate_config(self) -> None:
        """Validate configuration structure and required fields."""
        if not self._config_cache:
            raise ValueError("Configuration is empty")

        # Check required top-level sections
        required_sections = ['defaults', 'tools']
        for section in required_sections:
            if section not in self._config_cache:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate tools section
        tools = self._config_cache.get('tools', {})
        if not tools:
            raise ValueError("No tools defined in configuration")

        # Validate each tool configuration
        for tool_id, tool_config in tools.items():
            self._validate_tool_config(tool_id, tool_config)

    def _validate_tool_config(self, tool_id: str, tool_config: Dict[str, Any]) -> None:
        """Validate individual tool configuration."""
        required_fields = ['name', 'description', 'category', 'keywords', 'use_cases', 'parameters']

        for field in required_fields:
            if field not in tool_config:
                raise ValueError(f"Tool {tool_id} missing required field: {field}")

        # Validate parameters structure
        params = tool_config.get('parameters', {})
        if 'required' not in params or 'optional' not in params:
            raise ValueError(f"Tool {tool_id} parameters must have 'required' and 'optional' lists")

    def _get_fallback_config(self) -> Dict[str, Any]:
        """Get minimal fallback configuration when loading fails."""
        return {
            'defaults': {
                'spatial': {'buffer_distance_meters': 30.0},
                'confidence_thresholds': {'detection': 0.6, 'classification': 0.65, 'segmentation': 0.5},
                'visualization': {'alpha': 0.6, 'dpi': 150, 'font_size': 10},
                'matching': {'query_threshold': 0.3, 'primary_keyword_boost': 0.5, 'keyword_weight': 0.3, 'use_case_weight': 0.7}
            },
            'tools': {
                'segmentation': {
                    'tool_id': 'segmentation', 'name': 'RemoteSAM Segmentation Tool',
                    'description': 'Segment objects in satellite imagery using text prompts',
                    'category': 'perception',
                    'keywords': ['segment', 'segmentation', 'water', 'building', 'objects'],
                    'use_cases': ['segment buildings in urban areas', 'segment water bodies and rivers'],
                    'parameters': {'required': ['image_path', 'text_prompt'], 'optional': ['meters_per_pixel']},
                    'parameter_schema': {
                        'image_path': {'type': 'string', 'description': 'Path to image file'},
                        'text_prompt': {'type': 'string', 'description': 'Natural language description'},
                        'meters_per_pixel': {'type': 'number', 'description': 'Ground resolution', 'default': 0.3}
                    },
                    'default_prompts': {'fallback': 'buildings', 'object_mappings': {'building': 'buildings', 'water': 'water bodies'}}
                }
            }
        }

    def get_config(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            path: Configuration path using dot notation (e.g., 'defaults.spatial.meters_per_pixel')
            default: Default value if path not found

        Returns:
            Configuration value or default
        """
        if not self._config_cache:
            return default

        keys = path.split('.')
        current = self._config_cache

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def get_tool_config(self, tool_id: str) -> Optional[ToolConfig]:
        """
        Get tool configuration as a structured object.

        Args:
            tool_id: Tool identifier

        Returns:
            ToolConfig object or None if not found
        """
        tool_data = self.get_config(f'tools.{tool_id}')
        if not tool_data:
            return None

        try:
            return ToolConfig(
                tool_id=tool_data.get('tool_id', tool_id),
                name=tool_data.get('name', ''),
                description=tool_data.get('description', ''),
                category=tool_data.get('category', ''),
                keywords=tool_data.get('keywords', []),
                use_cases=tool_data.get('use_cases', []),
                parameters=tool_data.get('parameters', {}),
                parameter_schema=tool_data.get('parameter_schema', {}),
                default_prompts=tool_data.get('default_prompts', {})
            )
        except Exception as e:
            logger.error(f"Error creating ToolConfig for {tool_id}: {e}")
            return None

    def get_available_tools(self) -> List[str]:
        """Get list of available tool IDs."""
        tools = self.get_config('tools', {})
        return list(tools.keys())

    def get_default_confidence_threshold(self, category: str) -> float:
        """Get default confidence threshold for a tool category."""
        return self.get_config(f'defaults.confidence_thresholds.{category}', 0.5)

    def get_default_spatial_params(self) -> Dict[str, float]:
        """Get default spatial parameters."""
        return {
            'buffer_distance_meters': self.get_config('defaults.spatial.buffer_distance_meters', 30.0)
        }

    def get_matching_params(self) -> Dict[str, float]:
        """Get query matching parameters."""
        return {
            'query_threshold': self.get_config('defaults.matching.query_threshold', 0.3),
            'primary_keyword_boost': self.get_config('defaults.matching.primary_keyword_boost', 0.5),
            'keyword_weight': self.get_config('defaults.matching.keyword_weight', 0.3),
            'use_case_weight': self.get_config('defaults.matching.use_case_weight', 0.7)
        }

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation mode configuration."""
        return {
            'fail_on_no_plan': self.get_config('defaults.evaluation.fail_on_no_plan', True),
            'fail_on_no_steps': self.get_config('defaults.evaluation.fail_on_no_steps', True),
            'strict_json_validation': self.get_config('defaults.evaluation.strict_json_validation', True),
            'log_raw_planner_output': self.get_config('defaults.evaluation.log_raw_planner_output', True),
            'terminate_on_planner_failure': self.get_config('defaults.evaluation.terminate_on_planner_failure', True)
        }

    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._config_cache = None
        self._load_config()


class ToolRegistry:
    """
    Dynamic tool registry that creates tool interfaces from configuration.
    """

    def __init__(self, config_loader: Optional[ConfigurationLoader] = None):
        """
        Initialize tool registry.

        Args:
            config_loader: Configuration loader instance. If None, creates a new one.
        """
        self.config_loader = config_loader or ConfigurationLoader()
        self._interface_cache: Dict[str, Any] = {}

    def get_tool_interface(self, tool_id: str):
        """
        Get tool interface instance, creating it if necessary.

        Args:
            tool_id: Tool identifier

        Returns:
            Tool interface instance

        Raises:
            ValueError: If tool not found in configuration
        """
        if tool_id not in self._interface_cache:
            self._interface_cache[tool_id] = self._create_tool_interface(tool_id)

        return self._interface_cache[tool_id]

    def _create_tool_interface(self, tool_id: str):
        """Create tool interface from configuration."""
        tool_config = self.config_loader.get_tool_config(tool_id)
        if not tool_config:
            raise ValueError(f"Tool configuration not found for {tool_id}")

        # Create tool interface directly to avoid recursion
        if tool_id in ["segmentation", "detection", "classification"]:
            from spatialreason.tools.tool_interface import RemoteSAMToolInterface
            return RemoteSAMToolInterface(tool_id)
        else:
            # For other tools, create a basic interface
            from spatialreason.tools.tool_interface import ToolInterface
            return ToolInterface(tool_id)

    def get_available_tools(self) -> List[str]:
        """Get list of available tool IDs."""
        return self.config_loader.get_available_tools()

    def clear_cache(self) -> None:
        """Clear the interface cache."""
        self._interface_cache.clear()

    def reload_config(self) -> None:
        """Reload configuration and clear cache."""
        self.config_loader.reload_config()
        self.clear_cache()


# Global instances for easy access
_config_loader: Optional[ConfigurationLoader] = None
_tool_registry: Optional[ToolRegistry] = None


def get_config_loader() -> ConfigurationLoader:
    """Get global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigurationLoader()
    return _config_loader


def get_tool_registry() -> ToolRegistry:
    """Get global tool registry instance."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry(get_config_loader())
    return _tool_registry