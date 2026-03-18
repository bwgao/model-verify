import os
import re
import yaml
from typing import Any, Optional
from pathlib import Path
from utils.types import ProviderConfig, ModelConfig


class ConfigLoader:
    """
    Configuration loader for the Model Verification System.
    Handles loading provider and model configurations from YAML files
    with support for environment variable interpolation.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the ConfigLoader.

        Args:
            config_dir: Optional path to the configuration directory.
                        Defaults to the 'config' directory in the project root.
        """
        if config_dir is None:
            # Default to the config directory in the project root
            # Assuming this file is in utils/config_loader.py
            self.config_dir = Path(__file__).parent.parent / "config"
        else:
            self.config_dir = Path(config_dir)

        self._providers_cache: dict[str, ProviderConfig] = {}
        self._models_cache: dict[str, ModelConfig] = {}
        self._raw_providers: dict[str, Any] = {}
        self._raw_models: dict[str, Any] = {}

    def _interpolate(self, value: Any) -> Any:
        """
        Recursively interpolate ${ENV_VAR} patterns in strings.

        Args:
            value: The value to interpolate (str, dict, list, or other).

        Returns:
            The interpolated value.

        Raises:
            ValueError: If an environment variable is missing.
        """
        if isinstance(value, str):
            pattern = re.compile(r"\${(\w+)}")

            def replace(match):
                env_var = match.group(1)
                val = os.environ.get(env_var)
                if val is None:
                    raise ValueError(
                        f"Environment variable '{env_var}' not found for interpolation"
                    )
                return val

            return pattern.sub(replace, value)
        elif isinstance(value, dict):
            return {k: self._interpolate(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._interpolate(v) for v in value]
        return value

    def _load_yaml(self, filename: str) -> dict[str, Any]:
        """
        Load a YAML file from the config directory.

        Args:
            filename: The name of the YAML file.

        Returns:
            The parsed YAML data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the YAML is invalid.
        """
        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
                if data is None:
                    return {}
                return data
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")

    def load_provider(self, name: str) -> ProviderConfig:
        """
        Load a provider configuration by name.

        Args:
            name: The name of the provider.

        Returns:
            The ProviderConfig object.

        Raises:
            KeyError: If the provider is not found.
            ValueError: If interpolation fails or data is invalid.
        """
        if name in self._providers_cache:
            return self._providers_cache[name]

        if not self._raw_providers:
            self._raw_providers = self._load_yaml("providers.yaml")

        if name not in self._raw_providers:
            raise KeyError(f"Provider '{name}' not found in configuration")

        raw_data = self._raw_providers[name]
        interpolated_data = self._interpolate(raw_data)
        config = ProviderConfig.from_dict(interpolated_data)
        self._providers_cache[name] = config
        return config

    def load_model(self, name: str) -> ModelConfig:
        """
        Load a model configuration by name.

        Args:
            name: The name of the model.

        Returns:
            The ModelConfig object.

        Raises:
            KeyError: If the model is not found.
            ValueError: If interpolation fails or data is invalid.
        """
        if name in self._models_cache:
            return self._models_cache[name]

        if not self._raw_models:
            self._raw_models = self._load_yaml("models.yaml")

        if name not in self._raw_models:
            raise KeyError(f"Model '{name}' not found in configuration")

        raw_data = self._raw_models[name]
        interpolated_data = self._interpolate(raw_data)
        config = ModelConfig.from_dict(interpolated_data)
        self._models_cache[name] = config
        return config

    def list_providers(self) -> list[str]:
        """
        List all available provider names.

        Returns:
            A list of provider names.
        """
        if not self._raw_providers:
            try:
                self._raw_providers = self._load_yaml("providers.yaml")
            except FileNotFoundError:
                return []
        return list(self._raw_providers.keys())

    def list_models(self) -> list[str]:
        """
        List all available model names.

        Returns:
            A list of model names.
        """
        if not self._raw_models:
            try:
                self._raw_models = self._load_yaml("models.yaml")
            except FileNotFoundError:
                return []
        return list(self._raw_models.keys())
