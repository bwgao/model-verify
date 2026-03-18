import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from utils.types import ProbeResult, ScoreCard


class DataStore:
    """
    DataStore handles persistence of baselines and verification results.
    Uses file-based JSON storage with atomic writes.
    """

    def __init__(self, base_path: Path = Path("baselines")):
        self.base_path = base_path
        self.results_path = base_path / "results"

        # Ensure base directories exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)

    def _atomic_write(self, file_path: Path, data: dict[str, Any]) -> None:
        """Writes data to a temporary file and then renames it to the target path."""
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a temporary file in the same directory as the target file
        fd, temp_path = tempfile.mkstemp(dir=file_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            # Atomic rename
            os.replace(temp_path, file_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def save_baseline(self, model: str, probe_type: str, data: dict[str, Any]) -> Path:
        """
        Saves a baseline for a specific model and probe type.
        Path: baselines/{model}/{probe_type}.json
        """
        file_path = self.base_path / model / f"{probe_type}.json"
        self._atomic_write(file_path, data)
        return file_path

    def load_baseline(self, model: str, probe_type: str) -> Optional[dict[str, Any]]:
        """
        Loads a baseline for a specific model and probe type.
        Returns None if the file does not exist.
        """
        file_path = self.base_path / model / f"{probe_type}.json"
        if not file_path.exists():
            return None

        with open(file_path, "r") as f:
            return json.load(f)

    def list_baselines(self, model: str) -> list[str]:
        """
        Lists all available probe types for a given model.
        """
        model_dir = self.base_path / model
        if not model_dir.exists() or not model_dir.is_dir():
            return []

        return [f.stem for f in model_dir.glob("*.json") if f.is_file()]

    def save_result(
        self, provider: str, model: str, result: Union[ProbeResult, ScoreCard, dict[str, Any]]
    ) -> Path:
        """
        Saves a verification result or scorecard.
        Path: results/{provider}/{model}/{timestamp}.json
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = self.results_path / provider / model / f"{timestamp}.json"

        if isinstance(result, dict):
            data = result
        else:
            data = result.to_dict()
        self._atomic_write(file_path, data)
        return file_path

    def load_results(self, provider: str, model: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Loads the most recent verification results for a provider and model.
        """
        model_dir = self.results_path / provider / model
        if not model_dir.exists() or not model_dir.is_dir():
            return []

        # Get all json files, sorted by name (which includes timestamp) descending
        files = sorted(model_dir.glob("*.json"), key=lambda x: x.name, reverse=True)

        results = []
        for file_path in files[:limit]:
            try:
                with open(file_path, "r") as f:
                    results.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                continue

        return results

    def delete_baseline(self, model: str, probe_type: str) -> bool:
        """
        Deletes a baseline file. Returns True if deleted, False if not found.
        """
        file_path = self.base_path / model / f"{probe_type}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
