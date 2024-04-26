from typing import Optional
from pathlib import Path

from pyangstrom.config import Config
from pyangstrom.caching import Cache


def analyze_recording(
        recording_path: str | Path,
        config: str | Path | Config,
        *,
        memory_cache: Optional[Cache] = None,
        recording_cache_path: Optional[str | Path] = None,
):
    pass
