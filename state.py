from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from loguru import logger


class StateManager:
    def __init__(self, state_dir: str = "state", symbol: str = "BTCUSDT"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.state_dir / f"grid_{symbol.lower()}.json"

    def save(self, data: dict) -> None:
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=self.state_dir, suffix=".tmp", prefix=".state_",
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                os.replace(tmp_path, self.filepath)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.error("Failed to save state: {}", e)

    def load(self) -> dict | None:
        if not self.filepath.exists():
            logger.info("No existing state file at {}", self.filepath)
            return None
        try:
            with open(self.filepath) as f:
                content = f.read()
            if not content.strip():
                logger.warning("State file is empty, treating as no state")
                self.filepath.rename(self.filepath.with_suffix(".empty"))
                return None
            data = json.loads(content)
            logger.info("State loaded from {}", self.filepath)
            return data
        except json.JSONDecodeError as e:
            logger.error("Corrupt state file: {} — backing up and starting fresh", e)
            backup = self.filepath.with_suffix(f".corrupt.{int(time.time())}.{os.getpid()}")
            try:
                self.filepath.rename(backup)
            except FileExistsError:
                backup = self.filepath.with_suffix(f".corrupt.{int(time.time()*1000)}.{os.getpid()}")
                self.filepath.rename(backup)
            logger.info("Corrupt state backed up to {}", backup)
            return None
        except Exception as e:
            logger.error("Failed to load state: {}", e)
            return None

    def delete(self) -> None:
        if self.filepath.exists():
            self.filepath.unlink()
            logger.info("State file deleted: {}", self.filepath)
