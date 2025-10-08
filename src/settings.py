"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    gemini_api_key: Optional[str] = None
    kling_access_key: Optional[str] = None
    kling_secret_key: Optional[str] = None
    creative_variants_log_level: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            kling_access_key=os.getenv("KLING_ACCESS_KEY"),
            kling_secret_key=os.getenv("KLING_SECRET_KEY"),
            creative_variants_log_level=os.getenv("CREATIVE_VARIANTS_LOG_LEVEL"),
        )


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings.from_env()
