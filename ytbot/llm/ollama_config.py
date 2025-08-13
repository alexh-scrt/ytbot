# config.py
from __future__ import annotations

from functools import lru_cache

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMSettings(BaseModel):
    model: str = Field(default="llama3.3:70b", repr=False)
    base_url: str = Field(default="http://localhost:11434", repr=False)
    embedding_model: str = Field(default="nomic-embed-text", repr=False)
    temperature: float = Field(default=0.0, repr=False)

    @field_validator("temperature")
    @classmethod
    def _valid_temperature(cls, v: int) -> int:
        if not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v


class Settings(BaseSettings):
    # Top-level
    environment: Literal["local", "dev", "staging", "prod"] = "local"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # Nested groups
    llm: LLMSettings = LLMSettings()

    # Pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_prefix="YTB_",                 # e.g., APP_DATABASE__HOST
        env_file=".env",                   # load from .env if present
        env_nested_delimiter="__",         # nested model override
        case_sensitive=False,
        extra="ignore",                    # ignore unexpected env vars
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
