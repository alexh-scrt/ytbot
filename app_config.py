# config.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "app"
    password: SecretStr = Field(default=SecretStr("dev-password"), repr=False)
    name: str = "appdb"
    sslmode: Literal["disable", "require", "verify-ca", "verify-full"] = "disable"

    def sqlalchemy_url(self) -> str:
        # Keep password out of reprs/logs; only build on demand
        pw = self.password.get_secret_value()
        return f"postgresql+psycopg://{self.user}:{pw}@{self.host}:{self.port}/{self.name}?sslmode={self.sslmode}"

    @field_validator("port")
    @classmethod
    def _valid_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("port must be between 1 and 65535")
        return v


class RedisSettings(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0


class SecuritySettings(BaseModel):
    # Prefer Docker/K8s secrets; env override supported
    jwt_secret: SecretStr = Field(default=SecretStr("change-me"), repr=False)
    cors_allow_origins: list[str] = ["*"]

    # Optional pattern: allow _FILE envs to point to a file containing the secret
    jwt_secret_file: Path | None = None

    @field_validator("jwt_secret")
    @classmethod
    def _ensure_secret(cls, v: SecretStr) -> SecretStr:
        if v.get_secret_value() in {"change-me", ""}:
            # Not raising, but strongly encourage override in prod
            pass
        return v

    @field_validator("jwt_secret", mode="before")
    @classmethod
    def _read_from_file_if_present(cls, v, info):
        # If *_FILE was set (via env), read it
        path = info.data.get("jwt_secret_file")
        if path:
            return SecretStr(Path(path).read_text().strip())
        return v


class Settings(BaseSettings):
    # Top-level
    environment: Literal["local", "dev", "staging", "prod"] = "local"
    app_name: str = "my-app"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # Nested groups
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    security: SecuritySettings = SecuritySettings()

    # Pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_prefix="APP_",                 # e.g., APP_DATABASE__HOST
        env_file=".env",                   # load from .env if present
        env_nested_delimiter="__",         # nested model override
        secrets_dir="/run/secrets",        # Docker/K8s secrets
        case_sensitive=False,
        extra="ignore",                    # ignore unexpected env vars
    )


# Typical usage pattern (FastAPI or scripts)
from functools import lru_cache

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

if __name__ == "__main__":
    s = get_settings()
    # Safe printing: secrets are masked
    print(s.model_dump(exclude={"security"}))         # redact security block entirely
    print("DB URL (runtime only):", s.database.sqlalchemy_url())
