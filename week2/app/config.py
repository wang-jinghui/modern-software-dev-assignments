from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database settings
    database_url: str = Field(default="sqlite:///./data/app.db", description="Database URL")
    database_path: Path = Field(default=Path("./data/app.db"), description="Path to SQLite database")
    
    # Ollama settings
    ollama_model: str = Field(default="qwen3:4b", description="Ollama model to use for extraction")
    ollama_temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Ollama temperature setting")
    
    # App settings
    app_title: str = Field(default="Action Item Extractor", description="Application title")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    class Config:
        env_file = ".env"


# Create a global settings instance
settings = Settings()