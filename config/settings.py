"""Configuration settings for the CPG Data Analysis Agent."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    
    default_llm_provider: Literal["openai", "gemini"] = Field(
        default="openai", 
        alias="DEFAULT_LLM_PROVIDER"
    )
    
    # OpenAI Models - Tiered architecture
    openai_fast_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_FAST_MODEL")
    openai_reasoning_model: str = Field(default="o3-mini", alias="OPENAI_REASONING_MODEL")
    
    # Gemini Models - Tiered architecture
    gemini_fast_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_FAST_MODEL")
    gemini_reasoning_model: str = Field(default="gemini-2.0-flash-thinking-exp", alias="GEMINI_REASONING_MODEL")
    
    temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    max_tokens: int = Field(default=4096, alias="MAX_TOKENS")
    
    data_dir: str = Field(default="data", alias="DATA_DIR")
    max_rows_display: int = Field(default=100, alias="MAX_ROWS_DISPLAY")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
