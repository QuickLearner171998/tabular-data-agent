"""LLM provider implementations for OpenAI and Gemini with tiered model support."""

from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import get_settings

LLMProvider = Literal["openai", "gemini"]
ModelTier = Literal["fast", "reasoning"]


def get_llm(
    provider: LLMProvider | None = None,
    tier: ModelTier = "fast",
    temperature: float | None = None,
    model: str | None = None,
) -> BaseChatModel:
    """
    Get an LLM instance based on the specified provider and tier.
    
    Args:
        provider: The LLM provider to use. Defaults to settings.default_llm_provider.
        tier: Model tier - "fast" for quick tasks (intent, routing), 
              "reasoning" for complex analysis (only when needed).
        temperature: Override the default temperature.
        model: Override the model (ignores tier if provided).
    
    Returns:
        A configured LLM instance.
    
    Model Selection:
        OpenAI:
            - fast: gpt-4.1-mini (intent detection, routing, simple queries)
            - reasoning: o3-mini (complex analysis, multi-step reasoning)
        
        Gemini:
            - fast: gemini-2.0-flash (quick responses, simple tasks)
            - reasoning: gemini-2.0-flash-thinking-exp (deep analysis)
    """
    settings = get_settings()
    provider = provider or settings.default_llm_provider
    temperature = temperature if temperature is not None else settings.temperature
    
    if provider == "openai":
        if model is None:
            model = (
                settings.openai_fast_model 
                if tier == "fast" 
                else settings.openai_reasoning_model
            )
        
        # o3-mini doesn't support temperature parameter
        model_kwargs = {
            "model": model,
            "api_key": settings.openai_api_key,
            "max_tokens": settings.max_tokens,
        }
        
        if not model.startswith("o3") and not model.startswith("o1"):
            model_kwargs["temperature"] = temperature
        
        return ChatOpenAI(**model_kwargs)
    
    elif provider == "gemini":
        if model is None:
            model = (
                settings.gemini_fast_model 
                if tier == "fast" 
                else settings.gemini_reasoning_model
            )
        
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=settings.google_api_key,
            max_output_tokens=settings.max_tokens,
        )
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_fast_llm(provider: LLMProvider | None = None) -> BaseChatModel:
    """Get a fast LLM for intent detection and routing."""
    return get_llm(provider=provider, tier="fast")


def get_reasoning_llm(provider: LLMProvider | None = None) -> BaseChatModel:
    """Get a reasoning LLM for complex analysis tasks."""
    return get_llm(provider=provider, tier="reasoning")
