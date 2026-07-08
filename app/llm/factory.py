"""LLM provider factory.

Creates LangChain ``BaseChatModel`` instances for the configured provider.
All heavy imports are deferred so the module loads instantly.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.config.settings import LLMProvider, Settings, get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    """Create the configured LLM instance.

    Args:
        settings: Optional settings override.

    Returns:
        A LangChain ``BaseChatModel`` for the configured provider.

    Raises:
        ValueError: If the LLM provider is not supported.
    """
    if settings is None:
        settings = get_settings()

    provider = settings.llm_provider
    logger.info(
        "creating_llm",
        provider=provider.value,
        model=settings.get_llm_model_name(),
    )

    factories = {
        LLMProvider.OPENAI: _create_openai,
        LLMProvider.OLLAMA: _create_ollama,
        LLMProvider.ANTHROPIC: _create_anthropic,
        LLMProvider.GEMINI: _create_gemini,
        LLMProvider.LMSTUDIO: _create_lmstudio,
        LLMProvider.OPENROUTER: _create_openrouter,
    }

    factory = factories.get(provider)
    if factory is None:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    return factory(settings)


def _create_openai(settings: Settings) -> BaseChatModel:
    """Create an OpenAI chat model.

    Args:
        settings: Application settings.

    Returns:
        ``ChatOpenAI`` instance.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0.3,
        max_tokens=2048,
        request_timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def _create_ollama(settings: Settings) -> BaseChatModel:
    """Create an Ollama chat model.

    Args:
        settings: Application settings.

    Returns:
        ``ChatOllama`` instance.
    """
    from langchain_community.chat_models import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.3,
    )


def _create_anthropic(settings: Settings) -> BaseChatModel:
    """Create an Anthropic chat model.

    Args:
        settings: Application settings.

    Returns:
        ``ChatAnthropic`` instance.
    """
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=0.3,
        max_tokens=2048,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def _create_gemini(settings: Settings) -> BaseChatModel:
    """Create a Google Gemini chat model.

    Args:
        settings: Application settings.

    Returns:
        ``ChatGoogleGenerativeAI`` instance.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
    )


def _create_lmstudio(settings: Settings) -> BaseChatModel:
    """Create an LM Studio chat model (OpenAI-compatible local server).

    Args:
        settings: Application settings.

    Returns:
        ``ChatOpenAI`` instance pointing to LM Studio.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.lmstudio_model,
        base_url=settings.lmstudio_base_url,
        api_key="not-needed",
        temperature=0.3,
    )


def _create_openrouter(settings: Settings) -> BaseChatModel:
    """Create an OpenRouter chat model.

    Args:
        settings: Application settings.

    Returns:
        ``ChatOpenAI`` instance pointing to OpenRouter.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        temperature=0.3,
        max_tokens=2048,
        request_timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
