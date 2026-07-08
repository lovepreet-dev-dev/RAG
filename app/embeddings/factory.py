"""Factory for creating embedding model instances.

Uses lazy imports so that heavy libraries (torch, sentence-transformers)
are only loaded when an embedding model is actually needed.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from app.config.constants import EMBEDDING_MODEL_MAP
from app.config.settings import EmbeddingModelType, Settings, get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_model_cache: dict[str, Embeddings] = {}


def get_embeddings_model(settings: Settings | None = None) -> Embeddings:
    """Create or return a cached embedding model.

    Args:
        settings: Optional settings override; defaults to global settings.

    Returns:
        A LangChain-compatible ``Embeddings`` instance.

    Raises:
        ValueError: If the embedding model type is not recognised.
    """
    if settings is None:
        settings = get_settings()

    cache_key = f"{settings.embedding_model.value}:{settings.embedding_device}"
    if cache_key in _model_cache:
        logger.debug("returning_cached_embeddings", model=cache_key)
        return _model_cache[cache_key]

    model_type = settings.embedding_model
    logger.info(
        "creating_embeddings_model",
        model_type=model_type.value,
        device=settings.embedding_device,
    )

    if model_type in (
        EmbeddingModelType.BGE_SMALL,
        EmbeddingModelType.BGE_LARGE,
        EmbeddingModelType.MINILM,
    ):
        model = _create_sentence_transformer(model_type, settings)
    elif model_type == EmbeddingModelType.NOMIC:
        model = _create_nomic(settings)
    elif model_type == EmbeddingModelType.OPENAI:
        model = _create_openai(settings)
    else:
        raise ValueError(f"Unsupported embedding model type: {model_type}")

    _model_cache[cache_key] = model
    return model


def _create_sentence_transformer(
    model_type: EmbeddingModelType,
    settings: Settings,
) -> Embeddings:
    """Create a SentenceTransformer-based embedding model.

    Args:
        model_type: Which sentence-transformer model to use.
        settings: Application settings for device and batch size.

    Returns:
        HuggingFaceEmbeddings wrapper.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    model_name = EMBEDDING_MODEL_MAP[model_type.value]
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.embedding_batch_size,
        },
    )


def _create_nomic(settings: Settings) -> Embeddings:
    """Create a Nomic embeddings model.

    Args:
        settings: Application settings for device and batch size.

    Returns:
        HuggingFaceEmbeddings configured for Nomic.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={
            "device": settings.embedding_device,
            "trust_remote_code": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.embedding_batch_size,
        },
    )


def _create_openai(settings: Settings) -> Embeddings:
    """Create an OpenAI-compatible embedding model.

    Args:
        settings: Application settings containing API keys and URLs.

    Returns:
        OpenAIEmbeddings instance.
    """
    from langchain_openai import OpenAIEmbeddings

    kwargs: dict[str, str | None] = {
        "model": settings.openai_embedding_model,
        "api_key": settings.openai_api_key or "dummy",
    }

    if settings.embedding_model == EmbeddingModelType.OPENAI:
        kwargs["base_url"] = settings.openai_base_url
    elif settings.get_llm_base_url():
        kwargs["base_url"] = settings.get_llm_base_url()

    return OpenAIEmbeddings(**kwargs)  # type: ignore[arg-type]
