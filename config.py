"""Configuration management and LLM provider initialization."""

import os
from typing import Optional, Literal
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# Load environment variables
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # Model type selection: 'groq', 'gemini', or 'auto'
    MODEL_TYPE: str = os.getenv("MODEL_TYPE", os.getenv("LLM_PROVIDER", "auto")).lower()
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", MODEL_TYPE).lower()

    # Model configuration
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", os.getenv("MODEL_NAME", "qwen/qwen3.6-27b"))
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    MODEL_NAME: Optional[str] = os.getenv("MODEL_NAME")

    MAX_BUILD_RETRIES: int = int(os.getenv("MAX_BUILD_RETRIES", "3"))


settings = Settings()


def get_llm(
    provider: Optional[Literal["groq", "gemini", "auto"]] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    optional: bool = False,
    model_type: Optional[str] = None,
) -> Optional[BaseChatModel]:
    """Factory function to instantiate LLM client based on requested model_type/provider.

    Args:
        provider: 'groq', 'gemini', or 'auto'. Backwards compatible alias for model_type.
        model_name: Specific model name override.
        temperature: Model sampling temperature.
        optional: If True, returns None instead of raising ValueError when keys are missing.
        model_type: Specific model type ('groq', 'gemini', or 'auto').

    Returns:
        An initialized LangChain BaseChatModel instance or None.
    """
    selected_type = (model_type or provider or settings.MODEL_TYPE or settings.LLM_PROVIDER).lower()

    if selected_type == "auto":
        if settings.GROQ_API_KEY and settings.GOOGLE_API_KEY:
            selected_type = "groq_with_gemini_fallback"
        elif settings.GROQ_API_KEY:
            selected_type = "groq"
        elif settings.GOOGLE_API_KEY:
            selected_type = "gemini"
        else:
            if optional:
                return None
            raise ValueError(
                "No API keys configured. Set GROQ_API_KEY or GOOGLE_API_KEY in your .env file."
            )

    if selected_type in ("groq", "groq_with_gemini_fallback"):
        from langchain_groq import ChatGroq

        target_model = model_name or settings.GROQ_MODEL or "qwen/qwen3.6-27b"
        if target_model.startswith("groq/"):
            target_model = target_model.replace("groq/", "", 1)

        primary_llm = ChatGroq(
            model=target_model,
            groq_api_key=settings.GROQ_API_KEY or None,
            temperature=temperature,
            max_tokens=3500,
        )

        # Attach Gemini fallback if GOOGLE_API_KEY is available
        if settings.GOOGLE_API_KEY:
            from langchain_google_genai import ChatGoogleGenerativeAI

            gemini_model = settings.GEMINI_MODEL or "gemini-2.5-flash"
            if gemini_model.startswith("google/") or gemini_model.startswith("gemini/"):
                gemini_model = gemini_model.split("/", 1)[1]

            fallback_llm = ChatGoogleGenerativeAI(
                model=gemini_model,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=temperature,
            )
            return primary_llm.with_fallbacks([fallback_llm])

        return primary_llm

    elif selected_type == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        target_model = model_name or settings.GEMINI_MODEL or "gemini-2.5-flash"
        if target_model.startswith("google/") or target_model.startswith("gemini/"):
            target_model = target_model.split("/", 1)[1]

        return ChatGoogleGenerativeAI(
            model=target_model,
            google_api_key=settings.GOOGLE_API_KEY or None,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unsupported model_type '{selected_type}'. Choose 'groq', 'gemini', or 'auto'."
        )
