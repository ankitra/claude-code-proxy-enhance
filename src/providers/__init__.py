from src.core.config import config
from src.providers.openai_provider import OpenAIProvider
from src.providers.anthropic_provider import AnthropicProvider


def create_provider():
    """Factory: returns the active provider based on configuration."""
    if config.provider == "anthropic":
        return AnthropicProvider(config)
    return OpenAIProvider(config)
