from abc import ABC, abstractmethod
from typing import Union

from fastapi import Request
from fastapi.responses import StreamingResponse

from src.core.model_manager import ModelManager
from src.models.claude import ClaudeMessagesRequest


class BaseProvider(ABC):
    """Abstract base class for all API providers."""

    def __init__(self, config):
        self.config = config
        self.model_manager = ModelManager(config)

    @abstractmethod
    async def create_message(
        self, request: ClaudeMessagesRequest, http_request: Request
    ) -> Union[dict, StreamingResponse]:
        """Process a Claude-format request and return the response."""
        ...

    @abstractmethod
    async def test_connection(self) -> dict:
        """Test connectivity to the backend API."""
        ...

    def map_model(self, claude_model: str) -> str:
        """Map a Claude model name to the configured backend model."""
        return self.model_manager.map_claude_model_to_openai(claude_model)
