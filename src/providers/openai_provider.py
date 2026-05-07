import uuid
import traceback
from datetime import datetime
from typing import Union

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.core.client import OpenAIClient
from src.core.logging import logger
from src.conversion.request_converter import convert_claude_to_openai
from src.conversion.response_converter import (
    convert_openai_to_claude_response,
    convert_openai_streaming_to_claude_with_cancellation,
)
from src.models.claude import ClaudeMessagesRequest
from src.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """Provider that translates Claude-format requests to OpenAI-compatible
    backends, then translates responses back to Claude format."""

    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAIClient(
            api_key=config.provider_api_key,
            base_url=config.provider_base_url,
            timeout=config.request_timeout,
            api_version=config.azure_api_version,
        )

    async def create_message(
        self, request: ClaudeMessagesRequest, http_request: Request
    ) -> Union[dict, StreamingResponse]:
        request_id = str(uuid.uuid4())

        # Convert Claude request to OpenAI format
        openai_request = convert_claude_to_openai(request, self.model_manager)

        # Check if client disconnected before processing
        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        if request.stream:
            return await self._handle_streaming(openai_request, request, http_request, request_id)
        else:
            return await self._handle_non_streaming(openai_request, request, request_id)

    async def _handle_streaming(self, openai_request: dict, request: ClaudeMessagesRequest,
                                http_request: Request, request_id: str) -> StreamingResponse:
        try:
            openai_stream = self.client.create_chat_completion_stream(
                openai_request, request_id
            )
            return StreamingResponse(
                convert_openai_streaming_to_claude_with_cancellation(
                    openai_stream,
                    request,
                    logger,
                    http_request,
                    self.client,
                    request_id,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )
        except HTTPException as e:
            logger.error(f"Streaming error: {e.detail}")
            logger.error(traceback.format_exc())
            error_message = self.client.classify_openai_error(e.detail)
            error_response = {
                "type": "error",
                "error": {"type": "api_error", "message": error_message},
            }
            return JSONResponse(status_code=e.status_code, content=error_response)

    async def _handle_non_streaming(self, openai_request: dict, request: ClaudeMessagesRequest,
                                    request_id: str) -> dict:
        openai_response = await self.client.create_chat_completion(
            openai_request, request_id
        )
        return convert_openai_to_claude_response(openai_response, request)

    async def test_connection(self) -> dict:
        """Test connectivity to the OpenAI-compatible backend."""
        try:
            test_response = await self.client.create_chat_completion(
                {
                    "model": self.config.small_model,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 5,
                },
                "test-connection-request"
            )
            return {
                "status": "success",
                "message": "Successfully connected to provider API",
                "model_used": self.config.small_model,
                "timestamp": datetime.now().isoformat(),
                "response_id": test_response.get("id", "unknown"),
            }
        except Exception as e:
            logger.error(f"API connectivity test failed: {e}")
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "failed",
                    "error_type": "API Error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "suggestions": [
                        "Check your PROVIDER_API_KEY is valid",
                        "Verify your API key has the necessary permissions",
                        "Check if you have reached rate limits",
                    ],
                },
            )
