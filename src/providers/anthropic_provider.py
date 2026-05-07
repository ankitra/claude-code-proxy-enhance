import json
import traceback
from datetime import datetime
from typing import AsyncGenerator, Union

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.core.logging import logger
from src.models.claude import ClaudeMessagesRequest
from src.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Provider that forwards Claude-format requests directly to an
    Anthropic-compatible backend without any format translation."""

    def __init__(self, config):
        super().__init__(config)
        self.api_key = config.provider_api_key
        self.base_url = config.anthropic_base_url.rstrip("/")
        self.timeout = config.request_timeout

    async def create_message(
        self, request: ClaudeMessagesRequest, http_request: Request
    ) -> Union[dict, StreamingResponse]:
        # Build payload from the Claude request (already in the right format)
        payload = request.model_dump(exclude_none=True)

        # Map model name through the tier-based mapping
        payload["model"] = self.map_model(payload["model"])

        # Check if client disconnected before processing
        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        if request.stream:
            return StreamingResponse(
                self._stream_message(payload, http_request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )
        else:
            return await self._send_message(payload)

    async def _send_message(self, payload: dict) -> dict:
        """Forward a non-streaming request to the Anthropic-compatible backend."""
        headers = self._build_headers()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/messages",
                    json=payload,
                    headers=headers,
                )
                if response.status_code >= 400:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=self._classify_error(response.status_code, response.text),
                    )
                return response.json()
            except httpx.TimeoutException:
                raise HTTPException(
                    status_code=504,
                    detail="Request timed out connecting to the Anthropic-compatible backend.",
                )
            except httpx.RequestError as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to connect to backend: {e}",
                )

    async def _stream_message(
        self, payload: dict, http_request: Request
    ) -> AsyncGenerator[str, None]:
        """Forward a streaming request and pass through SSE events as-is."""
        headers = self._build_headers()
        payload["stream"] = True

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/messages",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code >= 400:
                        error_body = await response.aread()
                        yield self._sse_error(
                            self._classify_error(
                                response.status_code, error_body.decode()
                            )
                        )
                        return

                    async for line in response.aiter_lines():
                        # Check for client disconnection
                        if await http_request.is_disconnected():
                            logger.info("Client disconnected, stopping stream")
                            break
                        yield f"{line}\n"

            except httpx.TimeoutException:
                yield self._sse_error(
                    "Request timed out connecting to the Anthropic-compatible backend."
                )
            except httpx.RequestError as e:
                yield self._sse_error(f"Failed to connect to backend: {e}")

    def _build_headers(self) -> dict:
        """Build headers for requests to the Anthropic-compatible backend."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _classify_error(self, status_code: int, body: str) -> str:
        """Provide user-friendly error messages for common HTTP status codes."""
        if status_code == 401:
            return (
                "Authentication failed. Check your PROVIDER_API_KEY configuration."
            )
        elif status_code == 429:
            return "Rate limit exceeded. Please wait and try again."
        elif status_code == 400:
            # Try to extract a meaningful message from the response body
            try:
                err_data = json.loads(body)
                return err_data.get("error", {}).get("message", body)
            except (json.JSONDecodeError, AttributeError):
                return body
        elif status_code >= 500:
            return f"Backend server error ({status_code}). The provider may be experiencing issues."
        return body

    def _sse_error(self, message: str) -> str:
        """Format an error as an SSE error event."""
        error_data = json.dumps(
            {
                "type": "error",
                "error": {"type": "api_error", "message": message},
            }
        )
        return f"event: error\ndata: {error_data}\n\n"

    async def test_connection(self) -> dict:
        """Test connectivity to the Anthropic-compatible backend."""
        payload = {
            "model": self.map_model(self.config.small_model),
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        try:
            result = await self._send_message(payload)
            return {
                "status": "success",
                "message": "Successfully connected to provider API",
                "model_used": payload["model"],
                "timestamp": datetime.now().isoformat(),
            }
        except HTTPException as e:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "failed",
                    "error_type": "API Error",
                    "message": e.detail,
                    "timestamp": datetime.now().isoformat(),
                    "suggestions": [
                        "Check your PROVIDER_API_KEY is valid",
                        "Verify ANTHROPIC_BASE_URL is correct",
                        "Check if you have reached rate limits",
                    ],
                },
            )
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "failed",
                    "error_type": "Connection Error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
            )
