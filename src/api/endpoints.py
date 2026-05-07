from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Header, Depends

from src.core.config import config
from src.core.logging import logger
from src.models.claude import ClaudeMessagesRequest, ClaudeTokenCountRequest
from src.providers import create_provider

router = APIRouter()


def get_provider():
    """Dependency: returns the active provider based on configuration."""
    return create_provider()


async def validate_api_key(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Validate the client's API key from either x-api-key or Authorization header."""
    client_api_key = None

    if x_api_key:
        client_api_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        client_api_key = authorization.replace("Bearer ", "")

    if not config.client_api_key:
        return

    if not client_api_key or not config.validate_client_api_key(client_api_key):
        logger.warning("Invalid API key provided by client")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Please provide a valid API key.",
        )


@router.post("/v1/messages")
async def create_message(
    request: ClaudeMessagesRequest,
    http_request: Request,
    provider=Depends(get_provider),
    _=Depends(validate_api_key),
):
    try:
        logger.debug(
            f"Processing request: provider={config.provider}, model={request.model}, stream={request.stream}"
        )
        return await provider.create_message(request, http_request)
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        logger.error(f"Unexpected error processing request: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request: ClaudeTokenCountRequest,
    _: None = Depends(validate_api_key),
):
    try:
        total_chars = 0

        if request.system:
            if isinstance(request.system, str):
                total_chars += len(request.system)
            elif isinstance(request.system, list):
                for block in request.system:
                    if hasattr(block, "text"):
                        total_chars += len(block.text)

        for msg in request.messages:
            if msg.content is None:
                continue
            elif isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text") and block.text is not None:
                        total_chars += len(block.text)

        estimated_tokens = max(1, total_chars // 4)
        return {"input_tokens": estimated_tokens}

    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "provider": config.provider,
        "provider_api_configured": bool(config.provider_api_key),
        "client_api_key_validation": bool(config.client_api_key),
    }


@router.get("/test-connection")
async def test_connection(provider=Depends(get_provider)):
    """Test API connectivity to the configured backend."""
    try:
        return await provider.test_connection()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))
