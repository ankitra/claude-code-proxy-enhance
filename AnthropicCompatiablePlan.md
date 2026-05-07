# Plan: Provider Pattern Refactor + Anthropic-Compatible Passthrough

## Context

Tracking issue: https://github.com/ankitra/claude-code-proxy-enhance/issues/1

The proxy currently hardcodes OpenAI as the only backend, translating
Claude-format requests to OpenAI format and back. Users who want to connect
to an Anthropic-compatible API endpoint (e.g., DeepSeek's Anthropic-format
API) still go through unnecessary translation, causing API failures.

This plan refactors the proxy into a **provider pattern** that supports
both OpenAI-compatible and Anthropic-compatible backends, with a UI toggle
to select which one to use.

## Requirements (from issue)

1. **Provider pattern** — shared base provider with concrete implementations
2. **API Type selector** in UI — "Open AI" or "Anthropic"
3. **Direct forwarding** for Anthropic-compatible backends (no translation)
4. **Config fields**: Base URL, API Key, Optional client key, Big/Middle/Small models, Performance settings, API Type
5. **Naming changes**: `OPENAI_API_KEY` → `PROVIDER_API_KEY`, `ANTHROPIC_API_KEY` → `CLIENT_API_KEY`
6. **Backward compat** — old env vars still work, existing OpenAI mode unaffected

---

## Design

### Config Field Migration

| Old env var | New env var | Config attr | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | `PROVIDER_API_KEY` | `provider_api_key` | Key sent to backend |
| `OPENAI_BASE_URL` | `PROVIDER_BASE_URL` | `provider_base_url` | Backend endpoint URL |
| `ANTHROPIC_API_KEY` | `CLIENT_API_KEY` | `client_api_key` | Optional client auth |
| — | `PROVIDER` | `provider` | `"openai"` or `"anthropic"` |
| — | `ANTHROPIC_BASE_URL` | `anthropic_base_url` | Anthropic-specific endpoint |

Backward compat: `Config.__init__()` reads new name first, falls back to old name.

The `update()` and `to_dict()` methods use new names; `update()` also accepts
old names for one-time migration when a profile is saved.

### Provider Pattern

New package `src/providers/`:

```
src/providers/
├── __init__.py        # create_provider() factory
├── base.py            # BaseProvider ABC
├── openai_provider.py # OpenAIProvider
└── anthropic_provider.py  # AnthropicProvider
```

#### BaseProvider (`base.py`)

```python
class BaseProvider(ABC):
    def __init__(self, config):
        self.config = config
        self.model_manager = ModelManager(config)

    @abstractmethod
    async def create_message(
        self, request: ClaudeMessagesRequest, http_request: Request
    ) -> Union[dict, StreamingResponse]:
        ...

    @abstractmethod
    async def test_connection(self) -> dict:
        ...

    def map_model(self, claude_model: str) -> str:
        return self.model_manager.map_claude_model_to_openai(claude_model)
```

#### OpenAIProvider (`openai_provider.py`)

Moves existing endpoint logic into the provider:
- Uses `OpenAIClient` (from `client.py`) for backend communication
- Uses `convert_claude_to_openai()` for request translation
- Uses `convert_openai_to_claude_response()` / streaming converter for response translation
- Preserves all existing cancellation and error handling

No functional changes — just extracting existing code from `endpoints.py`
into the provider class.

#### AnthropicProvider (`anthropic_provider.py`)

New implementation using `httpx.AsyncClient` (already a transitive dependency):

```
Client (Claude-format)          Proxy                     Anthropic-Compatible Backend
       │                          │                              │
       │ POST /v1/messages        │                              │
       │ (Claude format)          │   Strip incoming x-api-key    │
       │                          │   Map model name              │
       │                          │   Inject configured API key   │
       │                          │   POST {anthropic_base_url}/messages │
       │                          │   (same Claude format)        │
       │                          │──────────────────────────────>│
       │                          │                              │
       │  <── SSE or JSON ────────│<─── SSE or JSON ────────────│
```

Key behaviors:
- **No request/response translation** — payload passes through as-is
- **Model mapping** still applied via `ModelManager` (big/middle/small mapping works for any backend)
- **API key injection** — strips incoming client `x-api-key`, replaces with `provider_api_key`
- **Streaming passthrough** — raw SSE events forwarded byte-for-byte (with client disconnection detection)
- **Headers**: sets `x-api-key`, `anthropic-version: 2023-06-01`, `content-type: application/json`
- **Non-streaming**: returns JSON response as-is
- **Error mapping**: HTTP errors from backend → `HTTPException` with clear messages

### Endpoint Changes (`src/api/endpoints.py`)

Becomes thin — delegates everything to the active provider:

```python
def get_provider():
    if config.provider == "anthropic":
        return AnthropicProvider(config)
    return OpenAIProvider(config)

@router.post("/v1/messages")
async def create_message(request, http_request, _ = Depends(validate_api_key)):
    provider = get_provider()
    return await provider.create_message(request, http_request)
```

- `/health` shows active provider + whether its API key is configured
- `/test-connection` delegates to `provider.test_connection()`
- `/v1/messages/count_tokens` stays unchanged (local estimation)

### Model Manager (`src/core/model_manager.py`)

Already provider-agnostic. No changes needed — the opus/sonnet/haiku → big/middle/small mapping works for any backend. Models starting with known prefixes pass through as-is.

### UI Changes (`src/web/templates/index.html`)

Add **API Type** dropdown at the top:

```
API Type: [Open AI ▼]   or   [Anthropic ▼]
```

All six field groups are **always visible** (labeled generically):

| Section | Fields |
|---|---|
| **API Type** | Dropdown: "Open AI" / "Anthropic" |
| **Base URL** | Single field, label stays "Base URL" — maps to `PROVIDER_BASE_URL` (OpenAI mode) or `ANTHROPIC_BASE_URL` (Anthropic mode) |
| **API Key** | "Provider API Key" — maps to `PROVIDER_API_KEY`, sent to backend as auth |
| **Client Auth** | "Client API Key (Optional)" — maps to `CLIENT_API_KEY`, used to validate incoming requests |
| **Models** | Big/Middle/Small model name fields (untouched, work for both providers) |
| **Performance** | Max/Min tokens, timeout, retries (untouched) |

JavaScript toggles which base URL field value is displayed when switching API Type
(i.e., `PROVIDER_BASE_URL` vs `ANTHROPIC_BASE_URL` in the same input field).

### Backward Compatibility

- Old env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_BASE_URL`) continue to work
- Old profiles in `profiles.json` using old key names are migrated on first save
- Existing OpenAI behavior is identical after migration (preserved in `OpenAIProvider`)

---

## Files to Create / Modify

| File | Action |
|---|---|
| `src/providers/__init__.py` | **Create** — `create_provider()` factory |
| `src/providers/base.py` | **Create** — `BaseProvider` ABC |
| `src/providers/openai_provider.py` | **Create** — `OpenAIProvider` (moves logic from endpoints) |
| `src/providers/anthropic_provider.py` | **Create** — `AnthropicProvider` (new passthrough) |
| `src/core/config.py` | **Modify** — rename fields, add `provider`, `anthropic_base_url`, backward compat |
| `src/core/client.py` | **Modify** — `classify_openai_error` → `classify_provider_error` (cosmetic), no other changes |
| `src/api/endpoints.py` | **Modify** — delegate to provider, remove inline conversion logic |
| `src/web/templates/index.html` | **Modify** — add API Type selector, rename labels, show/hide base URL per type |
| `.env.example` | **Modify** — document new config keys |

---

## Verification

1. **OpenAI mode regression** (default `PROVIDER=openai`, old env vars):
   - Streaming and non-streaming requests work
   - Tool calls work
   - Cancellation works
   - Client API key auth works (when `CLIENT_API_KEY` is set)

2. **Anthropic-compatible passthrough** (`PROVIDER=anthropic`):
   - Point at any Anthropic-compatible backend
   - Request forwarded without translation
   - Streaming SSE passes through byte-for-byte
   - Model mapping correctly applies big/middle/small config
   - Client disconnection cancels upstream

3. **Config migration**:
   - Old `profiles.json` with `OPENAI_API_KEY` loads successfully
   - First save migrates to new key names
   - Old env vars picked up as fallbacks

4. **UI**:
   - API Type dropdown switches between modes
   - Base URL field shows correct value for selected mode
   - Profile save/restore works for both modes
   - Switching profiles with different API types works
