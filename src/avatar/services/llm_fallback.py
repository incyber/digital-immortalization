"""A language model with providers to fall back to.

A free tier that runs out mid-conversation is not a billing problem, it is a
person mid-sentence with a recreation of their dead parent that has stopped
answering. So quota exhaustion moves to the next provider rather than
surfacing as an error.

Every provider here speaks the OpenAI chat-completions API - Gemini, Groq and
xAI all expose one - so switching provider is a client swap rather than a
second code path.

What counts as "fall back" is deliberately narrow. Quota, rate limit and
server errors are transient and another provider will do; a malformed request
or a revoked key will fail identically everywhere, and retrying it just spends
somebody else's quota to produce the same error more slowly.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from loguru import logger
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)
from pipecat.services.openai.llm import OpenAILLMService

# Substrings seen in the body of a quota refusal. Providers disagree about
# status codes - Gemini returns 429 for both rate limiting and daily quota,
# and some return 403 for an exhausted free tier - so the text is checked too.
_QUOTA_MARKERS = (
    "quota",
    "resource_exhausted",
    "insufficient_quota",
    "billing",
    "credits",
    "exceeded your current",
    "rate limit",
)


@dataclass(frozen=True)
class Provider:
    """One place to send a completion."""

    name: str
    base_url: str
    api_key: str
    model: str

    @property
    def usable(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)


def is_transient(error: Exception) -> bool:
    """Whether another provider is likely to do better.

    Authentication failures are excluded on purpose: a bad key is bad
    everywhere it is configured, and rotating through providers to discover
    that wastes quota and hides the real problem.
    """
    if isinstance(error, AuthenticationError):
        return False

    if isinstance(error, (RateLimitError, InternalServerError, APITimeoutError)):
        return True

    if isinstance(error, (APIConnectionError, httpx.ConnectError, httpx.ReadTimeout)):
        return True

    if isinstance(error, APIStatusError):
        if error.status_code in (408, 409, 429, 500, 502, 503, 504):
            return True
        # A 403 may be an exhausted free tier or a genuine permission problem;
        # the body is what distinguishes them.
        body = str(getattr(error, "message", "") or error).lower()
        return any(marker in body for marker in _QUOTA_MARKERS)

    return any(marker in str(error).lower() for marker in _QUOTA_MARKERS)


class FallbackLLMService(OpenAILLMService):
    """An OpenAI-compatible LLM that moves to the next provider on exhaustion.

    Providers are tried in order. Once one fails transiently the service stays
    on the next one for the rest of the session rather than reverting on every
    turn: a provider that just returned "daily quota exceeded" will say the
    same thing thirty seconds later, and rechecking costs a failed request on
    the latency path of every reply.
    """

    def __init__(self, providers: list[Provider], **kwargs):
        usable = [p for p in providers if p.usable]
        if not usable:
            raise ValueError("at least one usable language model provider is required")

        self._providers = usable
        self._index = 0

        primary = usable[0]
        super().__init__(
            api_key=primary.api_key,
            base_url=primary.base_url,
            model=primary.model,
            **kwargs,
        )
        if len(usable) > 1:
            logger.info(
                f"language model: {primary.name}, falling back to "
                f"{', '.join(p.name for p in usable[1:])}"
            )

    @property
    def current(self) -> Provider:
        return self._providers[self._index]

    def _advance(self) -> bool:
        """Move to the next provider. False when there are none left."""
        if self._index + 1 >= len(self._providers):
            return False

        previous = self.current
        self._index += 1
        nxt = self.current

        self.set_full_model_name(nxt.model)
        self._client = self.create_client(api_key=nxt.api_key, base_url=nxt.base_url)
        logger.warning(f"language model {previous.name} exhausted; switched to {nxt.name}")
        return True

    async def get_chat_completions(self, context):
        """Try each remaining provider once.

        Failing over here rather than in the frame handler means the switch is
        invisible to the pipeline: the reply arrives late from a different
        provider instead of the turn producing an error frame.
        """
        last: Exception | None = None

        while True:
            try:
                return await super().get_chat_completions(context)
            except Exception as exc:
                last = exc
                if not is_transient(exc):
                    raise
                if not self._advance():
                    logger.error(
                        f"every language model provider is exhausted; last error: {last}"
                    )
                    raise


def build_providers(cfg) -> list[Provider]:
    """Providers in priority order, from configuration.

    The primary is whatever llm_base_url points at, which is Ollama locally and
    a hosted provider in deployment. Fallbacks are only included when a key is
    actually set, so an unconfigured one is absent rather than failing on first
    use.
    """
    providers = [
        Provider(
            name=cfg.llm_provider_name,
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key,
            model=cfg.llm_model,
        )
    ]

    if cfg.fallback_llm_api_key:
        providers.append(
            Provider(
                name=cfg.fallback_llm_provider_name,
                base_url=cfg.fallback_llm_base_url,
                api_key=cfg.fallback_llm_api_key,
                model=cfg.fallback_llm_model,
            )
        )

    return providers
