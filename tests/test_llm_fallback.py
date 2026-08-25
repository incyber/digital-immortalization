"""Failover behaviour.

The property that matters: a provider running out of quota mid-conversation
must not surface to the person on the call.
"""

import httpx
import pytest
from openai import APIStatusError, AuthenticationError, InternalServerError, RateLimitError

from avatar.config import Settings
from avatar.services.llm_fallback import (
    FallbackLLMService,
    Provider,
    build_providers,
    is_transient,
)


def a_status_error(status: int, message: str = "") -> APIStatusError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(status, request=request, json={"error": {"message": message}})
    return APIStatusError(message, response=response, body=None)


def test_rate_limit_is_transient():
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(429, request=request, json={})
    assert is_transient(RateLimitError("slow down", response=response, body=None))


def test_server_errors_are_transient():
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(500, request=request, json={})
    assert is_transient(InternalServerError("boom", response=response, body=None))


def test_daily_quota_is_transient():
    assert is_transient(a_status_error(429, "You exceeded your current quota"))


def test_resource_exhausted_is_transient():
    # Gemini's wording for a spent free tier.
    assert is_transient(a_status_error(429, "RESOURCE_EXHAUSTED"))


def test_missing_credits_is_transient():
    # xAI's wording, seen on a team with no credits purchased.
    assert is_transient(
        a_status_error(403, "Your newly created team doesn't have any credits or licenses yet")
    )


def test_a_bad_key_is_not_transient():
    # A revoked key fails identically everywhere. Rotating providers to
    # discover that spends somebody else's quota and hides the real problem.
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(401, request=request, json={})
    assert not is_transient(AuthenticationError("bad key", response=response, body=None))


def test_a_malformed_request_is_not_transient():
    assert not is_transient(a_status_error(400, "invalid 'messages': expected an array"))


def test_a_connection_failure_is_transient():
    assert is_transient(httpx.ConnectError("connection refused"))


PRIMARY = Provider("primary", "https://a.test/v1", "key-a", "model-a")
BACKUP = Provider("backup", "https://b.test/v1", "key-b", "model-b")


def test_at_least_one_provider_is_required():
    with pytest.raises(ValueError, match="at least one"):
        FallbackLLMService([Provider("empty", "", "", "")])


def test_providers_without_a_key_are_dropped():
    service = FallbackLLMService([PRIMARY, Provider("nokey", "https://c.test/v1", "", "m")])
    assert len(service._providers) == 1


def test_the_first_provider_is_used_first():
    assert FallbackLLMService([PRIMARY, BACKUP]).current.name == "primary"


def test_advancing_switches_provider_and_model():
    service = FallbackLLMService([PRIMARY, BACKUP])
    assert service._advance() is True
    assert service.current.name == "backup"
    assert service.get_full_model_name() == "model-b"


def test_advancing_past_the_last_provider_fails():
    service = FallbackLLMService([PRIMARY])
    assert service._advance() is False


async def test_a_quota_error_moves_to_the_backup_and_succeeds():
    service = FallbackLLMService([PRIMARY, BACKUP])
    calls = []

    async def flaky(self, context):
        calls.append(service.current.name)
        if service.current.name == "primary":
            raise a_status_error(429, "You exceeded your current quota")
        return "completion-from-backup"

    from pipecat.services.openai.llm import BaseOpenAILLMService

    original = BaseOpenAILLMService.get_chat_completions
    BaseOpenAILLMService.get_chat_completions = flaky
    try:
        result = await service.get_chat_completions(context=None)
    finally:
        BaseOpenAILLMService.get_chat_completions = original

    assert result == "completion-from-backup"
    assert calls == ["primary", "backup"]


async def test_a_bad_key_is_raised_without_burning_the_backup():
    service = FallbackLLMService([PRIMARY, BACKUP])
    calls = []

    async def always_unauthorized(self, context):
        calls.append(service.current.name)
        request = httpx.Request("POST", "https://example.test")
        response = httpx.Response(401, request=request, json={})
        raise AuthenticationError("bad key", response=response, body=None)

    from pipecat.services.openai.llm import BaseOpenAILLMService

    original = BaseOpenAILLMService.get_chat_completions
    BaseOpenAILLMService.get_chat_completions = always_unauthorized
    try:
        with pytest.raises(AuthenticationError):
            await service.get_chat_completions(context=None)
    finally:
        BaseOpenAILLMService.get_chat_completions = original

    assert calls == ["primary"], "a bad key must not be retried against the backup"


async def test_exhausting_every_provider_raises():
    service = FallbackLLMService([PRIMARY, BACKUP])

    async def always_out(self, context):
        raise a_status_error(429, "quota exceeded")

    from pipecat.services.openai.llm import BaseOpenAILLMService

    original = BaseOpenAILLMService.get_chat_completions
    BaseOpenAILLMService.get_chat_completions = always_out
    try:
        with pytest.raises(APIStatusError):
            await service.get_chat_completions(context=None)
    finally:
        BaseOpenAILLMService.get_chat_completions = original


def test_no_fallback_is_configured_without_a_key():
    assert len(build_providers(Settings(_env_file=None))) == 1


def test_a_configured_fallback_is_included():
    cfg = Settings(_env_file=None, fallback_llm_api_key="gsk_something")
    providers = build_providers(cfg)
    assert len(providers) == 2
    assert providers[1].base_url.startswith("https://api.groq.com")


def test_reasoning_models_get_their_effort_turned_down():
    # Left at the default these return an empty content field, which on a live
    # call is the avatar saying nothing at all.
    from avatar.services.llm_fallback import extra_params_for

    for model in ("openai/gpt-oss-120b", "groq/compound-mini", "qwen/qwen3.6-27b"):
        assert extra_params_for(model) == {"reasoning_effort": "low"}


def test_ordinary_models_get_no_extra_parameters():
    # reasoning_effort is rejected outright by providers that do not know it.
    from avatar.services.llm_fallback import extra_params_for

    assert extra_params_for("llama3.2:3b") == {}
    assert extra_params_for("gemini-2.5-flash") == {}


def test_extra_params_follow_the_active_provider():
    plain = Provider("plain", "https://a.test/v1", "k", "llama3.2:3b")
    reasoning = Provider(
        "reasoning", "https://b.test/v1", "k", "openai/gpt-oss-120b",
        extra_params={"reasoning_effort": "low"},
    )
    service = FallbackLLMService([plain, reasoning])

    # The base builder reads real invocation params; only the merge is under
    # test here, so it is stubbed out.
    from pipecat.services.openai.llm import BaseOpenAILLMService

    original = BaseOpenAILLMService.build_chat_completion_params
    BaseOpenAILLMService.build_chat_completion_params = lambda self, p: {"model": "x"}
    try:
        assert "reasoning_effort" not in service.build_chat_completion_params(None)
        service._advance()
        assert service.build_chat_completion_params(None)["reasoning_effort"] == "low"
    finally:
        BaseOpenAILLMService.build_chat_completion_params = original


def test_the_configured_default_fallback_model_exists_in_config():
    cfg = Settings(_env_file=None, fallback_llm_api_key="gsk_x")
    provider = build_providers(cfg)[1]
    assert provider.model == "openai/gpt-oss-120b"
    assert provider.extra_params == {"reasoning_effort": "low"}


def test_the_request_carries_the_active_providers_model():
    # The failure this prevents: the client switches to the backup but the
    # model name does not, so the backup is asked for a model it has never
    # heard of and answers 404.
    plain = Provider("primary", "https://a.test/v1", "k", "llama3.2:3b")
    backup = Provider("backup", "https://b.test/v1", "k", "openai/gpt-oss-120b")
    service = FallbackLLMService([plain, backup])

    from pipecat.services.openai.llm import BaseOpenAILLMService

    original = BaseOpenAILLMService.build_chat_completion_params
    BaseOpenAILLMService.build_chat_completion_params = lambda self, p: {"model": "stale"}
    try:
        assert service.build_chat_completion_params(None)["model"] == "llama3.2:3b"
        service._advance()
        assert service.build_chat_completion_params(None)["model"] == "openai/gpt-oss-120b"
    finally:
        BaseOpenAILLMService.build_chat_completion_params = original
