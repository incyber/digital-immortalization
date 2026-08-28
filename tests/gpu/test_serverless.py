"""Serverless GPU work.

The design point these defend: nothing is allocated between jobs, so there is
no lifecycle to leak. The tests that used to matter for Pods - does teardown
run on exception, does the sweep find orphans - have no equivalent here,
because there is nothing to tear down.

What remains worth asserting is that we never silently accept an endpoint
configured to bill while idle, and that an unknown state is treated as still
running rather than finished.
"""

import pytest

from avatar.gpu.serverless import (
    DEFAULT_MAX_WORKERS,
    JobResult,
    JobState,
    ServerlessClient,
    ServerlessError,
    assert_endpoint_is_safe,
)


def test_an_api_key_is_required():
    with pytest.raises(ServerlessError, match="API key"):
        ServerlessClient("", "endpoint")


def test_an_endpoint_id_is_required():
    with pytest.raises(ServerlessError, match="endpoint id"):
        ServerlessClient("key", "")


@pytest.mark.parametrize(
    "state,terminal",
    [
        (JobState.IN_QUEUE, False),
        (JobState.IN_PROGRESS, False),
        (JobState.COMPLETED, True),
        (JobState.FAILED, True),
        (JobState.CANCELLED, True),
        (JobState.TIMED_OUT, True),
    ],
)
def test_terminal_states_are_classified(state, terminal):
    assert state.terminal is terminal


def test_only_a_running_job_is_billing():
    assert JobState.IN_PROGRESS.billing
    assert not JobState.IN_QUEUE.billing
    assert not JobState.COMPLETED.billing


def test_cost_is_measured_not_assumed():
    """Read back from the platform, so a pricier GPU shows up in the number."""
    half_hour = JobResult(id="j", state=JobState.COMPLETED, execution_ms=1_800_000)
    assert 0.21 < half_hour.cost < 0.23
    assert JobResult(id="j", state=JobState.COMPLETED, execution_ms=0).cost == 0.0


def test_an_endpoint_that_keeps_workers_warm_is_refused():
    """The whole safety argument is workersMin=0. Anything else bills while idle."""
    problems = assert_endpoint_is_safe({"workersMin": 1, "executionTimeout": 900})
    assert any("bill continuously" in p for p in problems)


def test_an_endpoint_with_no_execution_timeout_is_refused():
    problems = assert_endpoint_is_safe({"workersMin": 0})
    assert any("hung job" in p for p in problems)


def test_a_long_idle_timeout_is_flagged():
    problems = assert_endpoint_is_safe(
        {"workersMin": 0, "executionTimeout": 900, "idleTimeout": 300}
    )
    assert any("idleTimeout" in p for p in problems)


def test_too_many_workers_is_flagged():
    problems = assert_endpoint_is_safe(
        {"workersMin": 0, "executionTimeout": 900, "workersMax": DEFAULT_MAX_WORKERS + 5}
    )
    assert any("bill at once" in p for p in problems)


def test_a_correctly_configured_endpoint_passes():
    assert assert_endpoint_is_safe({
        "workersMin": 0, "workersMax": 1, "idleTimeout": 5, "executionTimeout": 900,
    }) == []


def test_configuration_is_read_back_not_trusted():
    """The previous design's ceiling was also 'configured' and never ran.

    assert_endpoint_is_safe takes what the platform reports, not what was sent.
    """
    import inspect

    from avatar.gpu import serverless

    source = inspect.getsource(serverless.assert_endpoint_is_safe)
    assert "config.get" in source


def test_an_unknown_state_is_treated_as_still_running():
    """Assuming completion would stop us watching something that still bills."""
    import inspect

    from avatar.gpu import serverless

    source = inspect.getsource(serverless.ServerlessClient.status)
    assert "IN_PROGRESS" in source
    assert "unknown job state" in source


def test_there_is_no_lifecycle_to_leak():
    """The structural claim, asserted so a Pod-shaped API cannot creep back in.

    A create/terminate pair is what made the previous design unsafe: it can be
    half-completed. Serverless has no such pair.
    """
    from avatar.gpu import serverless

    for forbidden in ("create_pod", "terminate", "sweep", "rented_gpu"):
        assert not hasattr(serverless, forbidden), (
            f"{forbidden} reintroduces a lifecycle that can leak"
        )
