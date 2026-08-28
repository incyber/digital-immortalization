"""Renting a GPU without leaking one.

RunPod does not stop idle pods. Every test here exists because a pod that
outlives the job that needed it bills the full hourly rate until somebody
notices, and the usual cause is a process dying between create and terminate.
"""

import pytest

from avatar.gpu.runpod import POD_TAG, GpuError, Pod, RunPodClient, rented_gpu


class FakeClient(RunPodClient):
    """Records what would have been asked of the API."""

    def __init__(self, balance=25.0, pods=None, fail_terminate=False):
        self._balance = balance
        self._pods = pods or []
        self.created: list[str] = []
        self.terminated: list[str] = []
        self._fail_terminate = fail_terminate

    def balance(self):
        return self._balance

    def list_pods(self):
        return self._pods

    def create(self, *, image, gpu_type, name, ports, cost_per_hour):
        import time

        self.created.append(name)
        return Pod(
            id=f"pod-{len(self.created)}", name=name, gpu=gpu_type,
            cost_per_hour=cost_per_hour, started_at=time.monotonic(),
        )

    def terminate(self, pod_id):
        if self._fail_terminate:
            raise GpuError("terminate failed")
        self.terminated.append(pod_id)


def test_an_api_key_is_required():
    with pytest.raises(GpuError, match="API key"):
        RunPodClient("")


def test_a_pod_is_terminated_on_success():
    client = FakeClient()
    with rented_gpu(client, image="img") as pod:
        assert pod.id == "pod-1"
    assert client.terminated == ["pod-1"]


def test_a_pod_is_terminated_when_the_job_raises():
    """The common way a pod leaks: the work fails and nobody cleans up."""
    client = FakeClient()
    with pytest.raises(ValueError), rented_gpu(client, image="img"):
        raise ValueError("the job failed")
    assert client.terminated == ["pod-1"], "a failed job must still return the GPU"


def test_a_pod_is_terminated_on_keyboard_interrupt():
    client = FakeClient()
    with pytest.raises(KeyboardInterrupt), rented_gpu(client, image="img"):
        raise KeyboardInterrupt
    assert client.terminated == ["pod-1"]


def test_nothing_is_created_when_the_balance_is_too_low():
    """Failing before creating is cheaper than failing halfway through."""
    client = FakeClient(balance=0.10)
    with pytest.raises(GpuError, match="below the"), rented_gpu(client, image="img"):
        pass
    assert client.created == []
    assert client.terminated == []


def test_an_unreadable_balance_does_not_block_a_run():
    # balance() returns -1 when the endpoint cannot be read. Refusing to run
    # because a billing endpoint was down would be worse than running.
    client = FakeClient(balance=-1.0)
    with rented_gpu(client, image="img"):
        pass
    assert client.terminated == ["pod-1"]


def test_a_failed_teardown_is_still_reported_not_swallowed():
    client = FakeClient(fail_terminate=True)
    with pytest.raises(GpuError, match="terminate failed"), rented_gpu(client, image="img"):
        pass


def test_every_pod_is_tagged_so_it_can_be_swept():
    client = FakeClient()
    with rented_gpu(client, image="img"):
        pass
    assert client.created[0].startswith(POD_TAG)


def test_the_sweep_kills_this_projects_pods():
    client = FakeClient(pods=[
        {"id": "ours-1", "name": f"{POD_TAG}-123", "env": {"AVATAR_TAG": POD_TAG}},
        {"id": "ours-2", "name": "something-else", "env": {"AVATAR_TAG": POD_TAG}},
    ])
    assert set(RunPodClient.sweep(client)) == {"ours-1", "ours-2"}


def test_the_sweep_leaves_unrelated_pods_alone():
    """Somebody else's work on the same account must survive a sweep."""
    client = FakeClient(pods=[
        {"id": "theirs", "name": "training-run", "env": {"PROJECT": "other"}},
        {"id": "ours", "name": f"{POD_TAG}-1", "env": {"AVATAR_TAG": POD_TAG}},
    ])
    assert RunPodClient.sweep(client) == ["ours"]


def test_cost_is_reported_from_actual_runtime():
    import time

    pod = Pod(id="p", name="n", gpu="L4", cost_per_hour=0.39,
              started_at=time.monotonic() - 1800)  # half an hour ago
    assert 0.18 < pod.cost_so_far < 0.21
    assert 29 < pod.minutes_running < 31


def test_a_fresh_pod_has_cost_nothing_yet():
    import time

    pod = Pod(id="p", name="n", gpu="L4", cost_per_hour=0.39, started_at=time.monotonic())
    assert pod.cost_so_far < 0.001
