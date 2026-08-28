# Not losing money on rented GPUs

RunPod bills per second and does **not** stop idle pods. A forgotten pod costs
about $281 a month for doing nothing. This is what stands between that and you.

## The layers, weakest first

Deliberately ordered this way: the layers I wrote are the weak ones, and they
are not what the guarantee rests on.

| # | Layer | Fails when |
|---|---|---|
| 1 | `finally` teardown in `rented_gpu` | the process is `SIGKILL`ed, the machine sleeps or dies |
| 2 | `make gpu-stop`, tagged sweep | nobody runs it |
| 3 | **In-pod deadman** — hard and idle ceilings | the pod itself is destroyed *(then there is no bill)* |
| 4 | **Prepaid balance** | never — RunPod enforces it server-side |

Layer 3 is the first that does not depend on anything outside the pod.
Layer 4 is the only one that is a guarantee, and it is not code.

## Why layer 3 matters

Layers 1 and 2 share an assumption: something outside the pod is still alive
and willing to stop it. That assumption is false exactly when it matters -
the orchestrator crashed, the laptop slept, the network dropped.

The deadman runs *inside* the pod, so its liveness is the pod's liveness. Two
independent triggers:

- **Hard ceiling** (default 30 min) - fires regardless of what the job is
  doing. A job that outruns it has hung.
- **Idle ceiling** (default 10 min) - the work touches a heartbeat file; if it
  stops, the work is gone and the GPU is rented for nothing.

If it cannot reach the API it stops the container anyway. A stopped container
is not a stopped pod, but it is a smaller bill and it makes the leak visible.

## What actually caps the loss

**RunPod is prepaid.** With auto-pay off, the balance is the maximum possible
loss, enforced by RunPod rather than by anything here. Load $10 and the worst
case - every layer above failing at once - is $10.

**Keep auto-pay off.** It reloads the balance automatically and removes the
only real guarantee on this page.

## Operating rules

1. Load $10. Not more.
2. Auto-pay **off**. Low-balance email alert on.
3. `make gpu-status` shows what is billing right now.
4. `make gpu-stop` terminates everything this project started, and is safe to
   run at any time, including when nothing is running.

## Honest limits

- A pod started outside this code is not tagged and will not be swept.
- If RunPod's API is down, nothing here can terminate anything; the deadman
  falls back to stopping the container.
- None of this protects against a mistake in a `gpu_type` that costs more per
  hour than expected. The balance still does.
