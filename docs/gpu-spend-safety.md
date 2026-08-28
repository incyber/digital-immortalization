# Not losing money on rented GPUs

## The short version

GPU work runs on **RunPod Serverless**, where nothing is allocated between
jobs. There is no pod to forget, so the entire class of "left running
overnight" does not apply.

This replaced a Pod design that could not be made safe. An adversarial review
found that two of its three claimed layers did not function at all.

## What was wrong before, stated plainly

The previous design was described in this file as layered protection with an
in-pod deadman as "the first layer that does not depend on anything outside
the pod." Verified against the code:

| Claimed layer | Reality |
|---|---|
| `finally` teardown | Works, but only when the process survives. Fail-open. |
| 30-minute wall-clock ceiling | **Never enforced.** `max_minutes` is read inside the `finally`, after the block returns. It logs a warning about a pod it has already released. |
| In-pod deadman | **Never deployed.** No Dockerfile or compose file references `infra/deadman.sh`. It had a passing unit test and had never run in a pod. |
| Prepaid balance | Real, and provider-enforced. |

Two of the three were theatre. The ceiling could not stop a hang, and the
deadman did not exist anywhere a pod would find it.

The deeper problem was shape, not bugs. A Pod runs until something acts to
stop it, so **every** failure - crashed orchestrator, slept laptop, dropped
network, `SIGKILL` - leaves a GPU billing. Layering more watchdogs onto a
fail-open design produces more things that can fail.

## Why serverless removes the problem instead of guarding it

Three guarantees, all enforced by RunPod, none dependent on this code being
correct, running, or alive:

| Setting | Value | What RunPod's documentation says |
|---|---|---|
| `workersMin` | `0` | Active workers "incur charges continuously, including when idle". Zero means none are active. |
| `idleTimeout` | `5s` (default) | A worker shuts down this long after finishing a request. |
| `executionTimeout` | `900s` | "When exceeded, the job fails and the worker stops." |
| `workersMax` | `1` | "Acts as a cost safety limit and concurrency cap." |

A hung job is killed by the platform. A crashed orchestrator leaves nothing
running, because nothing was allocated in the first place. There is no
create/terminate pair to half-complete.

`assert_endpoint_is_safe()` reads these back from the platform rather than
trusting what was sent at creation - the previous ceiling was also
"configured", and never ran.

## What still requires a human

1. **A dedicated RunPod account**, used for nothing else. Then everything is
   scoped by account rather than by tags that can be missed.
2. **Fund as little as the work needs.** The balance is not a loss cap, it is a
   *leak timer*: `balance / hourly rate = worst-case leak duration`. At the
   verified L4 rate of $0.44/hr, $5 is 11.4 hours and $20 is 45 hours. The
   account is funded at **$20**, so the timer is 45 hours rather than 11.
3. **Low-balance alert just below the funded amount** - at $20 funded, alert at
   $18. This turns the balance from a cap that fires after the money is gone
   into a detector that fires after about $2. RunPod's default threshold is $5,
   which on a $20 balance fires only after $15 is already spent.
4. **Auto-pay off, no saved card.** It reloads automatically and removes the
   only guarantee on this page that does not depend on code.
5. **Never use GraphQL `stopAfter` / `terminateAfter`.** They accept input,
   return success, and do nothing. RunPod's own CLI removed the flags on
   2026-08-27 for this reason, and their documentation still lists them.

## Standing checks

Two commands, neither of which trusts the application's own record.

    make endpoint-verify    re-read the four settings from the platform
    make gpu-status         what exists on the account right now

`endpoint-verify` exits non-zero when any of `workersMin`, `workersMax`,
`idleTimeout` or `executionTimeoutMs` is wrong. Worth running on its own rather
than only at creation: provisioning checks the settings once, but a console
click can change `workersMin` afterwards and nothing in the application would
notice.

`gpu-status` lists serverless endpoints and flags any with `workersMin` above
zero, which is the only setting that bills with no job running. It also reports
pods - and since this project uses serverless, **a pod appearing there is a
finding, not a reading**.

Account state at the time of writing: balance $20.00, zero endpoints, zero
pods, $0.00/hr.

One correction worth recording. `balance()` previously called
`GET /v1/billing/balance`, which returns 400 - *"that path ... does not
exist"*. The number the whole spend argument rests on was never being read. It
now goes through GraphQL `myself { clientBalance }`.

## Verifying it yourself

Cheap, and worth doing before trusting any of the above.

- **V1, free.** Submit a job to an endpoint configured with
  `executionTimeout` of 60s that sleeps for 300s. Observe the platform mark it
  `TIMED_OUT` and the worker stop. Nothing in this repository participates.
- **V2, free.** Submit a job, then `kill -9` the orchestrator. Observe that the
  job still completes or times out on its own, and that no worker remains
  afterwards. Under the Pod design this was the scenario that leaked.
- **V3, free.** Set the low-balance threshold just under the current balance
  and confirm the email reaches your phone. An untested alert is not an alert.
- **V4, free.** Leave the endpoint idle overnight. Confirm the next day's
  balance is unchanged.

## Residual exposure

| Failure | Bounded by | Cost |
|---|---|---|
| Orchestrator dies mid-job | job runs to completion or `executionTimeout` | under $0.11 |
| Job hangs | platform `executionTimeout` | $0.11 for 15 min |
| Endpoint misconfigured with warm workers | `assert_endpoint_is_safe` at creation deletes it; `make endpoint-verify` catches a later change; then the balance | up to the balance |
| Auto-pay switched on | nothing | unbounded - do not do this |

## What is not covered

A live call that must hold warm models across a multi-minute conversation
cannot be request/response, and will need a Pod. That decision should be made
when that renderer exists, with the pull-lease design below - not inherited by
default.

**The lease design, for when it is needed.** Invert the direction of authority.
Today an orchestrator *pushes* a kill, so every delivery failure means the pod
survives. Instead the pod should *pull* a renewable permission: a key in a
store with a storage-enforced TTL, written by the orchestrator every 15
seconds, read by an in-pod supervisor every 15 seconds. Missing, expired,
unreadable, or a non-200 all count as denial. Three consecutive failures and
the pod terminates itself. The supervisor must be PID 1 under
`timeout -s KILL 1800`, so that its own death is container exit rather than a
crashed watchdog. Renewal must come from outside the pod: a heartbeat written
by the worker is fail-open, because a hung worker keeps writing it.

One window remains genuinely fail-open even then - between pod creation and
the supervisor arming, nothing is enforcing anything. It is closed only by
pinning images by digest so a bad image fails before creation, and bounded by
the balance.

## Honest limits

- Revoking or rotating the API key disables the `finally`, the CLI, and any
  in-pod termination **simultaneously**. It is a genuine common cause and no
  layering removes it. Serverless is unaffected: there is nothing to terminate.
- A pod created by hand is not tagged and will not be swept. A dedicated
  account is the fix, not better tags.
- RunPod's auto-stop at $0 has no documented grace period. Assume the balance
  can go slightly negative.
