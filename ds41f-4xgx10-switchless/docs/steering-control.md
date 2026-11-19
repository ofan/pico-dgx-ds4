# GLP steering: the runtime alpha

GLP steering subtracts a fixed direction from the residual stream at the end of chosen layers
(`model/glp.cpp`, `model/glp.cu`). Three things decide what it does: whether a direction file was
loaded at all (`WEIGHTLESS_STEER_PATH`), which layers are steered (`WEIGHTLESS_STEER_LAYERS`), and one
scalar, **alpha**, that scales every direction.

Until this change alpha was read once at startup: `glp_from_env()` takes `WEIGHTLESS_STEER_ALPHA` if
it is set, otherwise the direction file's `glp.alpha_default`, and `model.cu` copies it to the device
once. Changing it meant recreating the containers and restarting all four ranks, which loses every
session. This page is the operator path that replaces that restart with one command.

## What it is

- **One alpha for all traffic.** The engine holds one effective configuration (steering on or off,
  alpha, the steered layers, the direction file's identity). A set applies to every request that
  starts after it; requests already in flight keep the value they started with, and the operator's
  front-end does not admit new ones while the change lands.
- **All four ranks or none.** Rank 0 broadcasts the set, each rank applies it at the same step
  boundary and answers with what it holds, and the four answers are compared. If any rank refuses or
  disagrees, every rank is forcibly restored to the pre-transition alpha at one explicit common
  recovery epoch newer than every observed epoch. An unconfirmed recovery keeps both engine and
  frontend admission closed. The operator retries `steer` (without an alpha) to complete recovery;
  a new set is not applied while recovery is pending.
- **Graph-safe.** Decode steps are captured as CUDA graphs keyed by shape, and the speculative step
  graphs are captured the same way; both run the layer loop. Two things in that loop are host
  decisions taken when a graph is captured, not device reads: whether the steering branch runs, and
  whether the pre-write copy it needs is made. Before any device write (including recovery or a failed set), the engine drops both graph caches (the decode
  graphs and the spec graphs), and the next step of each shape is captured again under the new alpha —
  including the crossings 0 to nonzero and back.
- **Cache-safe.** Retained KV is tagged with the configuration it was computed under: each live RAM/SSD store record and kept record carries on/off, exact alpha bits, layers, direction
  identity and epoch. Handle lookup, acquisition, unspill, load, restore and snapshot copy enforce that
  tag. Kept headers and the server index include both the epoch and a configuration hash that includes
  epoch. Old index formats are rejected, not upgraded or relabeled. An applied transition starts a new **epoch**, and drops the store's sessions, so a
  request that arrives afterwards cannot find KV from another configuration. State from an older
  epoch is never served, continued, forked or restored, even when alpha returns to a value it held
  before. Sessions are not lost: the front-end's session table holds only text and ids, never KV, and
  the engine re-prefills a session whose state it no longer has (its next answer reports
  `cached_tokens = 0`).

## Operator commands

The front-end's admin socket is mode 0600 in a 0700 directory: file access is the authorization, and
there is no network setter. `dsfront admin` writes one command and prints one line back.

```sh
# what is the engine doing now, on every rank?
dsfront admin --socket /run/pico/admin.sock steer

# set one alpha for all traffic (drains, applies on all four ranks, admits again)
dsfront admin --socket /run/pico/admin.sock steer 1.0
```

A read-back answers:

```
ok steering=true alpha=0.75 epoch=0 ranks=[0:0.75@0 1:0.75@0 2:0.75@0 3:0.75@0]
```

A set answers the same shape with the new epoch, or an error. A partial set can recover the old alpha
at a newer epoch; the error does not imply unchanged state:

```
error: alpha outside the finite runtime range [0, 2] alpha=0.75 epoch=0 ranks=[0:0.75@0 1:0.75@0 2:0.75@0 3:0.75@0]
```

Field by field: `steering` is whether a direction file is loaded at all, `alpha` the effective scalar,
`epoch` the configuration epoch (0 until the first applied set), and one `rank:alpha@epoch` entry per
rank from that rank's own read-back of the device.

## Rules, and why

- **Accepted range is 0.0 to 2.0 inclusive** (`model/steer.h`, `kSteerAlphaRange = {0.0f, 2.0f}`).
  GLP v1 subtracts alpha times a unit direction: 0.0 is the identity, 1.0 removes the whole component,
  values above it over-remove, and at 2.0 the removed component is a sign flip rather than a strength.
  Nothing measured exists beyond 1.0, so 2.0 is the documented ceiling, not a recommendation.
  Non-finite, negative, and out-of-range values are refused; so is text that is not a whole number
  (`steer_parse_alpha` rejects leading spaces, trailing text, `nan`, `inf` and overflow).
- **Alpha alone never enables steering.** With no direction file loaded the answer is
  `steering is off`, and the state does not move. Turning steering on stays a startup decision.
- **Never set: no change.** A process that is never given a `steer` command behaves exactly as it did
  before this change: epoch 0, the startup alpha (`glp.alpha_default`, else `WEIGHTLESS_STEER_ALPHA`).
- **Refused while requests are in flight.** Rank 0 refuses a set that arrives with requests queued or
  running, because a request must not change meaning halfway. The front-end drains first
  (`gate::quiesce`), and its own refusal names the number of requests still in flight.
- **Sets do not overlap.** The front-end refuses a second change while one is draining or applying, so
  two operators cannot interleave. The wire protocol carries no epoch: the engine applies at the epoch
  it holds, and reports it. Inside the engine the epoch is still checked on every set, which is what
  detects a stale view. Recovery bypasses that expectation (`kSteerAnyEpoch`) but assigns a
  common explicit target epoch and always writes/verifies the device, never a host-only no-op.

## What a set does, step by step

1. `dsfront admin … steer ALPHA` → the front-end stops admitting requests and waits for those in
   flight to finish (bounded, 120 s). A drain timeout/cancellation before dispatch is harmless and
   releases its hold. After dispatch, a timeout, disconnect or cancellation has an unknown outcome and
   latches admission closed; dropping the guard or issuing `release` cannot clear it.
2. The front-end sends `STEER_SET` on the rank-0 socket. Rank 0 refuses it if anything is still
   queued or running, otherwise it broadcasts the value to the four ranks.
3. Each rank, between steps, invalidates both graph caches **before** writing the alpha. It reads
   the device back and updates the host/epoch to the observed value even on mismatch; a mismatch is
   marked unconfirmed and cannot be mistaken for a safe host-equal no-op on rollback.
4. The coordinator checks all four confirmations (rank identity, alpha, epoch and full configuration
   hash). It drops retained RAM and SSD state after an applied change, and also on every recovery
   attempt, including an unsuccessful one. Recovery forcibly writes the old alpha on **all** ranks
   with the same target epoch. Failed recovery retains that target for the next operator command.
5. Rank 0 answers with the effective state and all four device read-backs. The frontend only reopens
   after a safe result with four agreeing ranks. A subsequent serialized `steer` read establishes the
   final outcome of an earlier timed-out command, or retries pending recovery. GET checks device
   state on every rank; it is not a rank-zero host-only success. A read can initiate recovery when it
   discovers disagreement, so it also drains and holds admission.

A no-op set (`steer 0.75` when alpha is already 0.75) moves nothing: no epoch, no dropped state, no
dropped graphs. It still reads and confirms every rank's device scalar.

## Protocol

`STEER_SET` (`0x05`) and `STEER_GET` (`0x06`) are answered by `STEER_STATE` (`0x84`). The message
layouts and their sizes are in [protocol.md](protocol.md) and in
[`engine/common/frontend_protocol.h`](../engine/common/frontend_protocol.h); the protocol version is 3.

## What is verified, and what is not

Verified on CPU, no GPU needed:

- `engine/model/tests/steer_test.cpp` (build target `steer_test`): strict parse/range, complete
  configuration-plus-epoch hashing, and the actual production four-rank coordinator under partial
  apply failures on each rank, failed recovery, subsequent convergence, safe no-op and invalid sets.
  The shared device-write transaction is failure-injected to prove host updates on read-back mismatch,
  pre-write graph invalidation and forced recovery even when host alpha is already the old value.
  CUDA-free production store records exercise handle/acquisition refusal on each stale configuration
  field and epoch, for RAM, SSD offsets and kept records. These are CPU transport/device stand-ins,
  not CUDA, RDMA or physical SSD integration tests.
- `frontend/src/protocol.rs`: frame layout/round-trip, reserved-field refusal and safe recovery refusal.
- `frontend/src/gate.rs`: drain, concurrent-change refusal (the first future is polled), cancellation
  before dispatch, and unknown-outcome/failed-recovery admission holds. A safe bit alone cannot reopen
  admission with divergent rank confirmations, nor can admin `release` bypass recovery.
- `cargo test --locked --offline --frozen`: 42 tests. Checks do not establish GPU correctness.

Only a Spark window can verify, and none of these has been run yet:

- GPU parity: with no `steer` command, output is identical to the previous build on the fixed check
  set.
- A live set under load: the transition lands on all four ranks, in-flight requests finish under the
  alpha they started with, and the four read-backs agree.
- Set-to-same-value is a no-op on device.
- Speed cost of the change (expected to be zero: the set path runs between steps, never on them).
