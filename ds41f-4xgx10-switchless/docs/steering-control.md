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
  disagrees, every rank is put back on the value it had and the answer says so.
- **Graph-safe.** Decode steps are captured as CUDA graphs keyed by shape, and the speculative step
  graphs are captured the same way; both run the layer loop. Two things in that loop are host
  decisions taken when a graph is captured, not device reads: whether the steering branch runs, and
  whether the pre-write copy it needs is made. A set therefore drops both graph caches (the decode
  graphs and the spec graphs), and the next step of each shape is captured again under the new alpha —
  including the crossings 0 to nonzero and back.
- **Cache-safe.** Retained KV is tagged with the configuration it was computed under: the RAM prefix
  cache and the SSD session tier carry the configuration id in their headers, and a mismatched index
  is refused rather than restored. A set starts a new **epoch**, and drops the store's sessions, so a
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

A set answers the same shape with the new epoch, or an error and the unchanged state:

```
error: rank 2: alpha is outside [0, 2] alpha=0.75 epoch=0 ranks=[0:refused(...) 1:refused(...) ...]
```

Field by field: `steering` is whether a direction file is loaded at all, `alpha` the effective scalar,
`epoch` the configuration epoch (0 until the first applied set), and one `rank:alpha@epoch` entry per
rank from that rank's own read-back of the device.

## Rules, and why

- **Accepted range is 0.0 to 2.0 inclusive** (`model/steer.h`, `kSteerAlphaMin`/`kSteerAlphaMax`).
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
  makes the rollback path safe (it restores a known value and skips the comparison,
  `kSteerAnyEpoch`).

## What a set does, step by step

1. `dsfront admin … steer ALPHA` → the front-end stops admitting requests and waits for those in
   flight to finish (bounded, 120 s; on timeout it refuses and admits again).
2. The front-end sends `STEER_SET` on the rank-0 socket. Rank 0 refuses it if anything is still
   queued or running, otherwise it broadcasts the value to the four ranks.
3. Each rank, between steps, writes the alpha to the device, reads it back, compares, updates the host
   copy that the layer boundary reads, moves its epoch, and drops its captured step graphs.
4. The four confirmations are exchanged and compared. Rank 0 drops retained state (the store's
   session handles; the front-end keeps the text and re-prefills on the next turn).
5. Rank 0 answers with the effective state and every rank's read-back; the front-end admits requests
   again.

A no-op set (`steer 0.75` when alpha is already 0.75) moves nothing: no epoch, no dropped state, no
dropped graphs.

## Protocol

`STEER_SET` (`0x05`) and `STEER_GET` (`0x06`) are answered by `STEER_STATE` (`0x84`). The message
layouts and their sizes are in [protocol.md](protocol.md) and in
[`engine/common/frontend_protocol.h`](../engine/common/frontend_protocol.h); the protocol version is 3.

## What is verified, and what is not

Verified on CPU, no GPU needed:

- `engine/model/tests/steer_test.cpp` (build target `steer_test`): the alpha parse and its refusals,
  the accepted range, `steer_check`, configuration identity (alpha, layers, direction file, on/off
  each change it), epochs (a no-op moves nothing; 0.75 → 1.0 → 0.75 lands at epoch 2, not 0), and a
  refusal when steering is off.
- `frontend/src/protocol.rs` tests: the three frames round-trip, the byte sizes are exact, and a
  non-zero reserved field is rejected.
- `frontend/src/gate.rs` test `quiesce_drains_and_admits_again`: a quiesce waits for requests in
  flight, holds new ones off, admits again when dropped, refuses on timeout, and refuses a second
  concurrent change.
- `cargo test --locked` in `frontend/` passes 40/40.

Only a Spark window can verify, and none of these has been run yet:

- GPU parity: with no `steer` command, output is identical to the previous build on the fixed check
  set.
- A live set under load: the transition lands on all four ranks, in-flight requests finish under the
  alpha they started with, and the four read-backs agree.
- Set-to-same-value is a no-op on device.
- Speed cost of the change (expected to be zero: the set path runs between steps, never on them).
