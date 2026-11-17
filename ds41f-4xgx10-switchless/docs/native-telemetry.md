# Native telemetry: as-built contract and qualification gates

Status: **implemented r3 trial feature; NOT GPU-qualified; NOT deployed**.
Trial tag: `harbor.lab.tf/library/pico-dsgx10:2026.11.16-native-telemetry-trial-r3`.
This revision closes teacher-forced slot accounting and aligns the HELP unit
with the contract: entries, not rows. The immutable image digest and CPU test
receipts are recorded outside source with the r3 build artifacts.
Baseline: `b7c3eb5b12a030725bfb0186d27065bc6e376685` (native vision).
No sampling, steering, KV precision, request budget, scheduling or wire-protocol
change is part of this feature. The existing GLP-39 and model settings remain
unchanged. Instrumentation changes executable identity; retained session files
must not be assumed compatible after a restart.

## Why native telemetry

The previous dashboard data path is completed-request summaries -> a host log
collector -> node_exporter textfile -> Prometheus. Summaries older than 120
seconds become unknown even if tokens are still decoding. The native feature
replaces that *source contract*, not the host/GPU/fabric exporters. There was no
native metrics handler in the baseline to increase a timeout on. Monitoring and
dashboard migration is a separate change; this build does not rewire them.

## Transport and lifecycle

* Each rank: `dsgb10_rank --metrics-listen 127.0.0.1:9108
  --metrics-model-name deepseek-v4.1-flash [--metrics-cpu N]`.
* Frontend: `dsfront serve --metrics-listen 127.0.0.1:9109` plus the existing
  serving arguments; the configured `--model-name` labels its metrics.
* Both listeners are opt-in in the executables (rank: IPv4; frontend: IPv4/IPv6), explicit-address only, with no
  wildcard bind. The trial image enables loopback 9108 / 9109. A separately
  approved private monitoring route is needed for a remote scraper. They expose
  `GET /metrics` (Prometheus text 0.0.4) and `GET /live` (endpoint thread only).
  There is no authentication; never expose them on a public interface.
* C++ starts its OS thread before CUDA/model initialization and before the
  follower branch. Rust starts its OS thread before loading the tokenizer and
  creating the Tokio serving runtime. Neither endpoint runs in the inference
  loop, the request-serving runtime, or the engine socket pool.
* The rank thread polls bounded nonblocking sockets every 20 ms; the frontend
  thread visits them every 5 ms. Four clients maximum, 4096-byte headers,
  256-KiB response maximum, one-second absolute connection deadline, connection
  close after the response, bounded accepts per loop. Slow clients do not block
  other accepted clients. Four malicious clients can exhaust this bounded
  endpoint briefly; admission then drops scrapes rather than growing memory.
* Threads choose an allowed housekeeping CPU before inference pins itself,
  excluding known engine/I/O/communication/Engram cores; the rank permits an
  explicit CPU override. This selection still needs host/cgroup qualification.
  Neither uses real-time priority. It does not reserve a CPU against other
  processes. OS starvation, host/process death or network failure can still
  prevent a scrape; external `up` and host metrics remain essential.
* Heartbeats describe the endpoint poll loop, not inference readiness. Progress
  timestamps change at existing owner boundaries; stalled inference does not
  get artificial progress. Empty latency histograms remain empty at idle.

## Accounting and concurrency

`engine/common/telemetry.h` contains fixed numeric storage shared by model and
server. `uint64_t` atomics must be lock-free at compile time (also checked by the
ARM64 build). Writers use relaxed atomic increments/stores, fixed histogram
bins, and existing clocks. Registry storage is statically below 1 MiB. There is
no inference mutex, GPU/CUDA call, collective, log, filesystem read, session-table
traversal or callback to serving in either scrape path.

The scheduler owns each request's non-atomic lifecycle tracker. Receipt increments
waiting once; initial admission moves it to running once; first emitted token
observes TTFT once; terminal accounting is idempotent, including pre-admission
cancellation and validation errors. Resume/slicing is not a new request. Accepted
output tokens are counted at the enqueue attempt (even if the socket is already dead), not when a long batch finally returns
DONE. Rejected speculative drafts and queued-ahead dummy outputs do not count.
Logical request/token metrics exist **only on rank zero**. Do not sum TP replicas.

Model snapshot usage mirrors scalar store fields at existing owning mutations,
including spill/unspill and eviction within those operations. Reads by the metrics
thread never touch the store. Bytes are logical allocated blocks, not an I/O
completion guarantee. Frontend gate, socket and retention mirrors update while
their owners already hold their existing locks; the metrics thread acquires none.

Histograms have second boundaries
`0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,30,60,120,300,600,+Inf`.
Writers increment noncumulative bins; the renderer computes cumulative buckets and
count from the same bin snapshot. Sum and different fields may have small
cross-field skew during concurrent updates. Scrapes never retry a writer seqlock.

## Metric contract

Every sample has bounded `model_name`. Every rank endpoint sample also has `rank`.
No request/session/user IDs, paths, prompts, tokens, arbitrary errors or revision
strings are labels. All names below have the `pico:` prefix. Types `counter` and
`histogram` retain their usual Prometheus reset and `_bucket/_count/_sum` semantics.

### Standard-aligned rank-zero metrics

| Name | Type / extra labels | Meaning and boundary |
|---|---|---|
| `num_requests_running` | gauge | Admitted core requests until DONE, including time-sliced requests and finished rows awaiting batch acknowledgement. Not a count of live GPU rows. |
| `num_requests_waiting` | gauge | Valid socket submits not admitted: socket, session-serialization and slot waits. Disconnected unadmitted requests become terminal cancellations when consumed. |
| `time_to_first_token_seconds` | histogram | Socket receipt to enqueue of first accepted token, including core queue time. Not first visible HTTP/SSE bytes. |
| `request_time_per_output_token_seconds` | histogram | `(DONE time - first token time)/(number of outputs - 1)` for requests with at least two accepted outputs; includes pauses and acknowledgement delay. Not an inter-token-gap histogram. |
| `request_queue_time_seconds` | histogram | Receipt to initial successful admission; no second observation for resumes. Invalid pre-admission requests have no successful admission observation. |
| `prompt_tokens_total` | counter | Initially planned logical prompt positions, including reused positions; not resume/recomputation. |
| `generation_tokens_total` | counter | Accepted core outputs at the frontend enqueue attempt, including first and stop tokens; excludes rejected speculative drafts and ahead-only samples. May exceed visible HTTP output after filtering or disconnect. |
| `request_success_total` | counter / `finished_reason=stop,length` | Successful core terminal requests. Context exhaustion, error and cancellation are not reported as successes. |
| `prefix_cache_queries_total` | counter | Prompt positions considered once per initial exact reuse plan. Token-weighted queries, not number of requests. |
| `prefix_cache_hits_total` | counter | Prompt positions reused by that plan, including snapshot/recomputed restart boundaries as selected by Pico. Not GPU allocation bytes. |

### Native rank additions (per rank unless marked leader)

| Name | Type / extra labels | Meaning |
|---|---|---|
| `metrics_heartbeat_timestamp_seconds` | gauge | Endpoint loop wall timestamp, independent of inference. |
| `process_start_time_seconds` | gauge | Timestamp when native telemetry was initialized near process entry. |
| `initialization_state` | gauge | 0 loading, 1 rank ready after startup barrier, 2 stopping. Read with heartbeat and external health, not alone. |
| `inference_progress_timestamp_seconds` | gauge / `phase=scheduler,prefill,decode` (separate values) | Last owner progress boundary. Zero before the phase first runs; idle progress need not advance. Followers update scheduler after receiving a command, not while blocked waiting for one. |
| `slots_active` | gauge | Entries participating in decode (including teacher forcing and draining rows); zero outside decode. Teacher-forced steps count entries with rows in that step, not the total number of rows. Not a prefill-occupancy metric. |
| `slots_capacity` | gauge | Configured logical slots. |
| `context_capacity_tokens` | gauge | Sum of configured slot context limits, not physical cache bytes. |
| `kv_cache_usage_available` | gauge | **0**: a vLLM-style allocated-block usage fraction has no exact mapping implemented. |
| `session_usage_bytes` | gauge / `tier=memory,ssd` | Allocated snapshot blocks in the device-memory store and SSD store respectively; not active KV. |
| `session_capacity_bytes` | gauge / `tier=memory,ssd` | Reserved capacity of those stores. Zero if disabled; not free GPU/SSD space. |
| `sessions` | gauge | Snapshot record count across both stores, including empty records; not active requests. |
| `session_observed_timestamp_seconds` | gauge | Last owner publication of store scalars. |
| `requests_received_total` | counter, leader | Valid parsed SUBMIT frames. Malformed wire frames do not become requests. |
| `requests_finished_total` | counter / `finished_reason=stop,length,context,cancelled,error,unknown`, leader | All terminal core request outcomes. |
| `prefill_tokens_total` | counter, leader | Completed logical prompt chunk positions, including resume recomputation, excluding reused positions. `rate()` is wall throughput, not GPU-kernel processing speed. |

### Frontend additions (9109 only)

These are deliberately `pico:frontend_*`, not copies of the rank-zero metrics.
They must not be summed with engine accounting.

| Name after `frontend_` | Type / extra labels | Meaning |
|---|---|---|
| `metrics_heartbeat_timestamp_seconds`, `process_start_time_seconds` | gauges | Endpoint poll heartbeat and telemetry initialization wall timestamp. |
| `initialization_state` | gauge | 0 initializing, 1 HTTP listeners bound, 2 stopping. |
| `engine_connected` | gauge | Engine socket handshake established; does not prove inference progress. |
| `admission_open` | gauge | Gate neither held nor closing; connectivity is reported separately. |
| `requests_running`, `requests_waiting` | gauges | Gate passes held (including stream lifetime), and gate waiters; Drop paths decrement them. |
| `engine_pending` | gauge | Socket submits awaiting terminal event. |
| `sessions`, `session_retained_tokens`, `session_token_budget` | gauges | Retained conversation entries, actual retained token IDs, configured token budget; never active KV. Busy histories may temporarily exceed the retention budget. |
| `requests_received_total` | counter | Chat handler entries, before authentication/JSON validation. HTTP body/extractor rejections before the handler are outside this count. |
| `http_responses_total` | counter / `status_class=1xx..5xx` | Chat response headers returned by handler. An SSE error may occur after successful 2xx headers. Not a successful-completions counter. |
| `generation_tokens_total` | counter | Engine tokens received on the socket, before stop/parser filtering. |
| `engine_finished_total` | counter / `finished_reason=stop,length,context,cancelled,error,unknown` | Terminal engine events including synthesized disconnect errors. |
| `engine_connection_errors_total` | counter | Failed connects or closed/lost connections. |
| `request_queue_time_seconds` | histogram | Admission enter to pass for successful gate admissions. |
| `time_to_first_engine_token_seconds` | histogram | Gen start to first engine token event, excluding gate wait and prior HTTP preparation. Not full end-to-end TTFT or first visible content. |

## Alignment and deliberate deviations

Reference conventions:
[vLLM production metrics](https://docs.vllm.ai/en/latest/design/metrics/),
[SGLang production metrics](https://docs.sglang.ai/references/production_metrics.html).
This is semantic alignment, not a claim that unmodified vLLM dashboards work.

* `pico:kv_cache_usage_perc` is intentionally **absent**, not 0 or a made-up
  fraction. Pico has preallocated mixed-capacity sliding/compressed slot windows,
  not a pageable vLLM block pool. Exact active-byte accounting is not implemented.
  Session snapshots, context-token limits and GPU memory usage are not substitutes.
  This metric is held at the escalation boundary; no extra inference locks or GPU
  queries were added to implement it. Consumers must show N/A when availability=0.
* Running means admitted core requests, not live batch rows; `slots_active` exposes
  that distinction. Prompt reuse is exact restart-position reuse, not a vLLM
  prefix-block cache. Query/hit counters are token weighted.
* TTFT begins at the core socket boundary. Frontend timing is separately named
  because preprocessing, vision work, admission wait and SSE keep-alives differ.
* No duplicate host/GPU/fabric metrics. No invented model-forward/execute timers,
  preemption counts, speculative accept ratios or session IO latency: those are
  not instrumented by this version. Decode progress covers existing sample waits;
  prefill progress uses existing chunk boundaries without another GPU wait.
* Per-tier session counts are not exported: only the exact total record count and
  actual bytes per tier are available without another container traversal.
* Error outcomes have fixed labels. Arbitrary HTTP error strings do not enter
  metrics. Header status counters are not a substitute for stream terminal events.

## Source map

* `engine/common/telemetry.h`: bounded numeric sink and CPU-testable lifecycle.
* `engine/server/metrics.{h,cpp}`: independent listener/renderer/affinity/lifecycle.
* `engine/server/main.cpp`, `CMakeLists.txt`: startup, CLI and linking.
* `engine/server/serve.cpp`: receipt/admission, first token, accepted outputs,
  initial reuse plan, prefill progress, exactly-once terminal accounting.
* `engine/server/cmd.cpp`: existing completed decode boundaries and live entries,
  including speculative paths and followers; no math or scheduling changes.
* `engine/model/state.cu`: O(1) owner snapshots after existing store mutations.
* `frontend/src/metrics.rs`, `lib.rs`, `main.rs`: numeric registry, independent
  thread, options and lifecycle. `server.rs`, `engine.rs`, `gate.rs`, `session.rs`
  publish HTTP/socket/gate/retention facts at their current owning transitions.

## Dev tests and image build

CPU-only contract and independent-endpoint tests:

```sh
cd ds41f-4xgx10-switchless
out=$(mktemp -d)
g++ -std=c++20 -O2 -Wall -Wextra -Werror -pthread -I engine \
  engine/server/metrics.cpp engine/server/tests/metrics_test.cpp -o "$out/metrics_test"
"$out/metrics_test"
python3 engine/server/tests/metrics_teacher_forcing_test.py --out "$out/teacher-forcing"
# Python environment needs prometheus_client; it is an independent text parser.
python3 engine/server/tests/metrics_endpoint_test.py "$out/metrics_test" --seconds 130
CARGO_TARGET_DIR="$out/rust" cargo test --locked --manifest-path frontend/Cargo.toml
CARGO_TARGET_DIR="$out/rust" cargo build --locked --manifest-path frontend/Cargo.toml --example metrics_fixture
python3 engine/server/tests/metrics_endpoint_test.py "$out/rust/debug/examples/metrics_fixture" --frontend --seconds 130
```

The teacher-forcing CPU test compiles the unchanged `batch_tf_loop` body from
`cmd.cpp` against a model double: unequal lengths and zero-length entries must
move `slots_active` through 3, 2, 1, then 0, including unwind cleanup. It checks
the rendered metric and its HELP text. This tests accounting control flow, not
GPU execution or token parity.

The long CPU fixture simulates 16 admitted requests and queued/error outcomes
while deliberately stopping its producer for 125 seconds. This proves listener
independence and exposition accounting only, **not GPU saturation or serving
regression**. ARM64 compilation and QEMU CPU-only tests do not replace GPU parity.
Build instructions live in `tools/native-metrics/README.md`.

## Qualification and rollout gates

The trial remains **NOT qualified** until all gates below pass on a non-serving
TP4 target or an explicitly authorized maintenance window. There is no spare TP4
target established by this work. No production load, restart or swap is authorized.

1. Fresh-context review of the engine diff against the brief and this contract;
   fix findings within three review rounds. A review is not runtime acceptance.
2. CPU tests, malformed/slow/aborted scrape clients, parser validation, concurrent
   atomics, sanitizer tests and ARM64 lock-free compilation. Keep raw evidence
   outside tracked source; do not put intermediate logs in this repository.
3. One OCI artifact: pinned base/checkpoint lineage plus engine and frontend
   binary hashes in the profile and source revision in labels. Build off Sparks,
   push a clear **unqualified telemetry trial** tag to the private registry and
   record the registry digest. No loose-binary production copy.
4. On the trial target, preserve baseline image/argv, clocks, session and SSD
   volumes. Current `--keep-gib 0` does not preserve sessions across restart.
   Establish tested export/restore or explicit session-loss authorization first.
5. Matched baseline and candidate GPU token-for-token parity with identical
   checkpoint, prompts, sampling, steering, precision and budgets. No qualification
   based on source inspection alone.
6. Simultaneous 1-Hz scrapes for at least 30 minutes idle and 30 minutes with all
   16 slots occupied, requests over 120 seconds, long prefill, queueing, validation,
   cancellation and error cases. Require 100% controlled scrape success, p99 below
   100 ms, max below 1 s, heartbeat age below 3 s. Reconcile values against a known
   client load, not logs. Verify real cgroup/CPU affinity and resource headroom.
7. Repeated standard decode battery versus the pinned production image under
   matched idle stack and clock conditions: <1% aggregate throughput regression
   and <1% p95 per-token latency regression, with noise bounds reported.
8. Production deployment, monitoring targets and dashboard rewiring each require
   separate authorization. Roll back through the same coordinated procedure to
   the recorded old OCI digest and exact argv; verify a real pinned completion
   and backend health. An image rollback alone cannot recover lost sessions.
