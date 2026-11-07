# Runtime GLP steering

Off by default. Set `WEIGHTLESS_STEER_PATH=/path/to/direction.gguf` before startup.
Optional overrides: `WEIGHTLESS_STEER_ALPHA` and `WEIGHTLESS_STEER_LAYERS` (comma-separated
subset of the file's layer IDs). The reader validates the file and fails closed.

Supported rank-one `project` hooks:

* Spec 1 / `residual_stream_post_layer`: reference BF16 four-stream fold, independent
  dot and projective removal per stream. GLP-39: layers 1..39, alpha 0.75.
* Spec 3 / `hc_fold_pre_engram`: same fold, then FP32 sequential mean of four BF16
  streams, one mean-direction dot, common correction subtracted from every stream.
  Also projects the BF16 rank-order-reduced attention write *before* its normal
  post/pre boundary. GLP-27 weight-SVD export: layers 11..37, alpha 3.5.

These hooks are not equivalent; rewriting v3 metadata to v1 changes the operation.
Directions are loaded from the actual GGUF; no directions are derived at runtime.
Both projections use 320 threads x 16 columns, reference rank-order write reduction,
rounded first multiply and four FP32 FMAs in the active post-layer fold. Mean-stream
addition is sequential FP32 before the dot. No companion operation runs for v1,
OFF, alpha=0, or unselected layers. The original fused attention boundary remains
untouched in those paths. Active v3 initially reuses that collective and then
recomputes its boundary from the projected write; this prioritizes correctness
rather than eliminating the redundant boundary computation.

The same hook is called in prefill and decode/target verification. The post-layer
operation occurs before the next layer's Engram input. Draft model weights are not
steered; target verification remains authoritative.

Host tests (real files as arguments): `glp_test GLP39.gguf`, `glp_test GLP27.gguf`,
`glp_math_test GLP39.gguf`, `glp_v3_test GLP27.gguf`. These are tensor arithmetic and
reader tests, not generation evidence. `glp_v3_gpu_test GLP27.gguf` checks the actual
CUDA kernels against a host reference using synthetic activations; run on a free
GPU in an authorized window. OFF/alpha=0 generation identity and model quality,
speed, stability and behavior still require the serving trial.
