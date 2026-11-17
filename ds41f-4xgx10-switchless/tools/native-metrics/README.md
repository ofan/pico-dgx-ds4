# Reproducible off-serving telemetry trial build

This is a private, **unqualified** trial, not a production deployment. All four
ranks use the same resulting image digest; engine/frontend stay together with
the baseline model, vision and steering asset lineage. Large text rank/Engram
files remain the baseline's externally mounted, pinned checkpoint assets, not
newly converted or silently substituted weights. This packaging does not claim
to embed the entire checkpoint in a new layer.

Build only on an off-serving host. Required: Docker, the reviewed CUDA13 ARM64
cross-builder image, ARM64 CUDA/RDMA sysroot, frontend Rust ARM64 cross-builder,
pinned production base image, and at least 30 GiB temporary free space. All
builder/base inputs must be immutable image IDs or tag+digest references. Never
build on a serving Spark or copy an individual binary into production.

From `ds41f-4xgx10-switchless/` with a **clean committed source revision**:

```sh
# Set these to the reviewed local inputs; no implicit downloads or model changes.
: "${CROSS_BUILDER:?immutable CUDA cross-builder image ID}"
: "${FRONTEND_BUILDER:?immutable Rust ARM64 builder image ID}"
: "${ARM_SYSROOT:?reviewed absolute sysroot directory}"
: "${BASE_IMAGE:?production tag@sha256 reference}"
: "${TRIAL_IMAGE:?private registry/repository:versioned-native-telemetry-trial tag}"
: "${OUT:?new absolute directory outside git}"
mkdir "$OUT"
mkdir "$OUT/engine" "$OUT/frontend"
src=$(pwd)
test -z "$(git status --porcelain)"
rev=$(git rev-parse HEAD)

docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" --entrypoint /bin/bash \
  -v "$src:/src:ro" -v "$ARM_SYSROOT:/sysroot:ro" \
  -v "$src/tools/native-metrics/cross-aarch64.cmake:/toolchain.cmake:ro" \
  -v "$OUT/engine:/build" "$CROSS_BUILDER" -c \
  'cmake -S /src/engine -B /build -DCMAKE_TOOLCHAIN_FILE=/toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/sysroot/cuda -DCUDA_CUDART=/sysroot/cuda/lib64/libcudart.so && cmake --build /build --target dsgb10_rank -j4'

docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" --entrypoint /bin/bash \
  -e CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
  -e CARGO_TARGET_DIR=/build -v "$src/frontend:/src:ro" \
  -v "$OUT/frontend:/build" "$FRONTEND_BUILDER" -c \
  'cd /src && /usr/local/cargo/bin/cargo build --offline --locked --release --target aarch64-unknown-linux-gnu -j4'

# Create but NEVER start the baseline container; copy only packaging metadata.
cid=$(docker create --platform linux/arm64 --entrypoint /bin/true "$BASE_IMAGE")
docker cp "$cid:/opt/pico/profile.json" "$OUT/base-profile.json"
docker cp "$cid:/opt/pico/entrypoint.py" "$OUT/base-entrypoint.py"
# Retain the stopped inspection container until the build is reviewed; no data cleanup here.
python3 tools/native-metrics/prepare-image.py --context "$OUT/context" \
  --baseline-profile "$OUT/base-profile.json" --baseline-entrypoint "$OUT/base-entrypoint.py" \
  --engine "$OUT/engine/server/dsgb10_rank" \
  --frontend "$OUT/frontend/aarch64-unknown-linux-gnu/release/dsfront" \
  --source-revision "$rev" --base-image "$BASE_IMAGE"
engine_sha=$(sha256sum "$OUT/context/dsgb10_rank" | cut -d' ' -f1)
frontend_sha=$(sha256sum "$OUT/context/dsfront" | cut -d' ' -f1)
docker buildx build --builder default --platform linux/arm64 --load \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" --build-arg "SOURCE_REVISION=$rev" \
  --build-arg "ENGINE_SHA256=$engine_sha" --build-arg "FRONTEND_SHA256=$frontend_sha" \
  --build-arg "TRIAL_VERSION=${TRIAL_IMAGE##*:}" -t "$TRIAL_IMAGE" "$OUT/context"
# After local tests, push only the UNQUALIFIED trial; record review status separately.
docker push "$TRIAL_IMAGE"
docker image inspect "$TRIAL_IMAGE" --format '{{json .RepoDigests}}'
```

`prepare-image.py` refuses an existing context or changed baseline launcher
patterns. It preserves existing serving args/model revisions and adds only
metrics listeners; binary hashes are enforced by the existing supervisor.
Loopback-only defaults are `127.0.0.1:9108` (each rank) and `127.0.0.1:9109`
(frontend). Optional explicit private addresses use
`PICO_ENGINE_METRICS_LISTEN` / `PICO_FRONTEND_METRICS_LISTEN`; this is not an
approval to alter routing, expose metrics publicly, or launch on production.

Before calling an image qualified, run every gate in
[the as-built contract](../../docs/native-telemetry.md), including actual TP4
saturation, token parity and <1% regression A/B. Review/test receipts belong in a
private build artifact directory, not tracked source. A pushed digest alone is
not qualification. Deployment remains a separate decision.
