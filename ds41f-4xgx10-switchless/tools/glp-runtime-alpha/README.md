# GLP runtime-alpha trial image

Builds the unqualified trial image that carries the runtime steering control
([docs/steering-control.md](../../docs/steering-control.md)): the pinned production base image with
`dsgb10_rank` and `dsfront` replaced, nothing else changed. Model, vision assets, entrypoint and the
engine argv are inherited.

Build only on an off-serving host. Required: Docker, the reviewed CUDA 13 aarch64 cross-builder image,
an aarch64 CUDA/RDMA sysroot (CUDA 13.0 sbsa libs, `libibverbs`, the RDMA headers), the Rust aarch64
cross-builder, the pinned production base image, and about 10 GiB of free space. Never build on a
serving Spark and never copy a single binary into production.

From `ds41f-4xgx10-switchless/`, with a clean committed revision:

```sh
: "${CROSS_BUILDER:?immutable CUDA cross-builder image}"        # e.g. pico-cross-builder:13.0
: "${FRONTEND_BUILDER:?Rust aarch64 builder image}"            # e.g. pico-vision-frontend-build:20261108
: "${ARM_SYSROOT:?absolute aarch64 sysroot directory}"
: "${BASE_IMAGE:?production image as tag@sha256}"
: "${TRIAL_IMAGE:?private repository:versioned tag}"           # e.g. harbor.lab.tf/library/pico-dsgx10:TAG
: "${OUT:?new absolute directory outside git}"
mkdir -p "$OUT/engine" "$OUT/frontend"
src=$(pwd); test -z "$(git status --porcelain)"; rev=$(git rev-parse HEAD)

docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" --entrypoint /bin/bash \
  -v "$src:/src:ro" -v "$ARM_SYSROOT:/sysroot:ro" \
  -v "$src/tools/glp-runtime-alpha/cross-aarch64.cmake:/toolchain.cmake:ro" \
  -v "$OUT/engine:/build" "$CROSS_BUILDER" -c \
  'cmake -S /src/engine -B /build -DCMAKE_TOOLCHAIN_FILE=/toolchain.cmake -DCMAKE_BUILD_TYPE=Release &&
   cmake --build /build --target dsgb10_rank -j6'

docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" --entrypoint /bin/bash \
  -e CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
  -e CARGO_TARGET_DIR=/build -e CARGO_HOME=/usr/local/cargo \
  -v "$src/frontend:/src:ro" -v "$OUT/frontend:/build" "$FRONTEND_BUILDER" -c \
  'cd /src && /usr/local/cargo/bin/cargo build --offline --locked --release --target aarch64-unknown-linux-gnu -j6'

# Create but never start the base container: take only its launcher metadata.
cid=$(docker create --platform linux/arm64 --entrypoint /bin/true "$BASE_IMAGE")
docker cp "$cid:/opt/pico/profile.json" "$OUT/base-profile.json"
docker cp "$cid:/opt/pico/entrypoint.py" "$OUT/base-entrypoint.py"

mkdir -p "$OUT/context"
cp "$OUT/engine/server/dsgb10_rank" "$OUT/frontend/aarch64-unknown-linux-gnu/release/dsfront" "$OUT/context/"
cp docs/steering-control.md "$OUT/context/"
python3 - "$OUT" "$rev" "$BASE_IMAGE" <<'PY'
import hashlib, json, pathlib, sys
out, rev, base = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
p = json.loads((out / 'base-profile.json').read_text())
for name, key in [('dsgb10_rank', 'engine_sha256'), ('dsfront', 'frontend_sha256')]:
    p[key] = hashlib.sha256((out / 'context' / name).read_bytes()).hexdigest()
p['source_commit'] = p['patched_source_commit'] = p['frontend_source_commit'] = rev
p['adoption'] = {'status': 'UNQUALIFIED GLP runtime-alpha trial', 'production_swapped': False}
p['production_note'] = 'Not deployed; needs independent review and the TP4 GPU gates'
p['baseline_source_commit'] = '4e70b2ab551c2e24f541981f185d634881728bcb'
p['rollback'] = {'image': base,
                 'authorization': 'separate coordinated restart authorization; image rollback does not recover sessions'}
(out / 'context' / 'profile.json').write_text(json.dumps(p, indent=2) + '\n')
print(json.dumps({k: p[k] for k in ('engine_sha256', 'frontend_sha256', 'source_commit')}, indent=2))
PY
cp "$OUT/base-entrypoint.py" "$OUT/context/entrypoint.py"

engine_sha=$(sha256sum "$OUT/context/dsgb10_rank" | cut -d' ' -f1)
frontend_sha=$(sha256sum "$OUT/context/dsfront" | cut -d' ' -f1)
docker buildx build --builder ringside-arm64-x86test --platform linux/arm64 --load \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" --build-arg "SOURCE_REVISION=$rev" \
  --build-arg "ENGINE_SHA256=$engine_sha" --build-arg "FRONTEND_SHA256=$frontend_sha" \
  --build-arg "TRIAL_VERSION=${TRIAL_IMAGE##*:}" -t "$TRIAL_IMAGE" "$OUT/context"
docker push "$TRIAL_IMAGE"
docker image inspect "$TRIAL_IMAGE" --format '{{json .RepoDigests}}'
```

A pushed digest is not qualification. Before any deployment: the TP4 GPU gates in
[steering-control.md](../../docs/steering-control.md) (parity with no set call, a live set under load
with four agreeing read-backs, set-to-same-value as a device no-op, and the speed cost), plus the
serving regression battery. Deployment and the coordinated restart are separate decisions.
