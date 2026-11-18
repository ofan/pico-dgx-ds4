# One trial artifact: the pinned production base, with both verified binaries replaced.
# Model, vision assets, entrypoint and engine argv are inherited unchanged; only the two
# binaries and the profile's hashes differ (docs/steering-control.md).
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG SOURCE_REVISION
ARG ENGINE_SHA256
ARG FRONTEND_SHA256
ARG TRIAL_VERSION
COPY --chmod=0755 dsgb10_rank dsfront /opt/pico/
COPY profile.json /opt/pico/profile.json
COPY steering-control.md /opt/pico/docs/steering-control.md
LABEL org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.version="${TRIAL_VERSION}" \
      org.opencontainers.image.source="https://github.com/ofan/pico-dgx-ds4" \
      homeinfra.engine.sha256="${ENGINE_SHA256}" \
      homeinfra.frontend.sha256="${FRONTEND_SHA256}" \
      homeinfra.build="off-Spark ARM64 cross-build" \
      homeinfra.checkpoint="authoritative pinned model and model_revision in /opt/pico/profile.json" \
      homeinfra.steering="runtime alpha control only; startup steering args and numerics unchanged" \
      pico.base.image="${BASE_IMAGE}" \
      homeinfra.change="operator-settable GLP steering alpha across all four ranks, epoch-tagged state" \
      homeinfra.status="UNQUALIFIED GLP runtime-alpha trial"
