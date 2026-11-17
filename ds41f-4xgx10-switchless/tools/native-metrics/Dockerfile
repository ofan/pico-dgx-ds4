# One trial artifact: retain pinned model/runtime assets, replace both verified binaries.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY --chmod=0755 dsgb10_rank dsfront /opt/pico/
COPY --chmod=0755 entrypoint.py /opt/pico/entrypoint.py
COPY profile.json /opt/pico/profile.json
COPY native-telemetry.md /opt/pico/docs/native-telemetry.md
ARG BASE_IMAGE
ARG SOURCE_REVISION
ARG ENGINE_SHA256
ARG FRONTEND_SHA256
ARG TRIAL_VERSION
LABEL org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.version="${TRIAL_VERSION}" \
      homeinfra.engine.sha256="${ENGINE_SHA256}" \
      homeinfra.build="off-Spark ARM64 cross-build" \
      homeinfra.checkpoint="authoritative pinned model and model_revision in /opt/pico/profile.json" \
      homeinfra.steering="unchanged inherited production argv; telemetry does not set steering" \
      pico.base.image="${BASE_IMAGE}" \
      pico.metadata.note="inherited runtime and legacy patch labels describe base history; profile and OCI revision describe this trial" \
      pico.frontend.sha256="${FRONTEND_SHA256}" \
      homeinfra.change="native independent engine and frontend telemetry; GPU gates pending" \
      homeinfra.status="UNQUALIFIED native telemetry trial" \
      org.opencontainers.image.source="https://github.com/ofan/pico-dgx-ds4" \
      pico.telemetry.schema="1" \
      pico.qualification="UNQUALIFIED: review and TP4 saturation parity serving-regression pending"
EXPOSE 9108 9109
