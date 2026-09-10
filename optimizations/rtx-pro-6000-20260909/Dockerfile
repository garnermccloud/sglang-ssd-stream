# This is only for the exact qualified local baseline, not stock SGLang.
# See README.md and manifest.json before building. There is no default base.
ARG QUALIFIED_BASE
FROM ${QUALIFIED_BASE}
COPY . /opt/rtx-pro-6000-qualification/
RUN python3 /opt/rtx-pro-6000-qualification/verify_bundle.py --base-root /
COPY payload/ /
RUN python3 /opt/rtx-pro-6000-qualification/verify_bundle.py --target-root /
ENV SGLANG_RAGGED_VERIFY_MODE=static
