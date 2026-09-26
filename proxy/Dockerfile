# syntax=docker/dockerfile:1
#
# Phase 15B — TLS-terminating reverse proxy (single public edge).
#
# Build context: ./proxy
#
# The proxy is the ONLY service with host-published ports. It terminates TLS
# (TLS 1.2/1.3 only), redirects HTTP to HTTPS, and routes:
#
#   /api/, /health, /health/ready, /docs, /openapi.json, /redoc -> backend:8000
#   everything else                                             -> frontend:80
#
# Certificates are NOT baked into the image. They are mounted read-only from
# the host (see compose.yaml) via configurable environment variables, so the
# same image works with the generated development certificates or an
# operator-provided trusted certificate.
#
# The mount targets /etc/nginx/certs/server.crt and /etc/nginx/certs/server.key,
# so the mount point directory must exist in the image. Docker can only
# bind-mount a host FILE when every parent directory of the target already
# exists; without this the container fails to start with "not a directory: Are
# you trying to mount a directory onto a file (or vice-versa)?".

FROM nginx:1.27-alpine

RUN mkdir -p /etc/nginx/certs

COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80 443