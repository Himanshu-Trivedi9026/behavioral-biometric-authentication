# syntax=docker/dockerfile:1
#
# Phase 15A — Frontend production static image (React + Vite).
#
# Multi-stage build:
#   1. build  — `npm ci` + `npm run build` produces the Vite bundle in dist/.
#   2. serve  — nginx serves the static bundle with SPA deep-link fallback.
#
# Build context: ./frontend (see compose.yaml `build.context: ./frontend`).
#
# The browser-facing API base URL is a BUILD-TIME argument that Vite embeds
# (VITE_API_BASE_URL -> import.meta.env.VITE_API_BASE_URL in src/api.js).
# compose.yaml passes it from the host `.env`; it is never hard-coded into the
# production bundle. The Vite dev server is never used in production.

# ---- build stage ------------------------------------------------------------
FROM node:22-alpine AS build

WORKDIR /app

# Install dependencies from the lockfile (reproducible).
COPY package.json package-lock.json ./
RUN npm ci

# Copy sources, then build with the injected API base URL.
COPY . .

ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

# ---- serve stage ------------------------------------------------------------
FROM nginx:1.27-alpine AS serve

# SPA fallback config so deep links (/login, /enrollment, /verification,
# /continuous, /register, /collector) resolve to index.html.
COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80