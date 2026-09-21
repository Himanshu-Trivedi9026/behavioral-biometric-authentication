#!/usr/bin/env sh
# =============================================================================
# Phase 15B — idempotent local DEVELOPMENT TLS certificates.
#
# Generates a throwaway local CA + a server certificate (with SANs for
# ``localhost`` and any configured host names) for the reverse proxy.
#
#   scripts/generate_dev_certs.sh            # generate if missing
#   scripts/generate_dev_certs.sh -f         # regenerate (rotate) everything
#
# Environment:
#   BBA_TLS_CERT_DIR   output directory (default: <repo>/proxy/certs)
#   BBA_TLS_HOST_NAMES comma-separated extra SAN DNS names (default: none)
#
# The generated keys/certs are DEVELOPMENT-ONLY. They are gitignored and must
# never be committed or shipped. For a real deployment, mount an
# operator-provided trusted certificate/private key into the proxy (see
# compose.yaml BBA_TLS_CERT_PATH / BBA_TLS_KEY_PATH) instead of these.
#
# Requires: openssl only. No other tooling.
# =============================================================================

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

CERT_DIR="${BBA_TLS_CERT_DIR:-"$REPO_ROOT/proxy/certs"}"
HOST_NAMES="${BBA_TLS_HOST_NAMES:-}"
FORCE="${1:-}"

CA_KEY="$CERT_DIR/bba-ca.key"
CA_CRT="$CERT_DIR/bba-ca.crt"
CA_SRL="$CERT_DIR/bba-ca.srl"
SERVER_KEY="$CERT_DIR/server.key"
SERVER_CSR="$CERT_DIR/server.csr"
SERVER_CRT="$CERT_DIR/server.crt"
SAN_FILE="$CERT_DIR/dev-san.ext"

DAYS=825 # ~2.25 years: a comfortable local-dev validity window.

if [ -f "$SERVER_CRT" ] && [ -f "$SERVER_KEY" ] && [ -f "$CA_CRT" ] && [ "$FORCE" != "-f" ]; then
    echo ":: development certificates already present ($SERVER_CRT) — skipping"
    echo ":: re-run with -f to regenerate"
    exit 0
fi

mkdir -p "$CERT_DIR"

if [ "$FORCE" = "-f" ]; then
    rm -f "$CA_KEY" "$CA_CRT" "$CA_SRL" "$SERVER_KEY" "$SERVER_CSR" "$SERVER_CRT" "$SAN_FILE"
fi

# ---- local CA ---------------------------------------------------------------
if [ ! -f "$CA_KEY" ] || [ ! -f "$CA_CRT" ]; then
    echo ":: generating local development CA"
    openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$CA_KEY"
    openssl req -x509 -new -key "$CA_KEY" -sha256 -days "$DAYS" \
        -subj "/CN=Behavioral Biometric Authentication Dev CA" -out "$CA_CRT"
fi

# ---- server certificate -----------------------------------------------------
echo ":: generating server certificate (SAN: DNS:localhost + host names)"
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$SERVER_KEY"
openssl req -new -key "$SERVER_KEY" -subj "/CN=localhost" -out "$SERVER_CSR"

{
    echo "subjectAltName = @alt_names"
    echo
    echo "[alt_names]"
    echo "DNS.1 = localhost"
    echo "IP.1 = 127.0.0.1"
    index=2
    for host in $(printf '%s' "$HOST_NAMES" | tr ',' ' '); do
        if [ -n "$host" ]; then
            echo "DNS.$index = $host"
            index=$((index + 1))
        fi
    done
} > "$SAN_FILE"

openssl x509 -req -in "$SERVER_CSR" -CA "$CA_CRT" -CAkey "$CA_KEY" -CAcreateserial \
    -days "$DAYS" -sha256 -extfile "$SAN_FILE" -out "$SERVER_CRT"

rm -f "$SERVER_CSR" "$SAN_FILE"

chmod 600 "$CA_KEY" "$SERVER_KEY"

echo ":: generated:"
echo "   CA     $CA_CRT"
echo "   cert   $SERVER_CRT"
echo "   key    $SERVER_KEY"
echo
echo ":: DEVELOPMENT ONLY — gitignored; never commit, never ship."
echo ":: For a real deployment mount a trusted certificate (see compose.yaml)."