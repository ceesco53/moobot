#!/usr/bin/env bash
set -eo pipefail

# Import OHLCV CSV.GZ files from MinIO into Postgres.
# Usage:
#   DB_URL=postgresql://user:pass@host:port/db \
#   MINIO_ALIAS=realmclick MINIO_PREFIX=flatfiles/us_stocks_sip/minute_aggs_v1/2025 \
#   PARALLEL=4 PAVE=0 \
#   ./scripts/import_ohlcv.sh
#
# Notes:
# - Requires `mc` configured with the given alias (defaults to realmclick).
# - Streams objects directly via `mc cat` into psql; no local storage needed.
# - PARALLEL controls concurrency of object imports.
# - PAVE=1 truncates ohlcv before import.

DB_URL="${DB_URL:-postgresql://moobot:moobot@localhost:5432/marketdata}"
PARALLEL="${PARALLEL:-4}"
PAVE="${PAVE:-0}"
MINIO_ALIAS="${MINIO_ALIAS:-realmclick}"
MINIO_PREFIX="${MINIO_PREFIX:-flatfiles/us_stocks_sip/minute_aggs_v1/2025}"

if ! command -v psql >/dev/null 2>&1; then
  echo "psql is required on the host to run this script." >&2
  exit 1
fi

if ! command -v mc >/dev/null 2>&1; then
  echo "mc (MinIO client) is required to run this script." >&2
  exit 1
fi

# Check access to prefix
if ! mc ls "${MINIO_ALIAS}/${MINIO_PREFIX}" >/dev/null 2>&1; then
  echo "Cannot list ${MINIO_ALIAS}/${MINIO_PREFIX}. Verify MINIO_ALIAS/MINIO_PREFIX and mc config." >&2
  exit 1
fi

if [ "$PAVE" = "1" ]; then
  echo "PAVE enabled: truncating ohlcv table..."
  psql "$DB_URL" -v ON_ERROR_STOP=1 <<'SQL'
\set ON_ERROR_STOP 1
TRUNCATE TABLE ohlcv;
SQL
fi

import_file() {
  local obj="$1"
  local db="$DB_URL"

  # decompressor (host-side)
  local dec_cmd
  if command -v pigz >/dev/null 2>&1; then
    dec_cmd="pigz -dc"
  else
    dec_cmd="gzip -dc"
  fi

  echo "Importing ${obj}"

  if [[ "$obj" == *.gz ]]; then
    # use COPY FROM PROGRAM with host path
    psql "$db" -v ON_ERROR_STOP=1 <<SQL
\set ON_ERROR_STOP 1
CREATE TEMP TABLE ohlcv_stage_tmp (LIKE ohlcv_stage INCLUDING DEFAULTS INCLUDING CONSTRAINTS);
\copy ohlcv_stage_tmp FROM PROGRAM 'mc cat "${obj}" | ${dec_cmd}' CSV HEADER;
INSERT INTO ohlcv (ticker, volume, open, close, high, low, window_start, transactions)
SELECT
  ticker,
  volume,
  open,
  close,
  high,
  low,
  to_timestamp(window_start_ns / 1e9)::timestamptz AS window_start,
  transactions
FROM ohlcv_stage_tmp
ON CONFLICT (ticker, window_start) DO NOTHING;
SQL
  else
    psql "$db" -v ON_ERROR_STOP=1 <<SQL
\set ON_ERROR_STOP 1
CREATE TEMP TABLE ohlcv_stage_tmp (LIKE ohlcv_stage INCLUDING DEFAULTS INCLUDING CONSTRAINTS);
\copy ohlcv_stage_tmp FROM PROGRAM 'mc cat "${obj}"' CSV HEADER;
INSERT INTO ohlcv (ticker, volume, open, close, high, low, window_start, transactions)
SELECT
  ticker,
  volume,
  open,
  close,
  high,
  low,
  to_timestamp(window_start_ns / 1e9)::timestamptz AS window_start,
  transactions
FROM ohlcv_stage_tmp
ON CONFLICT (ticker, window_start) DO NOTHING;
SQL
  fi
}

export -f import_file
export DB_URL
export MINIO_ALIAS MINIO_PREFIX

# Find objects in MinIO prefix and import
mc find "${MINIO_ALIAS}/${MINIO_PREFIX}" --name "*.csv" --name "*.csv.gz" --exec 'echo {}' | \
  xargs -I{} -n1 -P "$PARALLEL" bash -c 'import_file "$@"' _ {}

echo "Parallel import launched (PARALLEL=$PARALLEL)."
