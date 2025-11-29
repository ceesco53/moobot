#!/usr/bin/env bash
set -eo pipefail

# Import daily OHLCV CSV/CSV.GZ files into Postgres.
# Usage:
#   DB_URL=postgresql://user:pass@host:port/db \
#   PARALLEL=4 PAVE=0 \
#   ./scripts/import_ohlcv.sh /path/to/data/root
#
# Notes:
# - Processes files on the host and streams them into psql.
# - No container path mapping is required; files must be readable on the host.
# - PARALLEL controls concurrency.
# - PAVE=1 truncates ohlcv before import.

DATA_DIR="${1:-data}"
DB_URL="${DB_URL:-postgresql://moobot:moobot@localhost:5432/marketdata}"
PARALLEL="${PARALLEL:-4}"
PAVE="${PAVE:-0}"

if ! command -v psql >/dev/null 2>&1; then
  echo "psql is required on the host to run this script." >&2
  exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
  echo "Data directory not found: $DATA_DIR" >&2
  exit 1
fi

if [ "$PAVE" = "1" ]; then
  echo "PAVE enabled: truncating ohlcv table..."
  psql "$DB_URL" -v ON_ERROR_STOP=1 <<'SQL'
\set ON_ERROR_STOP 1
TRUNCATE TABLE ohlcv;
SQL
fi

# Early exit if no files
if ! find "$DATA_DIR" -type f \( -iname "*.csv" -o -iname "*.csv.gz" \) -print -quit | grep -q .; then
  echo "No CSV or CSV.GZ files were imported from $DATA_DIR or its subfolders."
  exit 0
fi

import_file() {
  local f="$1"
  local db="$DB_URL"

  # decompressor (host-side)
  local dec_cmd
  if command -v pigz >/dev/null 2>&1; then
    dec_cmd="pigz -dc"
  else
    dec_cmd="gzip -dc"
  fi

  if [ ! -s "$f" ]; then
    echo "Skipping empty file: $f"
    return 0
  fi
  if [ ! -r "$f" ]; then
    echo "Skipping unreadable file: $f"
    return 0
  fi

  echo "Importing $f"

  if [[ "$f" == *.gz ]]; then
    # use COPY FROM PROGRAM with host path
    psql "$db" -v ON_ERROR_STOP=1 <<SQL
\set ON_ERROR_STOP 1
CREATE TEMP TABLE ohlcv_stage_tmp (LIKE ohlcv_stage INCLUDING DEFAULTS INCLUDING CONSTRAINTS);
\copy ohlcv_stage_tmp FROM PROGRAM '${dec_cmd} "${f}"' CSV HEADER;
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
\copy ohlcv_stage_tmp FROM '${f}' CSV HEADER;
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

find "$DATA_DIR" -type f \( -iname "*.csv" -o -iname "*.csv.gz" \) -print0 | \
  xargs -0 -n1 -P "$PARALLEL" bash -c 'import_file "$@"' _

echo "Parallel import launched (PARALLEL=$PARALLEL)."
