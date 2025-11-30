#!/usr/bin/env bash
set -eo pipefail

# Import OHLCV CSV/CSV.GZ files from S3/MinIO directly into ClickHouse using the s3() table function.
# Usage:
#   CLICKHOUSE_HOST=localhost CLICKHOUSE_PORT=9000 CLICKHOUSE_USER=moobot CLICKHOUSE_PASSWORD=moobot CLICKHOUSE_DB=marketdata \
#   BLOBSTORE_HOST=minio.example.com BLOBSTORE_PORT=9000 BLOBSTORE_KEY=... BLOBSTORE_SECRET=... \
#   MINIO_PREFIX=bucket/flatfiles/us_stocks_sip/minute_aggs_v1/2025 \
#   MONTHS=2 PAVE=0 \
#   ./scripts/clickhouse_import_ohlcv.sh
#
# Notes:
# - Requires `clickhouse-client` in PATH.
# - Streams directly from object storage; no local files required.
# - MONTHS filters to last N months including current, based on YYYY/MM substrings in the object path.
# - PAVE=1 truncates ohlcv before import.

# Load .env if present
if [ -f ".env" ]; then
  set -a
  source ".env"
  set +a
fi

CLICKHOUSE_HOST="${CLICKHOUSE_HOST:-localhost}"
CLICKHOUSE_PORT="${CLICKHOUSE_PORT:-9000}" # native default
CLICKHOUSE_USER="${CLICKHOUSE_USER:-moobot}"
CLICKHOUSE_PASSWORD="${CLICKHOUSE_PASSWORD:-moobot}"
CLICKHOUSE_DB="${CLICKHOUSE_DB:-marketdata}"
CLICKHOUSE_CLIENT_BIN="${CLICKHOUSE_CLIENT_BIN:-clickhouse-client}"
PAVE="${PAVE:-0}"
MINIO_PREFIX="${MINIO_PREFIX:-bucket/flatfiles/us_stocks_sip/minute_aggs_v1}"
MONTHS="${MONTHS:-0}"
BLOBSTORE_HOST="${BLOBSTORE_HOST:?BLOBSTORE_HOST is required}"
BLOBSTORE_PORT="${BLOBSTORE_PORT:-9000}"
BLOBSTORE_KEY="${BLOBSTORE_KEY:?BLOBSTORE_KEY is required}"
BLOBSTORE_SECRET="${BLOBSTORE_SECRET:?BLOBSTORE_SECRET is required}"
BLOBSTORE_SCHEME="${BLOBSTORE_SCHEME:-http}"

if [ "$PAVE" = "1" ]; then
  echo "PAVE enabled: truncating ohlcv table..."
  ${CLICKHOUSE_CLIENT_BIN} --host "$CLICKHOUSE_HOST" --port "$CLICKHOUSE_PORT" \
    --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" \
    --query "TRUNCATE TABLE IF EXISTS ${CLICKHOUSE_DB}.ohlcv"
fi

# Build month patterns (YYYY/MM) for last N months including current
build_months() {
  python - <<'PY'
import datetime, os
m = int(os.getenv("MONTHS", "0"))
today = datetime.date.today()
months = []
count = max(m, 1) if m > 0 else 0
for i in range(count):
    total = today.year * 12 + today.month - 1 - i
    y, mo = divmod(total, 12)
    months.append(f"{y:04d}/{mo+1:02d}")
if months:
    print(" ".join(months))
PY
}

MONTH_LIST=$(build_months)

run_import() {
  local url="$1"
  echo "Importing from ${url}"
  ${CLICKHOUSE_CLIENT_BIN} --host "$CLICKHOUSE_HOST" --port "$CLICKHOUSE_PORT" \
    --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" \
    --query "
      INSERT INTO ${CLICKHOUSE_DB}.ohlcv (ticker, volume, open, close, high, low, window_start, transactions)
      SELECT
        ticker,
        volume,
        open,
        close,
        high,
        low,
        toDateTime64(window_start / 1e9, 9),
        transactions
      FROM s3(
        '${url}',
        '${BLOBSTORE_KEY}',
        '${BLOBSTORE_SECRET}',
        'CSVWithNames',
        'ticker String, volume Float64, open Float64, close Float64, high Float64, low Float64, window_start Int64, transactions UInt64'
      )
    "
}

if [ "$MONTHS" -gt 0 ] && [ -n "$MONTH_LIST" ]; then
  for ym in $MONTH_LIST; do
    run_import "${BLOBSTORE_SCHEME}://${BLOBSTORE_HOST}:${BLOBSTORE_PORT}/${MINIO_PREFIX}/${ym}/*.csv.gz"
    run_import "${BLOBSTORE_SCHEME}://${BLOBSTORE_HOST}:${BLOBSTORE_PORT}/${MINIO_PREFIX}/${ym}/*.csv"
  done
else
  run_import "${BLOBSTORE_SCHEME}://${BLOBSTORE_HOST}:${BLOBSTORE_PORT}/${MINIO_PREFIX}/*.csv.gz"
  run_import "${BLOBSTORE_SCHEME}://${BLOBSTORE_HOST}:${BLOBSTORE_PORT}/${MINIO_PREFIX}/*.csv"
fi

echo "Import complete."
