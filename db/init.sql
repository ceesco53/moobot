-- Initial schema for OHLCV data

-- Raw staging table mirrors the CSV layout (nanosecond epoch for window_start)
CREATE TABLE IF NOT EXISTS ohlcv_stage (
  ticker TEXT,
  volume BIGINT,
  open NUMERIC(18,6),
  close NUMERIC(18,6),
  high NUMERIC(18,6),
  low NUMERIC(18,6),
  window_start_ns BIGINT,
  transactions BIGINT
);

-- Final table with normalized timestamp
CREATE TABLE IF NOT EXISTS ohlcv (
  ticker TEXT NOT NULL,
  volume BIGINT,
  open NUMERIC(18,6),
  close NUMERIC(18,6),
  high NUMERIC(18,6),
  low NUMERIC(18,6),
  window_start TIMESTAMPTZ NOT NULL,
  transactions BIGINT,
  CONSTRAINT ohlcv_unique UNIQUE (ticker, window_start)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_time ON ohlcv(ticker, window_start);
