CREATE DATABASE IF NOT EXISTS marketdata;

CREATE TABLE IF NOT EXISTS marketdata.ohlcv
(
    ticker String,
    volume Float64,
    open Float64,
    close Float64,
    high Float64,
    low Float64,
    window_start DateTime64(9, 'UTC'),
    transactions UInt64
)
ENGINE = MergeTree
ORDER BY (ticker, window_start);
