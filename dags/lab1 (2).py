# ================================================================
# Project: Stock Price Prediction (AMZN & META)
#
# Overview:
# This file defines two Airflow DAGs that automate stock data collection
# and price forecasting using yfinance and Snowflake ML.
#
#   1. ETL_DAG – Fetches and loads 180 days of stock data (AMZN, META)
#   2. Stock_Forecasting_DAG – Verifies ETL data, trains a model in Snowflake,
#      runs a 7-day forecast, and merges the forecast with history.
#
# Notes:
# • The first DAG extracts, cleans, and loads daily stock data into Snowflake.
# • The second DAG checks that data exists, trains a forecasting model,
#   and produces a combined table with both actuals and predictions.
# • All Snowflake write operations use transactions (BEGIN/COMMIT/ROLLBACK)
#   for safety and consistency.
# • Airflow Connection: 'snowflake_conn' stores Snowflake credentials.
#
# Schedule:
#     ETL_DAG → runs daily at 02:30 AM
#     Stock_Forecasting_DAG → runs daily at 03:00 AM
#
# Team members:
#   1. Danish Waseem - 019101511
#   2. Srinidhi Jaya Revanth Srirangarajapally - 019123143
#
# This setup was implemented as part of our first lab for building
# automated stock price forecasting pipelines.
# ================================================================

from airflow import DAG
from airflow.decorators import task
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from datetime import datetime
import yfinance as yf
import pandas as pd


# -------------------------------------------------------------------------
# Helper: connect to Snowflake
# Uses Airflow’s SnowflakeHook so credentials remain secure inside
# the Airflow Connection named 'snowflake_conn'.
# -------------------------------------------------------------------------
def return_snowflake_conn():
    hook = SnowflakeHook(snowflake_conn_id='snowflake_conn')
    conn = hook.get_conn()
    return conn.cursor()


# ======================================================================
# DAG 1: ETL_DAG
# Purpose:
#   Extract → Transform → Load pipeline that fetches data from yfinance
#   and stores it in Snowflake for further processing.
# ======================================================================

@task
def extract_finance(tickers, days=180):
    """
    Fetch 180 days of daily OHLCV data for all tickers from yfinance.
    Using yfinance.Ticker().history() since it gives a simple DataFrame.
    """
    rows = []
    for sym in tickers:
        df = yf.Ticker(sym).history(period=f"{days}d", interval="1d", auto_adjust=False)
        if df is None or df.empty:
            continue

        # Sometimes 'Close' column is missing; fallback to 'Adj Close'
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        need = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in need):
            continue

        df = df[need].dropna(subset=need)

        # Convert each row to a dict for easy Snowflake insertion
        for dt, r in df.iterrows():
            rows.append({
                "symbol": sym,
                "date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
                "open": float(r["Open"]),
                "high": float(r["High"]),
                "low": float(r["Low"]),
                "close": float(r["Close"]),
                "volume": 0 if pd.isna(r["Volume"]) else int(r["Volume"]),
            })
    return rows


@task
def transform_finance(raw_rows):
    """
    Clean data to ensure valid values.
    Filters out rows with missing or negative values.
    """
    out = []
    for r in raw_rows:
        if not all(k in r for k in ("symbol", "date", "open", "high", "low", "close", "volume")):
            continue
        if any(v is None for v in (r["open"], r["high"], r["low"], r["close"], r["volume"])):
            continue
        if r["volume"] < 0:
            continue
        out.append(r)
    return out


@task
def load_to_snowflake(cur, records, target_table):
    """
    Create the target table if it doesn’t exist,
    delete existing rows for the same symbols (idempotent),
    and then insert the new data.
    """
    try:
        # Create table if not exists
        cur.execute("BEGIN;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {target_table} (
                DATE DATE NOT NULL,
                OPEN FLOAT NOT NULL,
                HIGH FLOAT NOT NULL,
                LOW FLOAT NOT NULL,
                CLOSE FLOAT NOT NULL,
                VOLUME BIGINT NOT NULL,
                SYMBOL VARCHAR(10) NOT NULL,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            );
        """)
        cur.execute("COMMIT;")

        if not records:
            return

        # Delete existing rows for these symbols (ensures idempotency)
        symbols = sorted({r["symbol"] for r in records})
        sym_list = ",".join([f"'{s}'" for s in symbols])
        cur.execute("BEGIN;")
        cur.execute(f"DELETE FROM {target_table} WHERE SYMBOL IN ({sym_list});")

        # Insert all new records
        for r in records:
            cur.execute(f"""
                INSERT INTO {target_table}
                (DATE, OPEN, HIGH, LOW, CLOSE, VOLUME, SYMBOL)
                VALUES (
                    TO_DATE('{r['date']}', 'YYYY-MM-DD'),
                    {r['open']}, {r['high']}, {r['low']},
                    {r['close']}, {r['volume']}, '{r['symbol']}'
                );
            """)

        cur.execute("COMMIT;")
    except Exception as e:
        cur.execute("ROLLBACK;")
        raise e


# --- Define ETL DAG ---
with DAG(
    dag_id='ETL_DAG',
    start_date=datetime(2025, 10, 5),
    schedule='30 2 * * *',  # runs daily at 2:30 AM
    catchup=False,
    tags=['ETL', 'StockAPI'],
) as etl_dag:

    tickers = ["META", "AMZN"]
    target_table = "RAW.MARKET_DATA"
    cur = return_snowflake_conn()

    extracted = extract_finance(tickers)
    transformed = transform_finance(extracted)
    loaded = load_to_snowflake(cur, transformed, target_table)

    extracted >> transformed >> loaded


# ======================================================================
# DAG 2: Stock_Forecasting_DAG
# Purpose:
#   Verify ETL data → train a Snowflake ML model → forecast next 7 days →
#   merge predictions with historical data.
# ======================================================================

@task
def verify_etl(cur, target_table):
    """
    Simple validation step to ensure ETL data exists.
    This prevents forecasting when no data is available.
    """
    result = cur.execute(f"SELECT COUNT(*) FROM {target_table}").fetchone()[0]
    if result == 0:
        raise ValueError(f"No data found in {target_table}. Run ETL_DAG first.")


@task
def train(cur, target_table, view_name, model_name, tickers):
    """
    Create a training view and train a multiseries forecast model in Snowflake ML.
    """
    sym_list = ",".join([f"'{s}'" for s in tickers])
    try:
        cur.execute("BEGIN;")

        # Create training view
        cur.execute(f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT
                TO_TIMESTAMP_NTZ(DATE) AS DATE_v1,
                CLOSE,
                SYMBOL
            FROM {target_table}
            WHERE SYMBOL IN ({sym_list});
        """)

        # Train the Snowflake ML Forecast model
        cur.execute(f"""
            CREATE OR REPLACE SNOWFLAKE.ML.FORECAST {model_name}(
                INPUT_DATA => SYSTEM$REFERENCE('VIEW', '{view_name}'),
                SERIES_COLNAME => 'SYMBOL',
                TIMESTAMP_COLNAME => 'DATE_v1',
                TARGET_COLNAME => 'CLOSE',
                CONFIG_OBJECT => {{ 'ON_ERROR': 'SKIP' }}
            );
        """)

        cur.execute("COMMIT;")
    except Exception as e:
        cur.execute("ROLLBACK;")
        raise e


@task
def predict(cur, model_name, forecast_table, combined_table, target_table, tickers):
    """
    Run the trained model’s forecast procedure (7-day horizon),
    capture results via RESULT_SCAN, and union forecasts with history.
    """
    view_clean = "MARKET_DATA_FORECAST_CLEAN"
    sym_list = ",".join([f"'{s}'" for s in tickers])

    try:
        # Forecast and capture results using Snowflake’s SQLID/RESULT_SCAN pattern
        cur.execute(f"""
            BEGIN
                CALL {model_name}!FORECAST(
                    FORECASTING_PERIODS => 14,
                    CONFIG_OBJECT => {{'prediction_interval': 0.95}}
                );
                LET x := SQLID;
                CREATE OR REPLACE TABLE {forecast_table} AS
                SELECT * FROM TABLE(RESULT_SCAN(:x));
            END;
        """)

        # Create cleaned forecast view
        cur.execute(f"""
            CREATE OR REPLACE VIEW {view_clean} AS
            SELECT
                REPLACE(series, '"','') AS SYMBOL,
                CAST(ts AS TIMESTAMP_NTZ) AS DATE_v1,
                forecast,
                lower_bound,
                upper_bound
            FROM {forecast_table};
        """)

        # Combine historical and forecast data
        cur.execute(f"""
            CREATE OR REPLACE TABLE {combined_table} AS
            SELECT
                SYMBOL,
                DATE AS DATE,
                CLOSE AS ACTUAL,
                NULL AS FORECAST,
                NULL AS LOWER_BOUND,
                NULL AS UPPER_BOUND
            FROM {target_table}
            WHERE SYMBOL IN ({sym_list})

            UNION ALL

            SELECT
                SYMBOL,
                CAST(DATE_v1 AS DATE),
                NULL AS ACTUAL,
                FORECAST,
                LOWER_BOUND,
                UPPER_BOUND
            FROM {view_clean};
        """)
    except Exception as e:
        cur.execute("ROLLBACK;")
        raise e


# --- Define Forecast DAG ---
with DAG(
    dag_id='Stock_Forecasting_DAG',
    start_date=datetime(2025, 10, 5),
    schedule='0 3 * * *',  # runs daily at 3:00 AM
    catchup=False,
    tags=['Forecast', 'SnowflakeML'],
) as forecast_dag:

    DB="USER_DB_HEDGEHOG"
    tickers = ["META", "AMZN"]
    target_table = f"{DB}.RAW.MARKET_DATA_LAB"
    view_name = "MARKET_DATA_v1"
    model_name = "stock_price"
    forecast_table = f"{DB}.RAW.MARKET_DATA_FORECAST_RAW"
    combined_table = f"{DB}.RAW.MARKET_DATA_COMBINED"
    cur = return_snowflake_conn()

    # Step order for forecasting workflow
    verify = verify_etl(cur, target_table)
    t_train = train(cur, target_table, view_name, model_name, tickers)
    t_pred = predict(cur, model_name, forecast_table, combined_table, target_table, tickers)

    # Linear dependency chain
    verify >> t_train >> t_pred
