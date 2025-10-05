# dags/assignment5_stock_prices.py
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import List, Dict

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
import requests

# ------------------- CONFIG -------------------
DAG_ID = "assignment5_stock_prices"

# Airflow UI → Admin → Variables
SYMBOL = Variable.get("stock_symbol", default_var="GOOG").upper()

# Alpha Vantage key; set this in Admin → Variables
AV_VAR_KEY = "ALPHAVANTAGE_API_KEY"   # change here if you used a different name

# Airflow UI → Admin → Connections: create connection id = snowflake_conn (type: Snowflake)
SNOWFLAKE_CONN_ID = "snowflake_conn"

TARGET_SCHEMA = "RAW"
TARGET_TABLE  = "ASSIGNMENT_5"        # RAW.ASSIGNMENT_5

# ------------------- TASKS -------------------
@task
def ensure_objects():
    """Create RAW schema and target table if missing."""
    hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
    conn = hook.get_conn()
    cur = conn.cursor()
    try:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TARGET_SCHEMA}.{TARGET_TABLE} (
              SYMBOL VARCHAR(10) NOT NULL,
              DATE   DATE        NOT NULL,
              OPEN   FLOAT       NOT NULL,
              CLOSE  FLOAT       NOT NULL,
              HIGH   FLOAT       NOT NULL,
              LOW    FLOAT       NOT NULL,
              VOLUME INTEGER     NOT NULL,
              PRIMARY KEY (SYMBOL, DATE)
            )
            """
        )
        conn.commit()
        print(f"Ensured {TARGET_SCHEMA}.{TARGET_TABLE} exists.")
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close(); conn.close()


@task
def extract_prices(symbol: str = SYMBOL, days: int = 90) -> List[Dict]:
    """
    Get last `days` trading days from Alpha Vantage.
    IMPORTANT: return date as string for JSON/XCom compatibility.
    """
    api_key = Variable.get(AV_VAR_KEY, default_var=None)
    if not api_key:
        raise ValueError(f"Airflow Variable '{AV_VAR_KEY}' is missing.")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": api_key,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    ts = data.get("Time Series (Daily)")
    if not ts:
        # Usually rate limit or bad key; the payload has "Note" or "Error Message"
        raise RuntimeError(f"Alpha Vantage response error: {data}")

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date()

    rows: List[Dict] = []
    for d_str, vals in ts.items():
        d = datetime.strptime(d_str, "%Y-%m-%d").date()
        if d >= cutoff:
            rows.append({
                "symbol": symbol,
                "date": d.strftime("%Y-%m-%d"),     # <-- string, safe for XCom
                "open": float(vals["1. open"]),
                "high": float(vals["2. high"]),
                "low":  float(vals["3. low"]),
                "close": float(vals["4. close"]),
                "volume": int(float(vals["5. volume"])),
            })
    rows.sort(key=lambda r: r["date"])
    return rows[-days:]  # exactly last `days` trading days


@task
def load_prices_idempotent(rows: List[Dict]):
    """
    Idempotent upsert:
      1) load into a TEMP stage table,
      2) MERGE into RAW.ASSIGNMENT_5.
    """
    if not rows:
        print("No rows to load.")
        return

    hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
    conn = hook.get_conn()
    cur = conn.cursor()
    try:
        conn.cursor().execute("BEGIN")

        # session-scoped temp table (no schema qualifier)
        cur.execute("""
            CREATE TEMP TABLE IF NOT EXISTS ASSIGNMENT5_STAGE (
              SYMBOL VARCHAR(10),
              DATE   DATE,
              OPEN   FLOAT,
              CLOSE  FLOAT,
              HIGH   FLOAT,
              LOW    FLOAT,
              VOLUME INTEGER
            )
        """)
        cur.execute("DELETE FROM ASSIGNMENT5_STAGE")

        cur.executemany(
            """
            INSERT INTO ASSIGNMENT5_STAGE
              (SYMBOL, DATE, OPEN, CLOSE, HIGH, LOW, VOLUME)
            VALUES
              (%(symbol)s, TO_DATE(%(date)s), %(open)s, %(close)s, %(high)s, %(low)s, %(volume)s)
            """,
            rows,
        )

        cur.execute(
            f"""
            MERGE INTO {TARGET_SCHEMA}.{TARGET_TABLE} t
            USING ASSIGNMENT5_STAGE s
            ON t.SYMBOL = s.SYMBOL AND t.DATE = s.DATE
            WHEN MATCHED THEN UPDATE SET
              OPEN = s.OPEN, CLOSE = s.CLOSE, HIGH = s.HIGH, LOW = s.LOW, VOLUME = s.VOLUME
            WHEN NOT MATCHED THEN INSERT (SYMBOL, DATE, OPEN, CLOSE, HIGH, LOW, VOLUME)
            VALUES (s.SYMBOL, s.DATE, s.OPEN, s.CLOSE, s.HIGH, s.LOW, s.VOLUME)
            """
        )

        conn.commit()
        print(f"Upserted {len(rows)} rows into {TARGET_SCHEMA}.{TARGET_TABLE}.")
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close(); conn.close()


# ------------------- DAG -------------------
with DAG(
    dag_id=DAG_ID,
    start_date=datetime(2025, 10, 1),
    schedule="30 2 * * *",   # run daily at 02:30
    catchup=False,
    tags=["ETL", "stocks"],
) as dag:
    ensure = ensure_objects()
    prices = extract_prices()
    ensure >> load_prices_idempotent(prices)
