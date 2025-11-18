from airflow.decorators import task
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import get_current_context
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook

from datetime import datetime
from datetime import timedelta
import logging
import snowflake.connector

"""
This pipeline assumes that there are two other tables in your snowflake DB
 - user_session_channel
 - session_timestamp

With regard to how to set up these two tables, please refer to this README file:
 - https://github.com/keeyong/sjsu-data226-FA25/blob/main/week8/How-to-setup-ETL-tables-for-ELT.md
"""

def return_snowflake_conn():

    # Initialize the SnowflakeHook
    hook = SnowflakeHook(snowflake_conn_id='snowflake_conn')

    # Execute the query and fetch results
    conn = hook.get_conn()
    return conn.cursor()


@task
def run_ctas(schema, table, select_sql, primary_key=None):

    logging.info(table)
    logging.info(select_sql)

    cur = return_snowflake_conn()

    try:
        sql = f"CREATE OR REPLACE TABLE {schema}.temp_{table} AS {select_sql}"
        logging.info(sql)
        cur.execute(sql)

        # do primary key uniquess check
        if primary_key is not None:
            sql = f"""
              SELECT {primary_key}, COUNT(1) AS cnt 
              FROM {schema}.temp_{table}
              GROUP BY 1
              ORDER BY 2 DESC
              LIMIT 1"""
            print(sql)
            cur.execute(sql)
            result = cur.fetchone()
            print(result, result[1])
            if int(result[1]) > 1:
                print("!!!!!!!!!!!!!!")
                raise Exception(f"Primary key uniqueness failed: {result}")

            # ---- Extra duplicate-records check (beyond PK uniqueness) ----
            # Detect exact duplicate rows on the business columns:
            # (sessionId, userId, channel, ts). Raise if any found.
            sql = f"""
              SELECT COUNT(*) AS dup_cnt
              FROM (
                SELECT
                  sessionId, userId, channel, ts,
                  ROW_NUMBER() OVER (
                    PARTITION BY sessionId, userId, channel, ts
                    ORDER BY sessionId
                  ) AS rn
                FROM {schema}.temp_{table}
              )
              WHERE rn > 1
            """
            print(sql)
            cur.execute(sql)
            dup_cnt = int(cur.fetchone()[0])
            if dup_cnt > 0:
                raise Exception(
                    f"Duplicate record check failed: {dup_cnt} duplicate rows "
                    f"on (sessionId, userId, channel, ts) in {schema}.temp_{table}"
                )

        main_table_creation_if_not_exists_sql = f"""
            CREATE TABLE IF NOT EXISTS {schema}.{table} AS
            SELECT * FROM {schema}.temp_{table} WHERE 1=0;"""
        cur.execute(main_table_creation_if_not_exists_sql)

        swap_sql = f"""ALTER TABLE {schema}.{table} SWAP WITH {schema}.temp_{table};"""
        cur.execute(swap_sql)
    except Exception as e:
        raise


with DAG(
    dag_id = 'ETL_PIPELINE',
    start_date = datetime(2024,10,2),
    catchup = False,
    tags = ['ELT'],
    schedule = '45 2 * * *'
) as dag:

    schema = "analytics"
    table = "session_summary"
    select_sql = """SELECT u.*, s.ts
    FROM raw.user_session_channel u
    JOIN raw.session_timestamp s ON u.sessionId=s.sessionId
    """

    run_ctas(schema, table, select_sql, primary_key='sessionId')
