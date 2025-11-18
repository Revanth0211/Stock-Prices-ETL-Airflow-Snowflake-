from airflow import DAG
from airflow.models import Variable
from airflow.decorators import task
from airflow.operators.python import get_current_context
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from datetime import datetime

def return_snowflake_conn():
    hook = SnowflakeHook(snowflake_conn_id='snowflake_conn')
    conn = hook.get_conn()
    return conn.cursor()

@task
def create_table_and_load(cursor):
    try:
        cursor.execute("BEGIN;")

        # Creating the tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS USER_DB_PUMA.raw.user_session_channel (
                userId int not NULL,
                sessionId varchar(32) primary key,
                channel varchar(32) default 'direct'
            );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS USER_DB_PUMA.raw.session_timestamp (
                sessionId varchar(32) primary key,
                ts timestamp
            );
        ''')

        # Create the stage to prepare to load data
        cursor.execute('''
            CREATE OR REPLACE STAGE USER_DB_PUMA.raw.blob_stage
            url = 's3://s3-geospatial/readonly/'
            file_format = (type = csv, skip_header = 1, field_optionally_enclosed_by = '"');
        ''')

        # Load the data using COPY INTO
        cursor.execute('''
            COPY INTO USER_DB_PUMA.raw.user_session_channel
            FROM @week8_demo.raw.blob_stage/user_session_channel.csv;
        ''')

        cursor.execute('''
            COPY INTO USER_DB_PUMA.raw.session_timestamp
            FROM @week8_demo.raw.blob_stage/session_timestamp.csv;
        ''')

        cursor.execute("COMMIT;")

        print("Tables created successfully.")
        print("Data loaded successfully.")

    except Exception as e:
        cursor.execute("ROLLBACK;")
        print(e)
        raise e

with DAG(
    dag_id='etl',
    start_date=datetime(2024, 10, 23),
    catchup=False,
    tags=['ETL'],
    schedule='30 2 * * *'
) as dag:
    cursor = return_snowflake_conn()
    create_table_and_load(cursor)