import os
import pandas as pd
import numpy as np
from typing import Optional
from snowflake.connector import connect
import sqlparse
from snowflake.connector.pandas_tools import write_pandas

# Snowflake connection settings
connection_settings = {
    'user': 'ogunes@chewy.com',
    'password': '',
    'authenticator': 'https://chewy.okta.com',
    'account': 'chewy.us-east-1',
    'database': 'EDLDB',
    'schema': 'public',
    'warehouse': 'SC_PROMISE_WH',
    'role': 'SC_PROMISE_DEVELOPER_MARGIN',
    'session_parameters': {'session_timeout': '12000', },
}

def columns_to_lower(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all column names in a DataFrame to lowercase.

    Args:
        df (pd.DataFrame): The DataFrame to convert.

    Returns:
        pd.DataFrame: A new DataFrame with lowercase column names.
    """
    return df.rename(columns=lambda x: x.lower())


def execute_query_and_return_formatted_data(
    query_name: Optional[str] = None,
    date_col: Optional[str] = None,
    query_path: Optional[str] = None,
    query: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    convert_to_lowercase: bool = True
) -> pd.DataFrame:
    """
    Executes a Snowflake query and returns the formatted data as a DataFrame. Optionally converts column names to
    lowercase and sets the date column as datetime.

    Args:
        query_name (str, optional): The name of the SQL query file (without extension). Defaults to None, in which case
                                    the query is assumed to be provided directly in the `query` parameter.
        date_col (str, optional): The name of the column containing date data. If provided, the column is converted
                                  to datetime format. Defaults to None.
        query_path (str, optional): The path to the directory containing the SQL query file.
        query (str, optional): The SQL query string to execute. If provided, overrides reading the query from a file.
                               Defaults to None.
        convert_to_lowercase (bool, optional): Whether to convert column names to lowercase. Defaults to True.

    Returns:
        pd.DataFrame: The DataFrame containing the results of the executed query.
    """
    if query:
        query_to_execute = query
    elif query_name and query_path:
        with open(f"{query_path}/{query_name}.sql", "r") as query_file:
            query_to_execute = query_file.read()
    else:
        raise ValueError("Either `query` or both `query_name` and `query_path` must be provided.")

    if start_date and end_date:
        start_date = "'" + start_date + "'"
        query_to_execute = query_to_execute.replace('{start_date}', start_date)
        end_date = "'" + end_date + "'"
        query_to_execute = query_to_execute.replace('{end_date}', end_date)

    with connect(**connection_settings) as connection:
        df = pd.read_sql(query_to_execute, connection)

    if convert_to_lowercase:
        df = df.rename(columns=lambda x: x.lower())

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])

    return df


def read_sql_query(file_name: str, file_path: str) -> Optional[str]:

    """
    Reads a SQL query from a file and returns it as a string.

    Args:
        file_name (str): The name of the SQL file.
        file_path (str): The path to the directory containing the SQL file.

    Returns:
        Optional[str]: The SQL query as a string.
    """
    try:
        with open(f"{file_path}/{file_name}.sql", "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"SQL file not found: {file_path}/{file_name}.sql")
        return None


def execute_query(
    query_name: Optional[str] = None,
    query_path: Optional[str] = None,
    query: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    Executes a Snowflake query.

    Args:
        query_name (str, optional): The name of the SQL query file (without extension). Defaults to None, in which case
                                    the query is assumed to be provided directly in the `query` parameter.
        query_path (str, optional): The path to the directory containing the SQL query file.
        query (str, optional): The SQL query string to execute. If provided, overrides reading the query from a file.
                               Defaults to None.
    """
    if query:
        query_to_execute = query
    elif query_name and query_path:
        with open(f"{query_path}/{query_name}.sql", "r") as query_file:
            query_to_execute = query_file.read()
    else:
        raise ValueError("Either `query` or both `query_name` and `query_path` must be provided.")

    if start_date and end_date:
        start_date = "'" + start_date + "'"
        query_to_execute = query_to_execute.replace('{start_date}', start_date)
        end_date = "'" + end_date + "'"
        query_to_execute = query_to_execute.replace('{end_date}', end_date)

    # print(query_to_execute)

    with connect(**connection_settings) as connection:
        cursor = connection.cursor()
        cursor.execute(query_to_execute)


def execute_multiple_query_and_return_formatted_data(
    query: Optional[str] = None,
    query_name: Optional[str] = None,
    query_path: Optional[str] = None,
    date_col: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    convert_to_lowercase: bool = True
) -> pd.DataFrame:
    """
    Executes multiple Snowflake queries and returns the formatted data as a DataFrame. Optionally converts column names to
    lowercase and sets the date column as datetime.

    Args:
        query_name (str, optional): The name of the SQL query file (without extension). Defaults to None, in which case
                                    the query is assumed to be provided directly in the `query` parameter.
        date_col (str, optional): The name of the column containing date data. If provided, the column is converted
                                  to datetime format. Defaults to None.
        query_path (str, optional): The path to the directory containing the SQL query file.
        convert_to_lowercase (bool, optional): Whether to convert column names to lowercase. Defaults to True.

    Returns:
        pd.DataFrame: The DataFrame containing the results of the executed query.
    """
    if query:
        query_to_execute = query
    elif query_name and query_path:
        with open(f"{query_path}/{query_name}.sql", "r") as query_file:
            query_to_execute = query_file.read()
    else:
        raise ValueError("Either `query` or both `query_name` and `query_path` must be provided.")

    if start_date and end_date:
        start_date = "'" + start_date + "'"
        query_to_execute = query_to_execute.replace('{start_date}', start_date)
        end_date = "'" + end_date + "'"
        query_to_execute = query_to_execute.replace('{end_date}', end_date)

    stmts = [s.strip() for s in sqlparse.split(query_to_execute) if s.strip()]
    df = None

    connection = connect(**connection_settings)
    cs = connection.cursor()

    for stmt in stmts:
        cur = cs.execute(stmt)
        if cur.description:
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
            df = pd.DataFrame(rows, columns=cols)
            if convert_to_lowercase:
                df.columns = [col.lower() for col in df.columns]
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])

    return df

def insert_data_to_snowflake(
    df: pd.DataFrame,
    table_name: str,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    overwrite: bool = False,
    auto_create_table: bool = True
):
    """
    Inserts data into a Snowflake table.

    Args:
        df (pd.DataFrame): The DataFrame to insert.
        table_name (str): The name of the Snowflake table.
        database (str, optional): The database name. If None, uses connection default.
        schema (str, optional): The schema name. If None, uses connection default.
        overwrite (bool): Whether to overwrite existing table. Defaults to False.
        auto_create_table (bool): Whether to auto-create table if it doesn't exist. Defaults to True.

    Returns:
        tuple: (success, nchunks, nrows, output) from write_pandas
    """
    # Convert column names to uppercase for Snowflake compatibility
    df_copy = df.copy()
    df_copy.columns = [col.upper() for col in df_copy.columns]

    connection = connect(**connection_settings)

    try:
        # Build write_pandas arguments
        write_kwargs = {
            'conn': connection,
            'df': df_copy,
            'table_name': table_name.upper(),
            'overwrite': overwrite,
            'auto_create_table': auto_create_table
        }
        
        # Only add database/schema if explicitly provided
        if database:
            write_kwargs['database'] = database.upper()
        if schema:
            write_kwargs['schema'] = schema.upper()
        
        success, nchunks, nrows, output = write_pandas(**write_kwargs)
        
        # Commit the transaction
        connection.commit()
        
        return success, nchunks, nrows, output
    finally:
        connection.close()
