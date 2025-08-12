from dotenv import load_dotenv
import psycopg2
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

def connect_db():
    """Connect to the PostgreSQL database."""
    try:
        load_dotenv()  # Load environment variables from .env file
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        print("\033[92mConnected to the database successfully!\033[0m")
        return conn
    except Exception as e:
        print(f"\033[91mError connecting to the database: {e}\033[0m")
        return None

def fill_data_in_db(conn):
    """Fill in the database with sample data."""
    try:
        sql_query = """
            SELECT
                price
            FROM
                customers
            WHERE
                event_type = 'purchase'
        """
        print("\033[92mDatabase table created successfully!\033[0m")
        return pd.read_sql_query(sql_query, conn)
    except Exception as e:
        print(f"\033[91mError creating table: {e}\033[0m")

def statistics(conn):
    """
    Connects to a PostgreSQL database, fetches data, and plots the result as an area chart.
    
    Args:
        conn: A database connection object.
    """
    try:
        df = fill_data_in_db(conn)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

        # Print the mean, median, min, max, first, second and third quartile of the price of the items purchased
        print("\033[92mStatistics of the prices:\033[0m")
        print(f"count: {df['price'].count()}")
        print(f"mean: {df['price'].mean()}")
        print(f"std: {df['price'].std()}")
        print(f"min: {df['price'].min()}")
        print(f"25%: {df['price'].quantile(0.25)}")
        print(f"50%: {df['price'].quantile(0.5)}")
        print(f"75%: {df['price'].quantile(0.75)}")
        print(f"max: {df['price'].max()}")

    except Exception as e:
        print(f"\033[91mError fetching data for statistics: {e}\033[0m")

if __name__ == "__main__":
    conn = connect_db()
    if conn:
        statistics(conn)
        conn.close()
    else:
        print("Failed to connect to the database.")
