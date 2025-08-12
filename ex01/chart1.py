import psycopg2
import os
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


def connect_db():
    """Connect to the PostgreSQL database."""
    try:
        load_dotenv()
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
                DATE_TRUNC('month', event_time) AS purchase_date,
                SUM(price) AS total_sales
            FROM
                customers
            WHERE
                event_type = 'purchase'
            GROUP BY
                purchase_date
            ORDER BY
                purchase_date;
        """
        print("\033[92mDatabase table created successfully!\033[0m")
        return pd.read_sql_query(sql_query, conn)
    except Exception as e:
        print(f"\033[91mError creating table: {e}\033[0m")  


def create_charts(conn):
    """
    Connects to a PostgreSQL database, fetches monthly sales data,
    and plots the result as a bar chart with a customized background.
    Args:
        conn: A database connection object.
    """
    try:
        df = fill_data_in_db(conn)
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        df['total_sales_million'] = df['total_sales'] / 1000000
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#ffffff')  
        ax.set_facecolor('#dee2e6')
        x_labels = df['purchase_date'].dt.strftime('%b')
        plt.bar(x_labels, df['total_sales_million'], color='lightsteelblue')
        ax.set_title('Total Sales in Million Altairian Dollars', color='#333333')
        ax.set_xlabel('Month', color='#555555')
        ax.set_ylabel('Total sales in million of A', color='#555555')
        ax.set_ylim(0, df['total_sales_million'].max() * 1.2)
        ax.tick_params(axis='x', colors='#555555')
        ax.tick_params(axis='y', colors='#555555')
        ax.grid(True, linestyle='-', alpha=1, color='#ffffff', zorder=0)
        ax.set_axisbelow(True)
        plt.tight_layout()
        # plt.savefig('monthly_sales_chart.png')
        plt.show()
        print("Monthly sales chart successfully created and saved as monthly_sales_chart.png")   
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    conn = connect_db()
    if conn:
        create_charts(conn)
        conn.close()
    else:
        print("\033[91mFailed to connect to the database.\033[0m")