import psycopg2
import os
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Connect to the PostgreSQL database.
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
                DATE_TRUNC('day', event_time) AS purchase_date,
                COUNT(DISTINCT user_id) AS number_of_customers
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
    try:
        df = fill_data_in_db(conn)
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#ffffff')  
        ax.set_facecolor('#dee2e6')
        ax.plot(
            df['purchase_date'].to_numpy(),
            df['number_of_customers'].to_numpy(),
            linestyle='-',
            color='steelblue',
            linewidth=1.5
        )

        ax.set_title('Daily Number of Customers (Oct 2022 - Feb 2023)', color='#333333')
        ax.set_xlabel('Month', color='#555555')
        ax.set_ylabel('Number of Customers', color='#555555')

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        ax.tick_params(axis='x', colors='#555555')
        ax.tick_params(axis='y', colors='#555555')

        ax.grid(True, linestyle='-', alpha=1, color='#ffffff')
        end_date = pd.to_datetime('2023-02-28')
        ax.set_xlim(left=df['purchase_date'].min(), right=end_date)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        # plt.savefig('daily_customers_plot.png')
        plt.show()
        print("Chart successfully created and saved as daily_customers_plot.png")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    conn = connect_db()
    if conn:
        create_charts(conn)
        conn.close()
    else:
        print("\033[91mFailed to connect to the database.\033[0m")