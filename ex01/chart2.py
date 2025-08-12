import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import psycopg2
import os
from dotenv import load_dotenv
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


def create_chart(conn):
    """
    Connects to a PostgreSQL database, fetches daily average spend per customer,
    and plots the result as an area chart.
    
    Args:
        conn: A database connection object.
    """
    try:
        sql_query = """
        SELECT
            DATE_TRUNC('day', event_time) AS purchase_date,
            SUM(price) / COUNT(DISTINCT user_id) AS average_per_user
        FROM
            customers
        WHERE
            event_type = 'purchase'
            AND event_time >= '2022-10-01'
            AND event_time < '2023-03-01'
        GROUP BY
            purchase_date
        ORDER BY
            purchase_date;
        """
        
        # Use pandas to read the query result directly into a DataFrame
        df = pd.read_sql_query(sql_query, conn)
        
        # Ensure 'purchase_date' is a datetime object
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])

        # Create the area plot
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#ffffff')  
        ax.set_facecolor('#dee2e6')
        # Plot the data as a filled area
        ax.fill_between(df['purchase_date'], df['average_per_user'], color='lightsteelblue', alpha=0.8)

        # Set the x-axis to show monthly ticks
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        
        # Set the limits of the y-axis to start from 0
        plt.ylim(bottom=0)
        
        # Set titles and labels
        plt.title('Daily Average Spend Per Customer')
        plt.xlabel('Date')
        plt.ylabel('Average spend/customers in A')
        ax.grid(True, linestyle='-', alpha=1, color='#ffffff', zorder=0)
        ax.set_axisbelow(True)
        end_date = pd.to_datetime('2023-02-28')
        ax.set_xlim(left=df['purchase_date'].min(), right=end_date)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        # plt.savefig('average_spend_chart.png')
        plt.show()

        print("Average spend chart successfully created and saved as average_spend_chart.png")
        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    conn = connect_db()
    if conn:
        create_chart(conn)
        conn.close()
    else:
        print("\033[91mFailed to connect to the database.\033[0m")