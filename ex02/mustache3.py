import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import seaborn as sns

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

def create_avg_basket_price_boxplot(conn):
    """
    Creates a box plot for the average basket price per user.
    Outliers are hidden for clarity.
    """
    try:
        sql_query = """
        SELECT
            user_id,
            AVG(price) AS avg_basket_price
        FROM
            customers
        WHERE
            event_type = 'purchase'
            AND event_time >= '2022-10-01'
            AND event_time < '2023-03-01'
        GROUP BY
            user_id;
        """
        
        df = pd.read_sql_query(sql_query, conn)

        plt.figure(figsize=(8, 5))
        sns.set_style("whitegrid")

        # Boxplot without outliers
        sns.boxplot(
            x=df['avg_basket_price'],
            color='skyblue',
            showfliers=False
        )

        plt.title("Average Basket Price per User (Oct 2022 - Feb 2023)")
        plt.xlabel("Average Basket Price (A)")

        plt.tight_layout()
        # plt.savefig("avg_basket_price_boxplot.png")
        plt.show()

        print("Box plot saved as avg_basket_price_boxplot.png")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    conn = connect_db()
    if conn:
        create_avg_basket_price_boxplot(conn)
        conn.close()
    else:
        print("\033[91mFailed to connect to the database.\033[0m")
