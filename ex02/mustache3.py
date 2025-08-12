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

def fill_data_in_db(conn):
    """Fill in the database with sample data."""
    try:
        sql_query = """
            SELECT
                user_id,
                AVG(price) AS avg_basket_price
            FROM
                customers
            WHERE
                event_type = 'purchase'
            GROUP BY
                user_id;
        """
        print("\033[92mDatabase table created successfully!\033[0m")
        return pd.read_sql_query(sql_query, conn)
    except Exception as e:
        print(f"\033[91mError creating table: {e}\033[0m")

def create_avg_basket_price_boxplot(conn):
    """
    Creates a box plot for the average basket price per user.
    Outliers are hidden for clarity.
    """
    try:
        df = fill_data_in_db(conn)
        if df.empty:
            print("\033[93mNo data found for the specified query.\033[0m")
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#ffffff')  
        ax.set_facecolor('#dee2e6')
        ax.boxplot(
            df['avg_basket_price'],
            vert=False,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor='lightblue', edgecolor='black'),
            medianprops=dict(color='red', linewidth=1),
            whiskerprops=dict(color='#555555'),
            capprops=dict(color='#555555')
        )
        ax.set_yticks([])
        ax.set_title('Average Basket Price per User (Oct 2022 - Feb 2023)', color='#333333')
        ax.set_xlabel('Average Basket Price (A)', color='#555555')
        ax.set_xlim(-5, 16)
        ax.tick_params(axis='x', colors='#555555')
        ax.grid(True, axis='x', linestyle='-', alpha=1, color='#ffffff')
        ax.grid(False, axis='y')

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
