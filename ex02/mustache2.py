import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

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


def create_price_boxplot(conn):
    """
    Connects to a PostgreSQL database, fetches price data for purchases,
    and plots the result as a box plot with a customized background.
    """
    try:
        df = fill_data_in_db(conn)
        if df.empty:
            print("\033[93mNo data found for the specified query.\033[0m")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#ffffff')  
        ax.set_facecolor('#dee2e6')
        ax.boxplot(
            df['price'], 
            vert=False,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor='green', edgecolor='black'),
            medianprops=dict(color='red', linewidth=1),
            whiskerprops=dict(color='#555555'),
            capprops=dict(color='#555555')
        )
        ax.set_yticks([])
        ax.set_title('Distribution of Product Prices (Oct 2022 - Feb 2023)', color='#333333')
        ax.set_xlabel('Price in A', color='#555555')
        ax.set_xlim(-0.5, 12)
        ax.tick_params(axis='x', colors='#555555')
        ax.grid(True, axis='x', linestyle='-', alpha=1, color='#ffffff')
        ax.grid(False, axis='y')
        plt.tight_layout()
        # plt.savefig('price_boxplot.png')
        plt.show()

        print("Box plot successfully created and saved as price_boxplot.png")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    conn = connect_db()
    if conn:
        create_price_boxplot(conn)
        conn.close()
    else:
        print("\033[91mFailed to connect to the database.\033[0m")
