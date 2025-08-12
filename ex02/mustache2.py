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

def create_price_boxplot(conn):
    """
    Connects to a PostgreSQL database, fetches price data for purchases,
    and plots the result as a box plot with a customized background.
    """
    try:
        sql_query = """
        SELECT
            price
        FROM
            customers
        WHERE
            event_type = 'purchase'
            AND event_time >= '2022-10-01'
            AND event_time < '2023-03-01';
        """
        
        df = pd.read_sql_query(sql_query, conn)
        
        fig, ax = plt.subplots(figsize=(10, 6))

        # Background colors
        fig.patch.set_facecolor('#f0f4f8')
        ax.set_facecolor('#e6f2ff')

        # Create the box plot without outliers
       # Create the box plot without showing outliers
        ax.boxplot(
            df['price'], 
            vert=False,
            patch_artist=True,
            showfliers=False,  # <<< Hides outliers
            boxprops=dict(facecolor='green', edgecolor='black'),
            medianprops=dict(color='red', linewidth=1),
            whiskerprops=dict(color='#555555'),
            capprops=dict(color='#555555')
        )


        # Remove y-axis ticks
        ax.set_yticks([])

        # Titles & labels
        ax.set_title('Distribution of Product Prices (Oct 2022 - Feb 2023)', color='#333333')
        ax.set_xlabel('Price in A', color='#555555')

        # Set x-axis limit (zoom)
        ax.set_xlim(-0.5, 12)  # Only show range from 0 to 12

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
