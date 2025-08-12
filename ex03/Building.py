import psycopg2
import os
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np


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


def create_charts(conn):
    """
    Creates two bar charts:
    1. Number of orders according to frequency
    2. Altairian Dollars spent by customers
    
    Args:
        conn: A database connection object.
    """
    try:
        # Query 1: Get order frequency per customer
        frequency_query = """
        SELECT 
            user_id,
            COUNT(*) as order_count
        FROM customers
        WHERE event_type = 'purchase'
            AND event_time >= '2022-10-01'
            AND event_time < '2023-03-01'
        GROUP BY user_id
        ORDER BY order_count;
        """
        
        # Query 2: Get total spending per customer
        spending_query = """
        SELECT 
            user_id,
            SUM(price) as total_spent
        FROM customers
        WHERE event_type = 'purchase'
            AND event_time >= '2022-10-01'
            AND event_time < '2023-03-01'
            AND price IS NOT NULL
        GROUP BY user_id
        ORDER BY total_spent;
        """

        # Fetch frequency data
        cursor = conn.cursor()
        cursor.execute(frequency_query)
        frequency_results = cursor.fetchall()
        frequency_df = pd.DataFrame(frequency_results, columns=['user_id', 'order_count'])
        
        # Fetch spending data
        cursor.execute(spending_query)
        spending_results = cursor.fetchall()
        spending_df = pd.DataFrame(spending_results, columns=['user_id', 'total_spent'])
        cursor.close()
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Set background colors
        fig.patch.set_facecolor('#f0f4f8')
        ax1.set_facecolor('#e6f2ff')
        ax2.set_facecolor('#e6f2ff')
        
        # Chart 1: Order frequency histogram
        ax1.hist(frequency_df['order_count'], bins=20, color='lightsteelblue', alpha=0.7, edgecolor='white')
        ax1.set_title('Number of Orders by Frequency', color='#333333', fontsize=12)
        ax1.set_xlabel('frequency', color='#555555')
        ax1.set_ylabel('customers', color='#555555')
        ax1.tick_params(axis='both', colors='#555555')
        ax1.grid(True, linestyle='-', alpha=0.3, color='#ffffff', zorder=0)
        ax1.set_axisbelow(True)
        
        # Chart 2: Spending distribution histogram
        ax2.hist(spending_df['total_spent'], bins=20, color='lightsteelblue', alpha=0.7, edgecolor='white')
        ax2.set_title('Altairian Dollars Spent by Customers', color='#333333', fontsize=12)
        ax2.set_xlabel('monetary value in A', color='#555555')
        ax2.set_ylabel('customers', color='#555555')
        ax2.tick_params(axis='both', colors='#555555')
        ax2.grid(True, linestyle='-', alpha=0.3, color='#ffffff', zorder=0)
        ax2.set_axisbelow(True)
        
        plt.tight_layout()
        plt.show()

        print("Order frequency and spending distribution charts successfully created")
        print(f"Total customers analyzed: {len(frequency_df)}")
        print(f"Average orders per customer: {frequency_df['order_count'].mean():.2f}")
        print(f"Average spending per customer: {spending_df['total_spent'].mean():.2f} A")
        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    conn = connect_db()
    if conn:
        create_charts(conn)
        conn.close()
    else:
        print("\033[91mFailed to connect to the database.\033[0m")