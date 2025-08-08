
import psycopg2
import os
import pandas as pd
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




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
    try:
        query = """
        SELECT event_time, price, user_id, category_id, brand
        FROM customers 
        WHERE event_type = 'purchase' 
        AND event_time >= '2022-10-01' 
        AND event_time <= '2023-02-28'
        """
        df = pd.read_sql_query(query, conn)

        # Convert event_time to datetime
        df['event_time'] = pd.to_datetime(df['event_time'])
        df['month'] = df['event_time'].dt.to_period('M')

        # Chart 1: Monthly Revenue Trend
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        monthly_revenue = df.groupby('month')['price'].sum()
        plt.plot(range(len(monthly_revenue)), monthly_revenue.values, marker='o', linewidth=2, markersize=8)
        plt.title('Monthly Revenue Trend', fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Revenue (Altairian Dollars)')
        plt.xticks(range(len(monthly_revenue)), [str(m) for m in monthly_revenue.index], rotation=45)
        plt.grid(True, alpha=0.3)

        # Chart 2: Number of Purchases per Month
        plt.subplot(1, 3, 2)
        monthly_purchases = df.groupby('month').size()
        bars = plt.bar(range(len(monthly_purchases)), monthly_purchases.values, color='skyblue', alpha=0.8)
        plt.title('Monthly Purchase Count', fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Number of Purchases')
        plt.xticks(range(len(monthly_purchases)), [str(m) for m in monthly_purchases.index], rotation=45)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')

        # Chart 3: Average Order Value per Month
        plt.subplot(1, 3, 3)
        monthly_aov = df.groupby('month')['price'].mean()
        plt.plot(range(len(monthly_aov)), monthly_aov.values, marker='s', linewidth=2, 
                markersize=8, color='orange')
        plt.title('Average Order Value', fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('AOV (Altairian Dollars)')
        plt.xticks(range(len(monthly_aov)), [str(m) for m in monthly_aov.index], rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        # plt.savefig('ex01/purchase_analysis_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"\033[91mError creating charts: {e}\033[0m")




if __name__ == "__main__":
    conn = connect_db()
    if conn:
        create_charts(conn)
        conn.close()
    else:
        print("\033[91mFailed to connect to the database.\033[0m")
    