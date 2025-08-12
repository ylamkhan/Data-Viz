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
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )

        print("\033[92mDatabase connection successful.\033[0m")
        return conn
    except Exception as e:
        print(f"\033[91mError connecting to the database: {e}\033[0m")
        return None



def create_charts_pie(conn):
    """Create pie charts for the number of items per category and brand."""
    try:
        # Load data
        query = "SELECT event_type, COUNT(*) as count FROM customers GROUP BY event_type"
        df = pd.read_sql_query(query, conn)
        print(df)

        if df.empty:
            print("\033[93mNo data found for pie chart.\033[0m")
            return

        # Colors similar to your example
        colors = ['blue', 'green', 'orange', 'red']

        # Create pie chart
        plt.figure(figsize=(6, 6))
        wedges, texts, autotexts = plt.pie(
            df['count'],
            labels=df['event_type'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            pctdistance=0.8,  # Push % closer to center
            labeldistance=1.05,  # Push labels slightly outside
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'}  # White border
        )

        # Make percentage labels white and bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.axis('equal')  # Equal aspect ratio ensures circle shape
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\033[91mError creating pie chart: {e}\033[0m")


def main():
    conn = connect_db()
    if conn:
        create_charts_pie(conn)
        conn.close()
    else:
        print("\033[91mFailed to connect to the database.\033[0m")


if __name__ == "__main__":
    main()
    