import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv
import psycopg2

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

def get_frequency_data(conn):
    """Get frequency data from the database."""
    try:
        sql_query = """
            SELECT
                user_id,
                COUNT(*) AS frequency
            FROM
                customers
            WHERE
                event_type = 'purchase'
            GROUP BY
                user_id;
        """
        df = pd.read_sql_query(sql_query, conn)
        print("\033[92mFrequency data fetched successfully!\033[0m")
        return df
    except Exception as e:
        print(f"\033[91mError fetching frequency data: {e}\033[0m")
        return None

def get_monetary_data(conn):
    """Get monetary data from the database."""
    try:
        sql_query = """
            SELECT user_id, SUM(price) as total_spent
            FROM customers
            WHERE event_type = 'purchase'
            GROUP BY user_id
            HAVING SUM(price) < 225;
        """
        df = pd.read_sql_query(sql_query, conn)
        print("\033[92mMonetary data fetched successfully!\033[0m")
        return df
    except Exception as e:
        print(f"\033[91mError fetching monetary data: {e}\033[0m")
        return None

def create_histograms(conn, save_plot=False, show_plot=True):
    """
    Connects to a PostgreSQL database, fetches frequency and monetary data,
    and plots the result as histograms.
    
    Args:
        conn: A database connection object.
        save_plot: Boolean to save the plot as PNG file.
        show_plot: Boolean to display the plot.
    """
    try:
        # Fetch data from database
        df_frequency = get_frequency_data(conn)
        df_monetary = get_monetary_data(conn)
        
        if df_frequency is None or df_monetary is None:
            print("\033[91mFailed to fetch data from database.\033[0m")
            return
        
        # Process the data
        frequency = [freq for freq in df_frequency['frequency'].tolist() if freq <= 40]
        monetary = df_monetary['total_spent'].tolist()
        
        print(f"\033[94mData Summary:\033[0m")
        print(f"  - Frequency data points: {len(frequency)}")
        print(f"  - Monetary data points: {len(monetary)}")
        print(f"  - Frequency range: {min(frequency)}-{max(frequency)}")
        print(f"  - Monetary range: ${min(monetary):.2f}-${max(monetary):.2f}")

        # Create the plots
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('#ffffff')
        
        # Set background colors
        axs[0].set_facecolor('#dee2e6')
        axs[1].set_facecolor('#dee2e6')

        # Remove top and right spines
        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Set axis limits
        axs[0].set_xlim(0, 40)
        axs[1].set_xlim(0, 235)

        # Set tick marks
        axs[0].set_xticks(range(0, 41, 10))
        axs[1].set_xticks(range(0, 231, 50))

        # Set titles
        axs[0].set_title('Frequency distribution of the number of orders per customer', 
                        color='#333333', fontweight='bold', pad=20)
        axs[1].set_title('Frequency distribution of the purchase prices per customer', 
                        color='#333333', fontweight='bold', pad=20)

        # Set axis labels
        axs[0].set_xlabel('frequency', color='#555555')
        axs[1].set_xlabel('Monetary value in Altairian Dollars (A$)', color='#555555')

        axs[0].set_ylabel('customers', color='#555555')
        axs[1].set_ylabel('Count of customers', color='#555555')

        # Set tick colors
        for ax in axs:
            ax.tick_params(axis='x', colors='#555555')
            ax.tick_params(axis='y', colors='#555555')

        # Set y-axis limits - both start at 0 like traditional histograms
        axs[0].set_ylim(0, 80000)
        axs[1].set_ylim(0, 80000)

        # Add grid
        for ax in axs:
            ax.grid(True, linestyle='-', alpha=1, color='#ffffff', zorder=0)
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, linestyle='-', alpha=0.7)

        # Plot the frequency histogram with separate bars
        n_freq, bins_freq, patches_freq = axs[0].hist(frequency, bins=5, edgecolor='k', alpha=0.8, color='#87CEEB', rwidth=0.8)
        
        # Color each bar differently for frequency chart
        colors_freq = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        for patch, color in zip(patches_freq, colors_freq):
            patch.set_facecolor(color)
            
        # Plot the monetary histogram as traditional histogram like the example
        n_mon, bins_mon, patches_mon = axs[1].hist(monetary, bins=5, edgecolor='k', alpha=0.8, color='#87CEEB', rwidth=1.0)
        
        # Color each bar differently for monetary chart  
        colors_mon = ['#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43']
        for patch, color in zip(patches_mon, colors_mon):
            patch.set_facecolor(color)

        # Adjust layout
        plt.tight_layout()

        # Save plot if requested
        if save_plot:
            plt.savefig('frequency_monetary_distribution.png', dpi=300, bbox_inches='tight')
            print("\033[92mPlot saved as 'frequency_monetary_distribution.png'\033[0m")

        # Show plot if requested
        if show_plot:
            plt.show()

        print("\033[92mHistograms successfully created!\033[0m")

    except Exception as e:
        print(f"\033[91mError creating histograms: {e}\033[0m")

def main():
    """Main function to run the histogram creation."""
    print("\033[96m" + "="*60)
    print("DATABASE HISTOGRAM GENERATOR")
    print("="*60 + "\033[0m")
    
    conn = connect_db()
    if conn:
        try:
            create_histograms(conn, save_plot=True, show_plot=True)
        except KeyboardInterrupt:
            print("\n\033[93mOperation cancelled by user.\033[0m")
        finally:
            conn.close()
            print("\033[92mDatabase connection closed.\033[0m")
    else:
        print("\033[91mFailed to connect to the database.\033[0m")

if __name__ == "__main__":
    main()