import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from tqdm import tqdm


def connect_db():
    """Connect to the PostgreSQL database."""
    try:
        load_dotenv()
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

def create_table(conn, table_name):
    """Create a table in the database."""
    try:
        with conn.cursor() as cursor:
            if table_name == "customers":
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS customers (
                        event_time TIMESTAMP,
                        event_type TEXT,
                        product_id NUMERIC,
                        price FLOAT,
                        user_id BIGINT,
                        user_session TEXT
                    );
                """)
            elif table_name == "items":
                cursor.execute("""
                        CREATE TABLE IF NOT EXISTS items (
                        product_id NUMERIC,
                        category_id NUMERIC,
                        category_code TEXT,
                        brand TEXT
                    );
                """)
            conn.commit()
            cursor.close()
            print(f"\033[92mTable '{table_name}' created successfully.\033[0m")
    except Exception as e:
        print(f"\033[91mError creating table '{table_name}': {e}\033[0m")
        cursor.close()  # Ensure cursor is closed on error

def insert_table(conn, file_path, table_name, batch_size=1000):
    """Insert data from a CSV file into the specified table with progress bar."""
    try:
        df = pd.read_csv(file_path)
        with conn.cursor() as cursor:
            if table_name == "customers":
                sql = """
                    INSERT INTO customers (event_time, event_type, product_id, price, user_id, user_session)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """
            elif table_name == "items":
                sql = """
                    INSERT INTO items (product_id, category_id, category_code, brand)
                    VALUES (%s, %s, %s, %s);
                """
            else:
                print(f"Unknown table: {table_name}")
                return

            data = df.values.tolist()
            for i in tqdm(range(0, len(data), batch_size), desc=f'Inserting into {table_name}'):
                batch = data[i:i+batch_size]
                execute_batch(cursor, sql, batch)
            conn.commit()
            cursor.close()
            print(f"Data from {file_path} inserted into {table_name} successfully.")
    except Exception as e:
        print(f"Error inserting data from {file_path} into {table_name}: {e}")
        cursor.close()  # Ensure cursor is closed on error


def remove_duplicates(conn, table_name):
    """Remove duplicate rows from the specified table."""
    try:
        with conn.cursor() as cursor:
            if table_name == "customers":

                cursor.execute("SELECT COUNT(*) FROM customers;")
                before_count = cursor.fetchone()[0]
                print(f"Rows before removing duplicates: {before_count}")

                cursor.execute("""
                    CREATE TEMPORARY TABLE temp_customers AS SELECT DISTINCT * FROM customers;
                    TRUNCATE customers;
                    INSERT INTO customers SELECT * FROM temp_customers;
                """)
                conn.commit()

                # Count rows after removing duplicates
                cursor.execute("SELECT COUNT(*) FROM customers;")
                after_count = cursor.fetchone()[0]
                print(f"Rows after removing duplicates: {after_count}")
                print("\033[92mDuplicate rows removed from customers table.\033[0m")
            else:
                 # Count rows before removing duplicates
                cursor.execute("SELECT COUNT(*) FROM items;")
                before_count = cursor.fetchone()[0]
                print(f"Rows before removing duplicates: {before_count}")

                cursor.execute("""
                    CREATE TABLE items_nodup AS
                    SELECT
                        product_id,
                        MAX(CASE WHEN category_id IS NOT NULL AND category_id != 'NaN' THEN category_id ELSE NULL END) AS category_id,
                        MAX(CASE WHEN category_code IS NOT NULL AND category_code != 'NaN' THEN category_code ELSE NULL END) AS category_code,
                        MAX(CASE WHEN brand IS NOT NULL AND brand != 'NaN' THEN brand ELSE NULL END) AS brand
                    FROM
                        items
                    GROUP BY
                        product_id;
                """)
                conn.commit()

                # Count rows after removing duplicates
                cursor.execute("SELECT COUNT(*) FROM items_nodup;")
                after_count = cursor.fetchone()[0]
                print(f"Rows after removing duplicates: {after_count}")
                print("\033[92mDuplicate rows removed from customers table.\033[0m")

    except Exception as e:
        print(f"\033[91mError removing duplicates from {table_name}: {e}\033[0m")


def joined_table(conn):
    try:
        cursor = conn.cursor()
        va_quires = """
            ALTER TABLE customers
            ADD COLUMN category_id NUMERIC,
            ADD COLUMN category_code TEXT,
            ADD COLUMN brand TEXT;

            UPDATE customers c
            SET
                category_id = i.category_id,
                category_code = i.category_code,
                brand = i.brand
            FROM items_nodup i
            WHERE c.product_id = i.product_id;
        """
        cursor.execute(va_quires)
        conn.commit()
        print("\033[92mCustomers table updated with category and brand info.\033[0m")

    except Exception as e:
        print(f"\033[91mError updating customers table: {e}\033[0m")



   


if __name__ == "__main__":
    # Create connection to the database
    conn = connect_db()
    if conn:
        create_table(conn, "customers")
        print("Creating table customers done.")
        create_table(conn, "items")
        print("Creating table items done.")
        path_folder = "customer/"
        for file in os.listdir(path_folder):
            full_path = os.path.join(path_folder, file)
            if os.path.isfile(full_path):
                insert_table(conn, full_path, "customers")
        print("\033[92mInserting data into customers done.\033[0m")
        remove_duplicates(conn, "customers")
        path_folder = "item/"
        for file in os.listdir(path_folder):
            full_path = os.path.join(path_folder, file)
            if os.path.isfile(full_path):
                insert_table(conn, full_path, "items")
        print("\033[92mInserting data into items done.\033[0m")
        remove_duplicates(conn, "items")
        print("\033[94mRunning joined_table start...\033[0m")
        joined_table(conn)
    else:
        print("Failed to connect to the database.")
    
    # Close the connection
    if conn:
        conn.close()