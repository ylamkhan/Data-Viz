import os
import pandas as pd
import psycopg2  # Fixed typo: was psycopy2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from tqdm import tqdm


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

def create_table(conn, table_name):
    """Create a table in the database."""
    try:
        with conn.cursor() as cursor:
            if table_name == "customers":
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS customers (
                        event_time TIMESTAMP,
                        event_type TEXT,
                        product_id BIGINT,
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
            # Count rows before removing duplicates
            cursor.execute("SELECT COUNT(*) FROM customers;")
            before_count = cursor.fetchone()[0]
            print(f"Rows before removing duplicates: {before_count}")

            # Remove duplicates: same user_id, event_type, product_id within 1 second
            cursor.execute("""
                DELETE FROM customers a
                USING customers b
                WHERE
                    a.ctid < b.ctid
                    AND a.user_id = b.user_id
                    AND a.event_type = b.event_type
                    AND a.product_id = b.product_id
                    AND ABS(EXTRACT(EPOCH FROM (a.event_time - b.event_time))) <= 1;
            """)
            conn.commit()

            # Count rows after removing duplicates
            cursor.execute("SELECT COUNT(*) FROM customers;")
            after_count = cursor.fetchone()[0]
            print(f"Rows after removing duplicates: {after_count}")
            print("\033[92mDuplicate rows removed from customers table.\033[0m")
    except Exception as e:
        print(f"\033[91mError removing duplicates from {table_name}: {e}\033[0m")



def joined_table(conn):
    """
    Alternative version with progress tracking for very large tables.
    Still faster than the original batched approach.
    """
    try:
        with conn.cursor() as cursor:
            print("Adding columns to customers table...")
            
            # Add columns if they don't exist
            for col, coltype in [
                ("category_id", "NUMERIC"),
                ("category_code", "TEXT"),
                ("brand", "TEXT")
            ]:
                cursor.execute(f"ALTER TABLE customers ADD COLUMN IF NOT EXISTS {col} {coltype};")
            
            # Create index for faster joins
            print("Creating index on items.product_id...")
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_items_product_id 
                ON items(product_id);
            """)
            
            # Get total count for progress
            cursor.execute("SELECT COUNT(*) FROM customers c INNER JOIN items i ON c.product_id = i.product_id;")
            total_to_update = cursor.fetchone()[0]
            
            print(f"Found {total_to_update:,} customers with matching items")
            print("Updating customers table...")
            
            # Use a more efficient batched approach with temp table
            batch_size = 50000  # Larger batch size
            
            cursor.execute("SELECT COUNT(*) FROM customers;")
            total = cursor.fetchone()[0]
            
            for offset in tqdm(range(0, total, batch_size), desc="Updating customers"):
                cursor.execute("""
                    UPDATE customers c
                    SET 
                        category_id = i.category_id,
                        category_code = i.category_code,
                        brand = i.brand
                    FROM items i
                    WHERE c.product_id = i.product_id
                    AND c.ctid IN (
                        SELECT ctid FROM customers 
                        ORDER BY ctid 
                        LIMIT %s OFFSET %s
                    );
                """, (batch_size, offset))
                
                # Commit every batch to avoid long-running transactions
                if offset % (batch_size * 10) == 0:
                    conn.commit()
            
            conn.commit()
            
            # Get final statistics
            cursor.execute("SELECT COUNT(*) FROM customers;")
            total_customers = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM customers 
                WHERE category_id IS NOT NULL OR category_code IS NOT NULL OR brand IS NOT NULL;
            """)
            updated_customers = cursor.fetchone()[0]
            
            print(f"\033[92m✓ Customers table updated successfully!\033[0m")
            print(f"Total customers: {total_customers:,}")
            print(f"Updated customers: {updated_customers:,}")
            
    except Exception as e:
        conn.rollback()
        print(f"\033[91mError updating customers table: {e}\033[0m")
        raise


if __name__ == "__main__":
    # Create connection to the database
    conn = connect_db()
    if conn:
        # create_table(conn, "customers")
        # print("Creating table customers done.")
        # create_table(conn, "items")
        # print("Creating table items done.")
        # path_folder = "customer/"
        # for file in os.listdir(path_folder):
        #     full_path = os.path.join(path_folder, file)
        #     if os.path.isfile(full_path):
        #         insert_table(conn, full_path, "customers")
        # print("\033[92mInserting data into customers done.\033[0m")
        # remove_duplicates(conn, "customers")
        # path_folder = "item/"
        # for file in os.listdir(path_folder):
        #     full_path = os.path.join(path_folder, file)
        #     if os.path.isfile(full_path):
        #         insert_table(conn, full_path, "items")
        # print("\033[92mInserting data into items done.\033[0m")
        # print("\033[94mRunning joined_table start...\033[0m")
        joined_table(conn)
    else:
        print("Failed to connect to the database.")
    
    # Close the connection
    if conn:
        conn.close()