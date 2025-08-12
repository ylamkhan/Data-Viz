import psycopg2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
dbname = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")

try:
    # SQL query
    sql_script = """
        SELECT user_id, COUNT(*) AS purchases
        FROM customers
        WHERE event_type = 'purchase'
        GROUP BY user_id
        HAVING COUNT(*) < 30
        ORDER BY purchases DESC;
    """
    print("SQL code has been imported!")

    # Connect to DB
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    print("Connected to PostgreSQL!")

    # Execute query
    cursor = conn.cursor()
    cursor.execute(sql_script)
    print("SQL script executed successfully!")
    data = cursor.fetchall()
    print("Data has been fetched from the table.")

    conn.commit()
    cursor.close()
    conn.close()

    # Convert to NumPy array (only purchases column for clustering)
    data_array = np.array([[row[1]] for row in data])

    # Elbow method
    wss = []
    for k in range(1, 11):  # now loop matches the plot
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(data_array)
        wss.append(kmeans.inertia_)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wss, color='royalblue')
    plt.title("The Elbow Method", fontsize=14, color="#333")
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Within-Cluster Sum of Squares (WSS)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error: {e}")
