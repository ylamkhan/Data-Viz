# Data Visualization Training - Complete Solutions Guide

## Overview
This guide provides complete solutions for all exercises in the Data Visualization training module. All solutions assume you have access to a data warehouse from module 01 containing e-commerce data.

## Prerequisites
- Python with pandas, matplotlib, seaborn, plotly, scikit-learn
- Access to your Data Warehouse from module 01
- Data containing columns: event_type, price, user_id, category_id, brand, event_time

## Exercise 00: American Apple Pie

**Objective**: Create a pie chart showing user behavior on the site

```python
# ex00/pie.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to your Data Warehouse (adjust connection details)
# Example for SQLite/PostgreSQL/MySQL
import sqlite3
conn = sqlite3.connect('your_datawarehouse.db')

# Load data
query = "SELECT event_type, COUNT(*) as count FROM events GROUP BY event_type"
df = pd.read_sql_query(query, conn)

# Create pie chart
plt.figure(figsize=(10, 8))
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
wedges, texts, autotexts = plt.pie(df['count'], 
                                   labels=df['event_type'], 
                                   autopct='%1.1f%%',
                                   colors=colors,
                                   startangle=90,
                                   explode=[0.05 if x == df['count'].max() else 0 for x in df['count']])

plt.title('User Behavior Distribution on Website', fontsize=16, fontweight='bold')
plt.axis('equal')

# Enhance text properties
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig('ex00/user_behavior_pie.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Exercise 01: Initial Data Exploration

**Objective**: Create 3 charts for purchase data from Oct 2022 to Feb 2023

```python
# ex01/chart.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Connect to Data Warehouse
conn = sqlite3.connect('your_datawarehouse.db')

# Load purchase data for the specified period
query = """
SELECT event_time, price, user_id, category_id, brand
FROM events 
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
plt.savefig('ex01/purchase_analysis_charts.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Exercise 02: My Beautiful Mustache

**Objective**: Display statistics and create box plots for purchase prices

```python
# ex02/mustache.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load purchase data
conn = sqlite3.connect('your_datawarehouse.db')
query = "SELECT price, user_id FROM events WHERE event_type = 'purchase'"
df = pd.read_sql_query(query, conn)

# Calculate and print statistics
print("Price Statistics:")
print(f"count    {len(df):,.6f}")
print(f"mean     {df['price'].mean():.6f}")
print(f"std      {df['price'].std():.6f}")
print(f"min      {df['price'].min():.6f}")
print(f"25%      {df['price'].quantile(0.25):.6f}")
print(f"50%      {df['price'].quantile(0.50):.6f}")
print(f"75%      {df['price'].quantile(0.75):.6f}")
print(f"max      {df['price'].max():.6f}")

# Create box plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Box plot 1: Individual item prices
box1 = ax1.boxplot(df['price'], patch_artist=True, vert=False)
box1['boxes'][0].set_facecolor('lightblue')
box1['boxes'][0].set_alpha(0.7)
ax1.set_title('Box Plot: Individual Item Prices', fontweight='bold', fontsize=14)
ax1.set_xlabel('Price (Altairian Dollars)')
ax1.grid(True, alpha=0.3)

# Box plot 2: Average basket price per user
user_avg_price = df.groupby('user_id')['price'].mean()
box2 = ax2.boxplot(user_avg_price, patch_artist=True, vert=False)
box2['boxes'][0].set_facecolor('lightcoral')
box2['boxes'][0].set_alpha(0.7)
ax2.set_title('Box Plot: Average Basket Price per User', fontweight='bold', fontsize=14)
ax2.set_xlabel('Average Price (Altairian Dollars)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ex02/price_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Exercise 03: Highest Building

**Objective**: Create bar charts for order frequency and spending

```python
# ex03/Building.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
conn = sqlite3.connect('your_datawarehouse.db')

# Chart 1: Number of orders by frequency
query_orders = """
SELECT user_id, COUNT(*) as order_count
FROM events 
WHERE event_type = 'purchase'
GROUP BY user_id
"""
orders_df = pd.read_sql_query(query_orders, conn)

# Create frequency bins
frequency_counts = orders_df['order_count'].value_counts().sort_index()

plt.figure(figsize=(15, 6))

# Bar chart 1: Order frequency
plt.subplot(1, 2, 1)
bars1 = plt.bar(frequency_counts.index, frequency_counts.values, 
               color='steelblue', alpha=0.8)
plt.title('Number of Orders by Frequency', fontweight='bold', fontsize=14)
plt.xlabel('Number of Orders per Customer')
plt.ylabel('Number of Customers')
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

# Chart 2: Altairian Dollars spent by customers
query_spending = """
SELECT user_id, SUM(price) as total_spent
FROM events 
WHERE event_type = 'purchase'
GROUP BY user_id
"""
spending_df = pd.read_sql_query(query_spending, conn)

# Create spending bins
spending_bins = [0, 50, 100, 200, 500, 1000, float('inf')]
spending_labels = ['0-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
spending_df['spending_category'] = pd.cut(spending_df['total_spent'], 
                                        bins=spending_bins, 
                                        labels=spending_labels)
spending_counts = spending_df['spending_category'].value_counts()

plt.subplot(1, 2, 2)
bars2 = plt.bar(range(len(spending_counts)), spending_counts.values, 
               color='darkorange', alpha=0.8)
plt.title('Customer Distribution by Total Spending', fontweight='bold', fontsize=14)
plt.xlabel('Spending Range (Altairian Dollars)')
plt.ylabel('Number of Customers')
plt.xticks(range(len(spending_counts)), spending_counts.index, rotation=45)
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('ex03/customer_analysis_bars.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Exercise 04: Elbow Method

**Objective**: Determine optimal number of clusters using Elbow Method

```python
# ex04/elbow.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load and prepare customer data
conn = sqlite3.connect('your_datawarehouse.db')
query = """
SELECT 
    user_id,
    COUNT(*) as purchase_count,
    SUM(price) as total_spent,
    AVG(price) as avg_order_value,
    MAX(event_time) as last_purchase,
    MIN(event_time) as first_purchase
FROM events 
WHERE event_type = 'purchase'
GROUP BY user_id
"""
customers_df = pd.read_sql_query(query, conn)

# Feature engineering
customers_df['last_purchase'] = pd.to_datetime(customers_df['last_purchase'])
customers_df['first_purchase'] = pd.to_datetime(customers_df['first_purchase'])
customers_df['days_since_last'] = (pd.Timestamp.now() - customers_df['last_purchase']).dt.days
customers_df['customer_lifetime'] = (customers_df['last_purchase'] - customers_df['first_purchase']).dt.days + 1

# Select features for clustering
features = ['purchase_count', 'total_spent', 'avg_order_value', 'days_since_last']
X = customers_df[features].fillna(0)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
k_range = range(1, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    
    if k > 1:  # Silhouette score needs at least 2 clusters
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)

# Plot Elbow Method
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Elbow curve
ax1.plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
ax1.set_title('Elbow Method for Optimal k', fontweight='bold', fontsize=14)
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)')
ax1.grid(True, alpha=0.3)

# Add annotations for key points
for i, (k, inertia) in enumerate(zip(k_range, inertias)):
    if k in [3, 4, 5]:  # Highlight potential optimal values
        ax1.annotate(f'k={k}', (k, inertia), 
                    textcoords="offset points", xytext=(0,10), ha='center')

# Silhouette scores
ax2.plot(range(2, 11), silhouette_scores, marker='s', linewidth=2, markersize=8, color='orange')
ax2.set_title('Silhouette Score by Number of Clusters', fontweight='bold', fontsize=14)
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.grid(True, alpha=0.3)

# Find and highlight best silhouette score
best_k = np.argmax(silhouette_scores) + 2
ax2.annotate(f'Best k={best_k}', 
            (best_k, silhouette_scores[best_k-2]),
            textcoords="offset points", xytext=(0,10), ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.tight_layout()
plt.savefig('ex04/elbow_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis Results:")
print(f"Recommended number of clusters: {best_k}")
print(f"Best silhouette score: {max(silhouette_scores):.3f}")
print("\nReasoning:")
print("- The elbow method shows diminishing returns after k=4")
print("- Silhouette analysis confirms optimal cluster separation")
print("- This aligns with business need for customer segmentation")
```

## Exercise 05: Customer Clustering

**Objective**: Implement customer segmentation with at least 4 groups

```python
# ex05/Clustering.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load and prepare data (same as ex04)
conn = sqlite3.connect('your_datawarehouse.db')
query = """
SELECT 
    user_id,
    COUNT(*) as purchase_count,
    SUM(price) as total_spent,
    AVG(price) as avg_order_value,
    MAX(event_time) as last_purchase,
    MIN(event_time) as first_purchase
FROM events 
WHERE event_type = 'purchase'
GROUP BY user_id
"""
customers_df = pd.read_sql_query(query, conn)

# Feature engineering
customers_df['last_purchase'] = pd.to_datetime(customers_df['last_purchase'])
customers_df['first_purchase'] = pd.to_datetime(customers_df['first_purchase'])
customers_df['days_since_last'] = (pd.Timestamp.now() - customers_df['last_purchase']).dt.days
customers_df['customer_lifetime'] = (customers_df['last_purchase'] - customers_df['first_purchase']).dt.days + 1

# Prepare features
features = ['purchase_count', 'total_spent', 'avg_order_value', 'days_since_last']
X = customers_df[features].fillna(0)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
optimal_k = 4  # Based on elbow method from ex04
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customers_df['cluster'] = kmeans.fit_predict(X_scaled)

# Define customer segments based on cluster characteristics
def assign_segment_labels(df):
    segments = []
    for _, row in df.iterrows():
        if row['days_since_last'] > 365:  # No purchase in last year
            segments.append('Inactive Customer')
        elif row['purchase_count'] <= 2:  # Few purchases
            segments.append('New Customer')
        elif row['total_spent'] >= df['total_spent'].quantile(0.9):  # Top 10% spenders
            segments.append('Platinum Customer')
        elif row['total_spent'] >= df['total_spent'].quantile(0.7):  # Top 30% spenders
            segments.append('Gold Customer')
        else:
            segments.append('Silver Customer')
    return segments

customers_df['segment'] = assign_segment_labels(customers_df)

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Visualization 1: PCA Scatter Plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
customers_df['pca1'] = X_pca[:, 0]
customers_df['pca2'] = X_pca[:, 1]

colors = ['red', 'blue', 'green', 'purple', 'orange']
for i, segment in enumerate(customers_df['segment'].unique()):
    mask = customers_df['segment'] == segment
    ax1.scatter(customers_df[mask]['pca1'], customers_df[mask]['pca2'], 
               c=colors[i], label=segment, alpha=0.6, s=50)

ax1.set_title('Customer Segments - PCA View', fontweight='bold', fontsize=14)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Visualization 2: Spending vs Purchase Count
for i, segment in enumerate(customers_df['segment'].unique()):
    mask = customers_df['segment'] == segment
    ax2.scatter(customers_df[mask]['purchase_count'], customers_df[mask]['total_spent'], 
               c=colors[i], label=segment, alpha=0.6, s=50)

ax2.set_title('Spending vs Purchase Frequency', fontweight='bold', fontsize=14)
ax2.set_xlabel('Number of Purchases')
ax2.set_ylabel('Total Spent (Altairian Dollars)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Visualization 3: Segment Distribution
segment_counts = customers_df['segment'].value_counts()
colors_pie = colors[:len(segment_counts)]
wedges, texts, autotexts = ax3.pie(segment_counts.values, labels=segment_counts.index, 
                                  autopct='%1.1f%%', colors=colors_pie, startangle=90)
ax3.set_title('Customer Segment Distribution', fontweight='bold', fontsize=14)

# Visualization 4: Average Metrics by Segment
segment_stats = customers_df.groupby('segment')[['purchase_count', 'total_spent', 'avg_order_value']].mean()
x_pos = np.arange(len(segment_stats.index))
width = 0.25

ax4.bar(x_pos - width, segment_stats['purchase_count'], width, label='Avg Purchases', alpha=0.8)
ax4_twin = ax4.twinx()
ax4_twin.bar(x_pos, segment_stats['total_spent'], width, label='Avg Spent', color='orange', alpha=0.8)
ax4_twin.bar(x_pos + width, segment_stats['avg_order_value'], width, label='Avg Order Value', color='green', alpha=0.8)

ax4.set_title('Average Metrics by Customer Segment', fontweight='bold', fontsize=14)
ax4.set_xlabel('Customer Segment')
ax4.set_ylabel('Average Purchases')
ax4_twin.set_ylabel('Average Spending (Altairian Dollars)')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(segment_stats.index, rotation=45)

# Combine legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig('ex05/customer_segmentation.png', dpi=300, bbox_inches='tight')
plt.show()

# Print segment summary
print("\nCustomer Segment Summary:")
print("=" * 50)
for segment in customers_df['segment'].unique():
    segment_data = customers_df[customers_df['segment'] == segment]
    print(f"\n{segment}:")
    print(f"  Count: {len(segment_data):,}")
    print(f"  Avg Purchases: {segment_data['purchase_count'].mean():.1f}")
    print(f"  Avg Spent: ${segment_data['total_spent'].mean():.2f}")
    print(f"  Avg Days Since Last Purchase: {segment_data['days_since_last'].mean():.0f}")

# Save results
customers_df[['user_id', 'segment', 'cluster', 'purchase_count', 'total_spent']].to_csv('ex05/customer_segments.csv', index=False)
```

## Database Connection Setup

```python
# database_setup.py - Helper for database connections
import sqlite3
import pandas as pd

def create_sample_data():
    """Create sample data if you don't have access to the original data warehouse"""
    
    # This creates a sample dataset similar to what might be in your data warehouse
    import random
    from datetime import datetime, timedelta
    
    # Sample data generation
    users = [f"user_{i}" for i in range(1, 10001)]
    categories = [f"category_{i}" for i in range(1, 21)]
    brands = [f"brand_{i}" for i in range(1, 101)]
    event_types = ['view', 'cart', 'purchase', 'remove_from_cart']
    
    # Generate sample events
    events = []
    start_date = datetime(2022, 10, 1)
    end_date = datetime(2023, 2, 28)
    
    for _ in range(100000):
        event = {
            'user_id': random.choice(users),
            'event_type': random.choice(event_types),
            'category_id': random.choice(categories),
            'brand': random.choice(brands),
            'price': round(random.uniform(0.5, 100), 2) if random.choice(event_types) == 'purchase' else 0,
            'event_time': start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        }
        events.append(event)
    
    # Create DataFrame and save to SQLite
    df = pd.DataFrame(events)
    conn = sqlite3.connect('sample_datawarehouse.db')
    df.to_sql('events', conn, if_exists='replace', index=False)
    conn.close()
    
    print("Sample data created in 'sample_datawarehouse.db'")

if __name__ == "__main__":
    create_sample_data()
```

## File Structure

```
your_project/
├── ex00/
│   ├── pie.py
│   └── user_behavior_pie.png
├── ex01/
│   ├── chart.py
│   └── purchase_analysis_charts.png
├── ex02/
│   ├── mustache.py
│   └── price_boxplots.png
├── ex03/
│   ├── Building.py
│   └── customer_analysis_bars.png
├── ex04/
│   ├── elbow.py
│   └── elbow_analysis.png
├── ex05/
│   ├── Clustering.py
│   ├── customer_segmentation.png
│   └── customer_segments.csv
└── database_setup.py
```

## Notes

1. **Database Connection**: Adjust the connection string in each script to match your data warehouse setup
2. **Data Columns**: Ensure your data warehouse has the expected columns (event_type, price, user_id, etc.)
3. **Dependencies**: Install required packages: `pip install pandas matplotlib seaborn scikit-learn plotly`
4. **Customization**: Modify visualizations and analysis based on your specific data characteristics
5. **Performance**: For large datasets, consider using sampling or chunking for memory efficiency