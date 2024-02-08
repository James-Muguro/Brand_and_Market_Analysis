# To start working with the dataset, we will import the following libraries

import pandas as pd  # Pandas for data manipulation and analysis
import numpy as np   # NumPy for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for basic data visualization
import seaborn as sns  # Seaborn for advanced data visualization
from sklearn.cluster import KMeans  # KMeans for clustering
from sklearn.preprocessing import StandardScaler  # StandardScaler for feature scaling
from sklearn.model_selection import train_test_split  # Train-test split for model evaluation
from sklearn.linear_model import LogisticRegression  # Logistic Regression for classification
from sklearn.metrics import f1_score, confusion_matrix  # Metrics for model evaluation
from sklearn.svm import SVC  # Support Vector Classifier for classification
import ipywidgets as widgets
from IPython.display import display, clear_output
from tkinter import Tk, filedialog
import numpy_financial as npf

# Load the data

def select_file(b):
    clear_output()
    root = Tk()
    root.withdraw()  # Hide the main window
    root.call('wm', 'attributes', '.', '-topmost', True)  # Raise the root to the top of all windows
    b.files = filedialog.askopenfilename(multiple=False)  # List of selected files
    path = b.files
    global df
    df = pd.read_excel(path)
    print(f'Loaded dataframe from {path}')
    display(df.head())

fileselect = widgets.Button(description="File select")
fileselect.on_click(select_file)

display(fileselect)
df.head()

df.dtypes

df.shape

# Convert 'Period' to datetime format
df['Period'] = pd.to_datetime(df['Period'], format='%Y%m')

# Set 'Period' as the index of the DataFrame
df.set_index('Period', inplace=True)

# Create line plots for time-series data
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

sns.lineplot(data=df['Gross Sales'], ax=axes[0], color='blue')
axes[0].set_title('Gross Sales over time')
axes[0].set_ylabel('Gross Sales')

sns.lineplot(data=df['Net Sales'], ax=axes[1], color='green')
axes[1].set_title('Net Sales over time')
axes[1].set_ylabel('Net Sales')

plt.tight_layout()
plt.show()

# Create histograms for distribution of continuous data
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

# Histogram for Gross Sales
axes[0].hist(df['Gross Sales'], bins=30, color='blue', alpha=0.7, density=True)
axes[0].set_title('Distribution of Gross Sales')
axes[0].set_xlabel('Gross Sales')
axes[0].set_ylabel('Density')

# Histogram for Net Sales
axes[1].hist(df['Net Sales'], bins=30, color='green', alpha=0.7, density=True)
axes[1].set_title('Distribution of Net Sales')
axes[1].set_xlabel('Net Sales')
axes[1].set_ylabel('Density')

plt.tight_layout()
plt.show()

# Brand Performance:

# Group the data by 'Brand'
grouped = df.groupby('Brand')

# Calculate total and average 'Volume' and 'Net Sales' for each brand
total_volume = grouped['Volume'].sum()
average_volume = grouped['Volume'].mean()
total_net_sales = grouped['Net Sales'].sum()
average_net_sales = grouped['Net Sales'].mean()

# Print the results
print("Total Volume per Brand:\n", total_volume)
print("\nAverage Volume per Brand:\n", average_volume)
print("\nTotal Net Sales per Brand:\n", total_net_sales)
print("\nAverage Net Sales per Brand:\n", average_net_sales)

# Supermarket Performance

# Filter the data for supermarkets only
df_supermarkets = df[df['Client Type'] == 'Supermarkets']

# Group the data by 'Client'
grouped = df_supermarkets.groupby('Client')

# Calculate total and average 'Volume' and 'Net Sales' for each supermarket
total_volume = grouped['Volume'].sum()
average_volume = grouped['Volume'].mean()
total_net_sales = grouped['Net Sales'].sum()
average_net_sales = grouped['Net Sales'].mean()

# Print the results
print("Total Volume per Supermarket:\n", total_volume)
print("\nAverage Volume per Supermarket:\n", average_volume)
print("\nTotal Net Sales per Supermarket:\n", total_net_sales)
print("\nAverage Net Sales per Supermarket:\n", average_net_sales)

# All retail stores Perfomance

# Group the data by 'Client Type'
grouped = df.groupby('Client Type')

# Calculate total and average 'Volume' and 'Net Sales' for each client type
total_volume = grouped['Volume'].sum()
average_volume = grouped['Volume'].mean()
total_net_sales = grouped['Net Sales'].sum()
average_net_sales = grouped['Net Sales'].mean()

# Print the results
print("Total Volume per Client Type:\n", total_volume)
print("\nAverage Volume per Client Type:\n", average_volume)
print("\nTotal Net Sales per Client Type:\n", total_net_sales)
print("\nAverage Net Sales per Client Type:\n", average_net_sales)

# Discount Impact

# Group the data by 'Brand' and 'Client'
grouped = df.groupby(['Brand', 'Client'])

# Calculate correlation between 'Discounts' and 'Volume' for each group
correlations = grouped.apply(lambda x: x['Discounts'].corr(x['Volume']))

# Print the results
print("Correlation between Discounts and Volume for each Brand and Supermarket:\n", correlations)

# Cost Efficiency

# Group the data by 'Brand' and 'Client Type'
grouped = df.groupby(['Brand', 'Client Type'])

# Calculate total costs (COGS, distribution, and warehousing) for each group
total_costs = grouped[['Cost of Goods Sold', 'Distribution', 'Warehousing']].sum().sum(axis=1)

# Calculate total net sales for each group
total_net_sales = grouped['Net Sales'].sum()

# Calculate the ratio of total costs to net sales
cost_efficiency = total_costs / total_net_sales

# Print the results
print("Cost Efficiency for each Brand and Store:\n", cost_efficiency)

# Product Preference:

# Option A.
# Group the data by 'Brand', 'Client', and 'Size'
grouped_size = df.groupby(['Brand', 'Client', 'Size'])

# Calculate total 'Volume' for each group
total_volume_size = grouped_size['Volume'].sum()

# Find the most and least popular product size for each brand and supermarket
most_popular_size = total_volume_size.idxmax()
least_popular_size = total_volume_size.idxmin()

# Print the results
print("Most Popular Product Size for each Brand and Store:\n", most_popular_size)
print("\nLeast Popular Product Size for each Brand and Store:\n", least_popular_size)

# Option B.
# Group the data by 'Brand', 'Client', and 'Pack'
grouped_pack = df.groupby(['Brand', 'Client', 'Pack'])

# Calculate total 'Volume' for each group
total_volume_pack = grouped_pack['Volume'].sum()

# Find the most and least popular product pack for each brand and supermarket
most_popular_pack = total_volume_pack.idxmax()
least_popular_pack = total_volume_pack.idxmin()

# Print the results
print("Most Popular Product Pack for each Brand and Store:\n", most_popular_pack)
print("\nLeast Popular Product Pack for each Brand and Store:\n", least_popular_pack)

