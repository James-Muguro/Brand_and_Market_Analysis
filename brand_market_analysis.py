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
import statsmodels.api as sm

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

# Problem Questions

# 1. Which brand has the highest performance in terms of sales volume and net sales across different stores?

# Group the data by 'Brand'
grouped = df.groupby('Brand')

# Calculate total 'Volume' and 'Net Sales' for each brand
total_volume = grouped['Volume'].sum()
total_net_sales = grouped['Net Sales'].sum()

# Find the brand with the highest total volume and net sales
highest_volume_brand = total_volume.idxmax()
highest_net_sales_brand = total_net_sales.idxmax()

# Print the results
print("Brand with the Highest Sales Volume:\n", highest_volume_brand)
print("\nBrand with the Highest Net Sales:\n", highest_net_sales_brand)

# 2. Which client type (Store) and client (name of store) generates the highest sales volume and net sales for different brands?

# Group the data by 'Brand', 'Client Type', and 'Client'
grouped = df.groupby(['Brand', 'Client Type', 'Client'])

# Calculate total 'Volume' and 'Net Sales' for each group
total_volume = grouped['Volume'].sum()
total_net_sales = grouped['Net Sales'].sum()

# Find the client type and client with the highest total volume and net sales for each brand
highest_volume_client = total_volume.idxmax()
highest_net_sales_client = total_net_sales.idxmax()

# Print the results
print("Client Type and Client with the Highest Sales Volume for each Brand:\n", highest_volume_client)
print("\nClient Type and Client with the Highest Net Sales for each Brand:\n", highest_net_sales_client)

# 3. How do discounts impact the sales volume of each brand in each Store (Client Type, and Client)?

# Group the data by 'Brand', 'Client Type', and 'Client'
grouped = df.groupby(['Brand', 'Client Type', 'Client'])

# Calculate correlation between 'Discounts' and 'Volume' for each group
correlations = grouped.apply(lambda x: x['Discounts'].corr(x['Volume']))

# Print the results
print("Correlation between Discounts and Volume for each Brand, Client Type, and Client:\n", correlations)

# 4. Which brand and Store have the most cost-efficient operations?

# Group the data by 'Brand', 'Client Type', and 'Client'
grouped = df.groupby(['Brand', 'Client Type', 'Client'])

# Calculate total costs (COGS, distribution, and warehousing) for each group
total_costs = grouped[['Cost of Goods Sold', 'Distribution', 'Warehousing']].sum().sum(axis=1)

# Calculate total net sales for each group
total_net_sales = grouped['Net Sales'].sum()

# Calculate the ratio of total costs to net sales
cost_efficiency = total_costs / total_net_sales

# Find the brand and store with the most cost-efficient operations
most_cost_efficient = cost_efficiency.idxmin()

# Print the results
print("Brand and Store with the Most Cost-Efficient Operations:\n", most_cost_efficient)

# 5. What are the most and least popular product sizes or packs for each brand and store?

# Group the data by 'Brand', 'Client Type', 'Client', and 'Pack'
grouped_pack = df.groupby(['Brand', 'Client Type', 'Client', 'Pack'])

# Calculate total 'Volume' for each group
total_volume_pack = grouped_pack['Volume'].sum()

# Find the most and least popular product pack for each brand and store
most_popular_pack = total_volume_pack.idxmax()
least_popular_pack = total_volume_pack.idxmin()

# Print the results
print("Most Popular Product Pack for each Brand and Store:\n", most_popular_pack)
print("\nLeast Popular Product Pack for each Brand and Store:\n", least_popular_pack)

# Visualizing Key Metrics

# 1. Brand Perfomance

# Group the data by 'Brand'
grouped = df.groupby('Brand')

# Calculate total 'Volume' and 'Net Sales' for each brand
total_volume = grouped['Volume'].sum()
total_net_sales = grouped['Net Sales'].sum()

# Plot total 'Volume' and 'Net Sales' for each brand
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
total_volume.plot(kind='bar', ax=axes[0], color='blue')
axes[0].set_title('Total Sales Volume for each Brand')
axes[0].set_ylabel('Total Sales Volume')
total_net_sales.plot(kind='bar', ax=axes[1], color='green')
axes[1].set_title('Total Net Sales for each Brand')
axes[1].set_ylabel('Total Net Sales')
plt.tight_layout()
plt.show()

# 2. Store Performance

# Group the data by 'Client Type'
grouped = df.groupby('Client Type')

# Calculate total 'Volume' and 'Net Sales' for each client type
total_volume = grouped['Volume'].sum()
total_net_sales = grouped['Net Sales'].sum()

# Plot total 'Volume' and 'Net Sales' for each client type
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
total_volume.plot(kind='bar', ax=axes[0], color='blue')
axes[0].set_title('Total Sales Volume for each Client Type')
axes[0].set_ylabel('Total Sales Volume')
total_net_sales.plot(kind='bar', ax=axes[1], color='green')
axes[1].set_title('Total Net Sales for each Client Type')
axes[1].set_ylabel('Total Net Sales')
plt.tight_layout()
plt.show()

# 3. Discount Impact

# Calculate correlation between 'Discounts' and 'Volume' for each brand
grouped = df.groupby('Brand')
correlations = grouped.apply(lambda x: x['Discounts'].corr(x['Volume']))

# Plot correlation between 'Discounts' and 'Volume' for each brand
correlations.plot(kind='barh', figsize=(10, 6), color='purple')
plt.title('Correlation between Discounts and Sales Volume for each Brand')
plt.xlabel('Correlation')  # Change ylabel to xlabel
plt.show()

# 4. Cost Efficiency

# Group the data by 'Brand'
grouped = df.groupby('Brand')

# Calculate total costs (COGS, distribution, and warehousing) for each brand
total_costs = grouped[['Cost of Goods Sold', 'Distribution', 'Warehousing']].sum().sum(axis=1)

# Calculate total net sales for each brand
total_net_sales = grouped['Net Sales'].sum()

# Calculate the ratio of total costs to net sales
cost_efficiency = total_costs / total_net_sales

# Plot cost efficiency for each brand (horizontal bar chart)
cost_efficiency.plot(kind='barh', figsize=(10, 6), color='orange')
plt.title('Cost Efficiency for each Brand')
plt.xlabel('Cost Efficiency')  # Change ylabel to xlabel
plt.show()

# Models

# 1. Regression Model

# Calculate total costs (COGS, distribution, and warehousing)
df['Total Costs'] = df['Cost of Goods Sold'] + df['Distribution'] + df['Warehousing']

# Define the independent variables (Discounts and Total Costs) and the dependent variable (Net Sales)
X = df[['Discounts', 'Total Costs']]
y = df['Net Sales']

# Add a constant to the independent variables matrix
X = sm.add_constant(X)

# Fit the ordinary least squares (OLS) model
model = sm.OLS(y, X)
results = model.fit()

# Print the summary statistics of the regression model
print(results.summary())

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each independent variable
X = df[['Discounts', 'Total Costs']]
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

print(vif)

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Define the independent variables (Discounts and Total Costs) and the dependent variable (Net Sales)
X = df[['Discounts', 'Total Costs']]
y = df['Net Sales']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Ridge Regression model
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Print the coefficients of the model
print("Coefficients: ", ridge.coef_)

# 2. Time Series

# Plotting Gross Sales to visualize trends and seasonality.
plt.figure(figsize=(10, 6))
plt.plot(df['Gross Sales'], marker='o', label='Gross Sales')
plt.title('Gross Sales Time Series')
plt.xlabel('Period')
plt.ylabel('Gross Sales')
plt.grid(True)
plt.show()

# Add a small constant to avoid zero and negative values
small_constant = 1e-10
adjusted_gross_sales = df['Gross Sales'] + small_constant

# Using seasonal_decompose from statsmodels to observe trend and seasonality more clearly.
result = seasonal_decompose(adjusted_gross_sales, model='multiplicative', period=12)

# Customize the plot layout for better visibility
plt.figure(figsize=(12, 8))

# Original time series
plt.subplot(4, 1, 1)
plt.plot(adjusted_gross_sales, label='Original')
plt.title('Original Time Series')
plt.legend()

# Trend component
plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Trend')
plt.title('Trend Component')
plt.legend()

# Seasonal component
plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Seasonal')
plt.title('Seasonal Component')
plt.legend()

# Residual component
plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Residual')
plt.title('Residual Component')
plt.legend()

# Adjust layout for better spacing
plt.tight_layout()
plt.show()



