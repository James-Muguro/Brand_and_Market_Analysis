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
