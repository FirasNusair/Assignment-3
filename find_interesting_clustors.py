import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_clustering(data_file):
    """Performs clustering on a dataset to find interesting clusters of data based on CO2 emissions.

    This function reads the data from a CSV file, selects relevant columns for clustering, and preprocesses the data.
    It applies a clustering method (in this case, K-means) to identify clusters based on CO2 emissions.
    The function then adds the cluster labels to the DataFrame and produces a plot showing the cluster membership
    and cluster centers."""

    """Doc 2: 
    Args:
        data_file (str): The path to the CSV file containing the data.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified data file is not found.
    """

    
    # Read the data into a DataFrame
    data = pd.read_csv(data_file)

    # Select relevant columns for clustering
    columns_to_cluster = ['1960', '1961', '1962', '1963', '1964', '1965', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
    df = data[columns_to_cluster].copy()

    # Clean and preprocess the data
    df = df.replace('NaN', np.nan)
    df = df.replace(',', '', regex=True).astype(float)
    df = df.dropna()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Perform clustering using K-means
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df_scaled)
    cluster_labels = kmeans.labels_
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # Add cluster labels to the DataFrame
    df['Cluster'] = cluster_labels

    # Plot the cluster membership and cluster centers
    plt.figure(figsize=(12, 6))
    plt.scatter(df.index, df['2020'], c=df['Cluster'], cmap='viridis', alpha=0.7)
    plt.scatter(range(len(cluster_centers)), cluster_centers[:, -1], c='red', marker='X', s=100, label='Cluster Centers')
    plt.xlabel('Country')
    plt.ylabel('CO2 emissions (kt) in 2020')
    plt.title('Clustering of Countries based on CO2 emissions')
    plt.legend()
    plt.show()

# Example usage:
data_file = 'Complete DS 2.csv'
perform_clustering(data_file)
