import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_clustering_and_visualization(data_file, num_clusters, countries_to_compare):

    """
    1st Docstring
    Performs clustering and visualization.

    Args:
        data_file (str): The path to the CSV file containing COVID-19 data.
        num_clusters (int): The number of clusters to create.
        countries_to_compare (list): A list of tuples specifying the countries and cluster IDs to compare.

    Returns:
        None
    """

    """
    2nd Docstring
    Performs clustering and visualization.

    This function reads the provided CSV file containing COVID-19 data, performs clustering using KMeans algorithm,
    and visualizes the clusters on a scatter plot. It also compares countries within clusters based on selected
    variables, identifies one country from each cluster, and performs additional analysis and visualization.

    Args:
        data_file (str): The path to the CSV file containing COVID-19 data.
        num_clusters (int): The number of clusters to create.
        countries_to_compare (list): A list of tuples specifying the countries and cluster IDs to compare.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified data file is not found.
    """


    # Read the data
    df = pd.read_csv(data_file)

    df = df.transpose()

    # Select the columns for clustering
    columns_for_clustering = df.columns[4:-1]

    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[columns_for_clustering])

    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)

    # Create a scatter plot of clusters
    plt.scatter(df['Long'], df['Lat'], c=df['Cluster'], cmap='viridis')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Country Clusters')
    plt.show()

    def compare_countries(df, cluster_id, countries):
        # Select the countries from the specified cluster
        cluster_countries = df[df['Cluster'] == cluster_id]['Country/Region']
        # Iterate over the selected countries and compare them
        for country in countries:
            country_data = df[df['Country/Region'] == country][columns_for_clustering]
            cluster_data = df[df['Country/Region'].isin(cluster_countries)][columns_for_clustering]

            # Perform comparison and analysis for the selected countries and the cluster

            # Example: Calculate mean values
            country_mean = country_data.mean()
            cluster_mean = cluster_data.mean()

            # Example: Calculate differences
            differences = country_mean - cluster_mean

            # Example: Print results
            print(f"Comparison for {country} and Cluster {cluster_id}:")
            print("Country Mean Values:")
            print(country_mean)
            print("Cluster Mean Values:")
            print(cluster_mean)
            print("Differences:")
            print(differences)
            print("\n")

    # Compare countries within clusters
    for country, cluster_id in countries_to_compare:
        compare_countries(df, cluster_id, [country])

    # Identify one country from each cluster
    cluster_centers = df.groupby('Cluster')[['Long', 'Lat']].mean()
    countries_per_cluster = []
    for cluster_id in df['Cluster'].unique():
        cluster_center = cluster_centers.loc[cluster_id]
        country = df.loc[(df['Cluster'] == cluster_id), 'Country/Region'].values[0]
        countries_per_cluster.append((country, cluster_center))

    # Compare countries within each cluster
    for country, cluster_center in countries_per_cluster:
        cluster_id = df.loc[(df['Country/Region'] == country), 'Cluster'].values[0]
        compare_countries(df, cluster_id, [country])

    # Additional Analysis and Visuals

    # Investigate trends within clusters
    for cluster_id in df['Cluster'].unique():
        cluster_countries = df[df['Cluster'] == cluster_id]['Country/Region']
        cluster_data = df[df['Country/Region'].isin(cluster_countries)]

        # Check if 'Date' column exists in the DataFrame
        if 'Date' in cluster_data.columns:
            # Plot trends within the cluster
            plt.figure(figsize=(12, 6))
            for country in cluster_countries:
                country_data = cluster_data[cluster_data['Country/Region'] == country]
                plt.plot(country_data['Date'], country_data['Confirmed'], label=country)
            plt.xlabel('Date')
            plt.ylabel('Confirmed Cases')
            plt.title(f'Cluster {cluster_id} - Confirmed Cases Trend')
            plt.legend()
            plt.xticks
            plt.xticks(rotation=45)
            plt.show()
        else:
            pass

    # Scatter plot of clusters with cluster centers
    plt.scatter(df['Long'], df['Lat'], c=df['Cluster'], cmap='viridis')
    plt.scatter(cluster_centers['Long'], cluster_centers['Lat'], c='red', marker='X', s=100, label='Cluster Centers')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Country Clusters with Cluster Centers')
    plt.legend()
    plt.show()

    # Histogram of cluster distribution
    plt.hist(df['Cluster'], bins=len(df['Cluster'].unique()))
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title('Cluster Distribution')
    plt.show()

    # Calculate the proportion of countries in each cluster
    cluster_proportions = df['Cluster'].value_counts(normalize=True)

    # Pie chart of cluster proportions
    plt.pie(cluster_proportions, labels=cluster_proportions.index, autopct='%1.1f%%')
    plt.title('Cluster Proportions')
    plt.show()
            

# Set the path to your data file
data_file = 'covid_19_data.csv'

# Specify the number of clusters
num_clusters = 3

# Specify the countries to compare within clusters
countries_to_compare = [('Country1', 0), ('Country2', 1)]  # Example countries and cluster IDs

# Call the function to perform clustering and visualization
perform_clustering_and_visualization(data_file, num_clusters, countries_to_compare)
