import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t


def fit_and_predict(data, degree, future_years, alpha=0.05):

    """
        Fits a simple model to the data using curve_fit and generates predictions for future time points.

    This function fits a simple model to the given data using curve_fit function from scipy.optimize.
    The model can represent time series or a relationship between two attributes, such as exponential growth,
    logistic function, or low order polynomials. The function utilizes the attached err_ranges function
    to estimate lower and upper limits of the confidence range for the predictions.
    """
     
    """    
    Doc 2:
    Args:
        data (str): The path to the CSV file containing the data.
        degree (int): The degree of the polynomial or complexity of the model.
        future_years (int): The number of years to predict into the future.
        alpha (float, optional): The significance level for confidence intervals. Default is 0.05.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified data file is not found."""


    # Step 2: Read the data sample into a pandas DataFrame
    df = pd.read_csv(data)
    
    # Step 3: Define the model(s) to fit the data
    def model(x, *params):
        return np.polyval(params, x)
    
    # Step 4: Implement the `err_ranges` function to estimate confidence intervals
    def err_ranges(y, popt, pcov, alpha=0.05):
        perr = np.sqrt(np.diag(pcov))
        tval = np.abs(t.ppf(alpha / 2, len(y) - len(popt)))
        return tval * perr
    
    # Step 5: Fit the model to the data and obtain the best-fitting parameters
    x = df.columns[4:].astype(int)
    y = df.iloc[0, 4:].str.replace(',', '').astype(float)
    
    # Handle missing values by removing rows with NaN values
    valid_mask = ~np.isnan(y)
    x = x[valid_mask]
    y = y[valid_mask]
    
    # Fit the polynomial model with initial parameter values
    initial_params = np.ones(degree + 1)
    popt, pcov = curve_fit(model, x, y, p0=initial_params)
    
    # Step 6: Generate predictions for future time points, including confidence ranges
    future_x = np.arange(np.max(x), np.max(x) + future_years + 1)
    predicted_y = model(future_x, *popt)
    
    # Generate confidence range for predictions
    y_hat = model(x, *popt)
    residual = y - y_hat
    sigma = np.std(residual)
    dof = len(x) - len(popt)
    confidence_range = t.ppf(1 - alpha / 2, dof) * sigma * np.sqrt(1 + 1 / len(x) + (future_x - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))

    lower_bound = predicted_y - confidence_range
    upper_bound = predicted_y + confidence_range

    # Step 7: Plot the best-fitting function along with the confidence range
    plt.plot(x, y, 'bo', label='Actual Data')
    plt.plot(future_x, predicted_y, 'r-', label='Best Fit')
    plt.fill_between(future_x, lower_bound, upper_bound, alpha=0.3, label='Confidence Range')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title('Model Fit and Predictions')
    plt.legend()
    plt.show()


data_file = 'Complete DS 2.csv'
degree = 10  # Adjust the degree to increase model complexity

fit_and_predict(data_file, degree, future_years=20)



# Cluster of countries

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

    df.transpose()

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
