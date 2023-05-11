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
