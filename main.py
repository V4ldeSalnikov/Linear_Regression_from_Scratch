import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class CustomLinearRegression:
    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))
        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
        if self.fit_intercept:
            self.intercept = self.coefficients[0]
            self.coefficients = self.coefficients[1:]
        else:
            self.intercept = 0

    def predict(self, X):
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))
        return X @ np.concatenate(([self.intercept], self.coefficients))

    def r2_score(self, y, y_pred):
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

    def rmse(self, y, y_pred):
        return np.sqrt(np.mean((y - y_pred) ** 2))

# Prepare the data
f1 = [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87]
f2 = [65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 96.1, 100.0, 85.9, 94.3]
f3 = [15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2]
y = [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.0]
X = np.column_stack((f1, f2, f3))
y = np.array(y)

# Fit the CustomLinearRegression model
custom_model = CustomLinearRegression()
custom_model.fit(X, y)
custom_y_pred = custom_model.predict(X)
custom_r2 = custom_model.r2_score(y, custom_y_pred)
custom_rmse = custom_model.rmse(y, custom_y_pred)

# Fit the sklearn LinearRegression model for comparisson
regSci = LinearRegression()
regSci.fit(X, y)
sci_y_pred = regSci.predict(X)
sci_r2 = r2_score(y, sci_y_pred)
sci_rmse = np.sqrt(mean_squared_error(y, sci_y_pred))

# Comparing the results
comparison = {
    'Intercept': regSci.intercept_ - custom_model.intercept,
    'Coefficient': regSci.coef_ - custom_model.coefficients,
    'R2': sci_r2 - custom_r2,
    'RMSE': sci_rmse - custom_rmse
}

# Print the comparison
print(comparison)
