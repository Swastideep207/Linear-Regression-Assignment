import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data (replace with your actual data)
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Normalize the data
X_mean, X_std = np.mean(X), np.std(X)
X_normalized = (X - X_mean) / X_std
y_mean, y_std = np.mean(y), np.std(y)
y_normalized = (y - y_mean) / y_std

# Gradient Descent Function
def gradient_descent(X, y, lr=0.5, iterations=50):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term (intercept)
    theta = np.random.randn(n + 1, 1)  # Initialize random parameters
    for _ in range(iterations):
        predictions = X_b.dot(theta)
        residuals = predictions - y
        gradients = 2 / m * X_b.T.dot(residuals)
        theta -= lr * gradients
    return theta

# Train the model
learning_rate = 0.5
iterations = 50
theta = gradient_descent(X_normalized, y_normalized, lr=learning_rate, iterations=iterations)

# Predict line values
X_b = np.c_[np.ones((len(X_normalized), 1)), X_normalized]  # Add bias term
predictions = X_b.dot(theta)

# Plot the dataset and the regression line
plt.figure(figsize=(8, 5))
plt.scatter(X_normalized, y_normalized, color='blue', label='Data')  # Scatter plot of data
plt.plot(X_normalized, predictions, color='red', label='Regression Line')  # Regression line
plt.title('Dataset and Fitted Regression Line', fontsize=14)
plt.xlabel('Normalized X', fontsize=12)
plt.ylabel('Normalized y', fontsize=12)
plt.legend()
plt.grid()
plt.show()

# Print theta values
print("Obtained Regression Parameters (Theta):")
print("Intercept (theta_0):", theta[0][0])
print("Slope (theta_1):", theta[1][0])
