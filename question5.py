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

# Gradient Descent Function to Track Cost History
def gradient_descent(X, y, lr, iterations=50):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term
    theta = np.random.randn(n + 1, 1)  # Initialize random parameters
    cost_history = []

    for i in range(iterations):
        predictions = X_b.dot(theta)
        residuals = predictions - y
        gradients = 2 / m * X_b.T.dot(residuals)
        theta -= lr * gradients
        cost = np.mean(residuals ** 2)  # Mean squared error
        cost_history.append(cost)

    return theta, cost_history

# Test for different learning rates
learning_rates = [0.005, 0.5, 5]
iterations = 50

plt.figure(figsize=(10, 6))

for lr in learning_rates:
    _, cost_history = gradient_descent(X_normalized, y_normalized, lr, iterations)
    plt.plot(range(1, iterations + 1), cost_history, label=f'Learning Rate = {lr}')

# Customize plot
plt.title('Cost Function vs Iterations for Different Learning Rates', fontsize=14)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Cost (Mean Squared Error)', fontsize=12)
plt.legend()
plt.grid()
plt.show()
