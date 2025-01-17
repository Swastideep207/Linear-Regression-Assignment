# Answer the following questions based on your observations .... 
# 1. Use linear regression to fit a straight line to the given database.
#  Set your learning rate to 0.5. 
# What are the cost function value and learning parameters values after convergence? 
# Also, mention the convergence criteria you used.


import numpy as np

# Generate synthetic data (replace with actual data)
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Normalize data
X_mean, X_std = np.mean(X), np.std(X)
X_normalized = (X - X_mean) / X_std
y_mean, y_std = np.mean(y), np.std(y)
y_normalized = (y - y_mean) / y_std

# Gradient Descent Function
def gradient_descent(X, y, lr=0.5, max_iterations=1000, tolerance=1e-6):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term
    theta = np.random.randn(n + 1, 1)  # Initialize parameters
    cost_history = []
    prev_cost = float('inf')

    for iteration in range(max_iterations):
        predictions = X_b.dot(theta)
        residuals = predictions - y
        gradients = 2 / m * X_b.T.dot(residuals)
        theta -= lr * gradients
        cost = np.mean(residuals ** 2)  # Mean Squared Error
        cost_history.append(cost)
        
        # Check convergence
        if abs(prev_cost - cost) < tolerance:
            break
        prev_cost = cost

    return theta, cost, iteration, cost_history

# Train model
learning_rate = 0.5
theta, final_cost, iterations, cost_history = gradient_descent(X_normalized, y_normalized, lr=learning_rate)

# Print results
print("Final Parameters (Theta):", theta.flatten())
print("Final Cost Function Value:", final_cost)
print("Convergence Criteria: Change in cost < 1e-6")
print("Iterations for Convergence:", iterations)
