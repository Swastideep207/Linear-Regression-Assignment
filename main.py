import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data (replace this with your actual dataset)
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Normalize data
X_mean, X_std = np.mean(X), np.std(X)
X_normalized = (X - X_mean) / X_std
y_mean, y_std = np.mean(y), np.std(y)
y_normalized = (y - y_mean) / y_std

# Gradient Descent Function
def gradient_descent(X, y, lr=0.5, iterations=100):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term (intercept)
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

# Train model with a learning rate of 0.5
lr = 0.5
iterations = 50
theta, cost_history = gradient_descent(X_normalized, y_normalized, lr, iterations)

# Plot Cost vs Iterations
plt.plot(range(1, len(cost_history) + 1), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations')
plt.show()

# Plot data and regression line
X_b = np.c_[np.ones((len(X_normalized), 1)), X_normalized]
predictions = X_b.dot(theta)

plt.scatter(X_normalized, y_normalized, color='blue', label='Data')
plt.plot(X_normalized, predictions, color='red', label='Regression Line')
plt.xlabel('Normalized X')
plt.ylabel('Normalized y')
plt.legend()
plt.title('Dataset and Regression Line')
plt.show()

# Experiment with other learning rates
for lr in [0.005, 0.5, 5]:
    theta, cost_history = gradient_descent(X_normalized, y_normalized, lr, iterations)
    plt.plot(range(1, len(cost_history) + 1), cost_history, label=f'lr={lr}')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations for Different Learning Rates')
plt.legend()
plt.show()

# Implement stochastic and mini-batch gradient descent
def stochastic_gradient_descent(X, y, lr=0.01, iterations=50):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(n + 1, 1)
    cost_history = []

    for iteration in range(iterations):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            theta -= lr * gradients
        residuals = X_b.dot(theta) - y
        cost = np.mean(residuals ** 2)
        cost_history.append(cost)
    return theta, cost_history

def mini_batch_gradient_descent(X, y, lr=0.01, iterations=50, batch_size=20):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(n + 1, 1)
    cost_history = []

    for iteration in range(iterations):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, batch_size):
            xi = X_b_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            gradients = 2 / batch_size * xi.T.dot(xi.dot(theta) - yi)
            theta -= lr * gradients
        residuals = X_b.dot(theta) - y
        cost = np.mean(residuals ** 2)
        cost_history.append(cost)
    return theta, cost_history

# Compare methods
theta_sgd, cost_history_sgd = stochastic_gradient_descent(X_normalized, y_normalized, lr=0.01, iterations=50)
theta_mbgd, cost_history_mbgd = mini_batch_gradient_descent(X_normalized, y_normalized, lr=0.01, iterations=50)

plt.plot(range(1, len(cost_history) + 1), cost_history, label='Batch Gradient Descent')
plt.plot(range(1, len(cost_history_sgd) + 1), cost_history_sgd, label='Stochastic Gradient Descent')
plt.plot(range(1, len(cost_history_mbgd) + 1), cost_history_mbgd, label='Mini-Batch Gradient Descent')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.legend()
plt.title('Cost vs Iterations for Different Methods')
plt.show()
