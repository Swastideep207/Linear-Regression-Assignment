import numpy as np

# Generate synthetic data (replace this with your actual data)
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term
X_b = np.c_[np.ones((len(X), 1)), X]  # Add intercept (bias) term

# Initialize parameters (theta)
theta = np.random.randn(2, 1)

# Cost Function Without Averaging
def cost_function_no_average(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / 2) * np.sum((predictions - y) ** 2)  # No averaging
    return cost

# Cost Function With Averaging
def cost_function_with_average(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)  # With averaging
    return cost

# Compute costs
cost_no_average = cost_function_no_average(X_b, y, theta)
cost_with_average = cost_function_with_average(X_b, y, theta)

# Print results
print("Cost without averaging:", cost_no_average)
print("Cost with averaging:", cost_with_average)

# Advantage demonstration
print("\nAdvantage of averaging:")
print(f"- Without averaging, the cost scales with the dataset size. (Cost: {cost_no_average:.2f})")
print(f"- With averaging, the cost reflects the mean error per sample. (Cost: {cost_with_average:.4f})")
