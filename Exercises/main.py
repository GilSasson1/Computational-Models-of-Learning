import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def peaks(X1, X2):
    return 3 * (1 - X1) ** 2 * np.exp(-(X1 ** 2) - (X2 + 1) ** 2) - 10 * (X1 / 5 - X1 ** 3 - X2 ** 5) * np.exp(
        -X1 ** 2 - X2 ** 2) - 1 / 3 * np.exp(-(X1 + 1) ** 2 - X2 ** 2)


def random_data(n_samples, n_features):
    return np.random.uniform(low=-3, high=3, size=(n_samples, n_features))


def derivative_tanh(x):
    return 1 - np.tanh(x) ** 2


def neural_network_3l(X, y, eta, n_hidden, max_iter=100000, error_interval=1000):
    samples = X.shape[0]
    input_layer = np.concatenate((np.ones((samples, 1)), X), axis=1)  # Add bias term
    W1 = np.random.randn(input_layer.shape[1], n_hidden)
    W2 = np.random.randn(n_hidden, 1)
    errors = []
    indices = np.random.permutation(samples)
    generalization_error = 0
    generalization_error_new = 0

    for counter in range(max_iter // samples + 1):
        for i in indices:
            # Forward pass for the selected sample
            h = np.dot(input_layer[i], W1)
            hidden = np.tanh(h)
            z = np.dot(hidden, W2)
            output_neuron = np.tanh(z)

            # Calculate deltas
            delta_2 = derivative_tanh(z) * (y[i] - output_neuron)
            delta_1 = derivative_tanh(h) * delta_2 * W2.T

            # Update weights
            W1 += eta * np.outer(input_layer[i], delta_1)
            W2 += eta * hidden[:, np.newaxis] * delta_2

        if counter * samples % error_interval == 0 or counter == 0:
            generalization_error = generalization_error_new
            generalization_error_new = 0.5 * np.mean((y - np.tanh(np.dot(np.tanh(np.dot(input_layer, W1)), W2))) ** 2)
            errors.append(generalization_error_new)

        # Stop if the error converges
        if np.absolute(generalization_error_new - generalization_error) < 1e-6:
            print("Model converged")
            break

    return W1, W2, generalization_error_new, errors


def run_model(n_trial, X, y, eta, n_hidden, max_iter):
    results = []

    for trial in range(n_trial):
        print(f"Trial: {trial + 1}")
        W1, W2, generalization_error, errors = neural_network_3l(X, y, eta, n_hidden, max_iter)
        results.append((W1, W2, generalization_error, errors))
        print(f"Generalization error: {generalization_error}\n")

    min_error_model = min(results, key=lambda x: x[2])
    min_W1, min_W2, min_generalization_error, min_errors = min_error_model
    print(f"The lowest Generalization error is: {min_generalization_error}")

    # Plot the errors for the best model
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(min_errors) * 1000, 1000), min_errors, marker='o', linestyle='-')

    plt.xlabel('Iterations')
    plt.ylabel('Generalization Error')
    plt.title('Error Rate for Best Model')

    plt.show()

    return min_error_model, results


# Generate alphabet once
n = 1000
X = random_data(n, 2)
y = peaks(X[:, 0], X[:, 1])

# Run
best_model, results = run_model(100, X, y, 0.001, 100, 10000)
min_W1, min_W2, min_generalization_error = best_model[:3]

# Prepare alphabet for plotting
X1 = X[:, 0]
X2 = X[:, 1]
X1_grid, X2_grid = np.meshgrid(X1, X2)

input_layer_grid = np.concatenate((np.ones((X1_grid.size, 1)), np.c_[X1_grid.ravel(), X2_grid.ravel()]), axis=1)
h_grid = np.dot(input_layer_grid, min_W1)
hidden_grid = np.tanh(h_grid)
z_grid = np.dot(hidden_grid, min_W2)
output_grid = np.tanh(z_grid).reshape(X1_grid.shape)

# Reshape y for the grid
y_grid = peaks(X1_grid, X2_grid)

# Plot
fig = plt.figure(figsize=(10, 5))

ax_1 = fig.add_subplot(121, projection='3d')
ax_1.plot_surface(X1_grid, X2_grid, y_grid, cmap='viridis')
ax_1.set_title("True Peaks Function")

ax_2 = fig.add_subplot(122, projection='3d')
ax_2.plot_surface(X1_grid, X2_grid, output_grid, cmap='viridis')
ax_2.set_title("Neural Network Approximation")

plt.show()
