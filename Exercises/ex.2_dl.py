
import numpy as np
import matplotlib.pyplot as plt


# Peaks function to generate target data for approximation
def peaks(X1, X2):
    """Computes the peaks function, a commonly used benchmark function for testing optimization algorithms.

    Args:
        X1 (ndarray): First input feature.
        X2 (ndarray): Second input feature.

    Returns:
        ndarray: The result of applying the peaks function on X1 and X2.
    """
    return 3 * (1 - X1) ** 2 * np.exp(-(X1 ** 2) - (X2 + 1) ** 2) - 10 * (X1 / 5 - X1 ** 3 - X2 ** 5) * np.exp(
        -X1 ** 2 - X2 ** 2) - 1 / 3 * np.exp(-(X1 + 1) ** 2 - X2 ** 2)


# Generate random data within the range [-3, 3]
def random_data(n_samples, n_features):
    """Generates a random dataset with values uniformly distributed between -3 and 3.

    Args:
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features for each sample.

    Returns:
        ndarray: Randomly generated dataset.
    """
    return np.random.uniform(low=-3, high=3, size=(n_samples, n_features))


# Derivative of the tanh activation function
def derivative_tanh(x):
    """Computes the derivative of the tanh activation function.

    Args:
        x (ndarray): Input array.

    Returns:
        ndarray: The derivative of tanh applied to the input.
    """
    return 1 - np.tanh(x) ** 2


# Neural network function with 3 layers (input, hidden, output)
def neural_network_3l(X, y, eta, n_hidden, max_iter=100000, error_interval=1000):
    """Implements a 3-layer neural network (1 hidden layer) with forward and backward propagation.

    Args:
        X (ndarray): Input data.
        y (ndarray): Target data.
        eta (float): Learning rate.
        n_hidden (int): Number of neurons in the hidden layer.
        max_iter (int): Maximum number of iterations (steps) to run the training loop.
        error_interval (int): Frequency to compute and record the generalization error.

    Returns:
        tuple: Final weights (W1, W2), generalization error, and error history.
    """
    samples = X.shape[0]  # Number of samples in the dataset
    input_layer = np.concatenate((np.ones((samples, 1)), X), axis=1)  # Add bias term to input layer
    W1 = np.random.randn(input_layer.shape[1], n_hidden)  # Initialize weights for input to hidden layer
    W2 = np.random.randn(n_hidden, 1)  # Initialize weights for hidden to output layer
    errors = []  # List to track error over time
    indices = np.random.permutation(samples)  # Shuffle the dataset indices
    generalization_error = 0  # Previous generalization error
    generalization_error_new = 0  # Current generalization error

    # Main training loop
    for counter in range(max_iter // samples + 1):
        for i in indices:
            # Forward pass for the selected sample
            h = np.dot(input_layer[i], W1)  # Compute hidden layer input
            hidden = np.tanh(h)  # Apply activation function (tanh)
            z = np.dot(hidden, W2)  # Compute output layer input
            output_neuron = np.tanh(z)  # Apply activation function (tanh)

            # Backward pass (calculate deltas)
            delta_2 = derivative_tanh(z) * (y[i] - output_neuron)  # Output layer delta
            delta_1 = derivative_tanh(h) * delta_2 * W2.T  # Hidden layer delta

            # Update weights using gradient descent
            W1 += eta * np.outer(input_layer[i], delta_1)
            W2 += eta * hidden[:, np.newaxis] * delta_2

        # Compute generalization error every error_interval steps
        if counter * samples % error_interval == 0 or counter == 0:
            generalization_error = generalization_error_new
            generalization_error_new = 0.5 * np.mean((y - np.tanh(np.dot(np.tanh(np.dot(input_layer, W1)), W2))) ** 2)
            errors.append(generalization_error_new)

        # Stop if the error converges
        if np.absolute(generalization_error_new - generalization_error) < 1e-6:
            print("Model converged")
            break

    return W1, W2, generalization_error_new, errors


# Function to run the model multiple times and track the best result
def run_model(n_trial, X, y, eta, n_hidden, max_iter):
    """Runs the neural network model multiple times and selects the best model based on generalization error.

    Args:
        n_trial (int): Number of times to run the model.
        X (ndarray): Input data.
        y (ndarray): Target data.
        eta (float): Learning rate.
        n_hidden (int): Number of neurons in the hidden layer.
        max_iter (int): Maximum number of iterations for each trial.

    Returns:
        tuple: The best model (weights and errors), and results of all trials.
    """
    results = []  # List to store the results of each trial

    for trial in range(n_trial):
        print(f"Trial: {trial + 1}")
        W1, W2, generalization_error, errors = neural_network_3l(X, y, eta, n_hidden, max_iter)
        results.append((W1, W2, generalization_error, errors))
        print(f"Generalization error: {generalization_error}\n")

    # Select the model with the lowest generalization error
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


# Generate dataset with 1000 samples and 2 features
n = 1000
X = random_data(n, 2)
y = peaks(X[:, 0], X[:, 1])

# Run the model 100 times with a learning rate of 0.001 and 100 hidden neurons
best_model, results = run_model(100, X, y, 0.001, 100, 10000)
min_W1, min_W2, min_generalization_error = best_model[:3]

# Prepare data for visualization (grid of input data)
X1 = X[:, 0]
X2 = X[:, 1]
X1_grid, X2_grid = np.meshgrid(X1, X2)

# Apply the trained model on the grid data
input_layer_grid = np.concatenate((np.ones((X1_grid.size, 1)), np.c_[X1_grid.ravel(), X2_grid.ravel()]), axis=1)
h_grid = np.dot(input_layer_grid, min_W1)
hidden_grid = np.tanh(h_grid)
z_grid = np.dot(hidden_grid, min_W2)
output_grid = np.tanh(z_grid).reshape(X1_grid.shape)

# Compute true peaks function on the grid
y_grid = peaks(X1_grid, X2_grid)

# Plot the true peaks function and the neural network's approximation
fig = plt.figure(figsize=(10, 5))

ax_1 = fig.add_subplot(121, projection='3d')
ax_1.plot_surface(X1_grid, X2_grid, y_grid, cmap='viridis')
ax_1.set_title("True Peaks Function")

ax_2 = fig.add_subplot(122, projection='3d')
ax_2.plot_surface(X1_grid, X2_grid, output_grid, cmap='viridis')
ax_2.set_title("Neural Network Approximation")

plt.show()
