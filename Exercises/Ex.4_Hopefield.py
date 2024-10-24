import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Load the alphabet alphabet
file_path = ('/Users/gilsasson/Library/CloudStorage/OneDrive-mail.tau.ac.il/פסיכולוגיה/שנה ג/'
             'סמסטר ב/מודלים חישוביים של למידה/Assignment 4/alphabet.xlsx') # Path to the alphabet file
alphabet = pd.read_excel(file_path, header=None)
alphabet = alphabet.values

# Convert 0/1 values to -1/1 values
alphabet = alphabet * 2 - 1

# Extract the vector for the letter 'G'
letter_index = 6  # 'G' is the 7th letter, index 6 in 0-based indexing
letter_vector_g = alphabet[:, letter_index]

# Calculate the weight matrix
n_neurons = letter_vector_g.size
weight_matrix = np.outer(letter_vector_g, letter_vector_g) / n_neurons
np.fill_diagonal(weight_matrix, 0)


# Run the simulation
def hopfield_update(state, weight_matrix, letter=letter_vector_g):
    states = []
    for i in range(state.size):
        net_input = np.dot(weight_matrix[i], state)
        state[i] = 1 if net_input >= 0 else -1
        states.append(state.copy())
    # Check if the memory has been stored
    is_stored = (np.array_equal(states[-1], letter) or np.array_equal(states[-1], -letter))
    return states, is_stored


# a. Initial state is the letter vector
initial_state = letter_vector_g.copy()

# Run the simulation
states, _ = hopfield_update(initial_state, weight_matrix)
# Check if the state is stable
is_stable = np.array_equal(states[-1], states[-2])
print("Is the state stable?", is_stable)


# b. initial state is the flipped letter vector
# Visualization function
def plot_letter(vector, title):
    vector = vector * -1
    matrix = vector.reshape((10, 9))  # Assuming the vectors are 9x10
    plt.imshow(matrix, cmap='gray')
    plt.title(title, fontsize=10)  # Increased font size for better readability
    plt.axis('off')  # Turn off axis labels and numbers


# Create a noisy version of the initial state
initial_state_noisy = initial_state.copy()
# Flip 20% of the bits
n_flips = int(0.2 * initial_state_noisy.size)
flip_indices = np.random.choice(range(initial_state_noisy.size), n_flips, replace=False)
initial_state_noisy[flip_indices] *= -1

# Run the simulation with the noisy state
states_noisy, is_stored_noisy = hopfield_update(initial_state_noisy, weight_matrix)

# Plot every fifth step and the last step
plt.figure(figsize=(12, 8))  # Increased figure size for better quality
n_updates = list(range(0, len(states_noisy), 5)) + [len(states_noisy) - 1]
for i, update in enumerate(n_updates):
    plt.subplot(4, 6, i + 1)
    plot_letter(states_noisy[update], f'Update {update}')

plt.suptitle('Noisy State Convergence', fontsize=16)
plt.tight_layout()
plt.show()

# Check if the noisy state has converged to the letter G
print("Has the noisy state converged to G?", is_stored_noisy)


# c. Set the network to remember letters G and H
# Extract the vector for the letter 'H'
letter_index = 7
letter_vector_h = alphabet[:, letter_index]
# Train the network with both letters
weight_matrix2 = np.outer(letter_vector_h, letter_vector_h) / n_neurons + weight_matrix
np.fill_diagonal(weight_matrix2, 0)

# Run the simulation with from several noisy states
noise_levels = range(5, 55, 5)
flips = [math.ceil(int(level / 100 * n_neurons)) for level in noise_levels]
n_simulations = 100
# Run the simulations
success_rates = []
for flip in flips:
    success_count = 0
    for _ in range(n_simulations):
        initial_state_noisy = letter_vector_g.copy()
        flip_indices = np.random.choice(range(n_neurons), flip, replace=False)
        initial_state_noisy[flip_indices] *= -1
        states_noisy, is_stored = hopfield_update(initial_state_noisy, weight_matrix2)
        success_count += is_stored
    success_rate = success_count / n_simulations
    success_rates.append(success_rate)

# Plot the success rates
plt.plot(noise_levels, success_rates, marker='o')
plt.title('Success rate vs. noise level')
plt.xlabel('Noise level (%)')
plt.ylabel('Success rate')
plt.grid(True)
plt.show()

# d.
success_rate_all = np.zeros(alphabet.shape[1])
weight_matrix_all = np.zeros((n_neurons, n_neurons))
for index in range(alphabet.shape[1]):
    # Update the weight matrix with the outer product of the current pattern
    added_letter = alphabet[:, index]
    weight_matrix_all += np.outer(added_letter, added_letter) / n_neurons
    np.fill_diagonal(weight_matrix_all, 0)

    success_count = 0
    for letter in alphabet[:, : index + 1].T:
        states, is_stored = hopfield_update(letter.copy(), weight_matrix_all, letter=letter)
        success_count += is_stored
    success_rate_all[index] = success_count / (index + 1)

# Plot the success rates as a function of the number of letters stored
plt.plot(range(1, alphabet.shape[1] + 1), success_rate_all * 100, marker='o')
plt.title('Success rate vs. number of stored letters')
plt.xlabel('Number of stored letters')
plt.ylabel('Success rate (%)')
plt.grid(True)
plt.show()