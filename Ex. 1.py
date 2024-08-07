import numpy as np
import matplotlib.pyplot as plt

# Define some helping functions
def roll_samples(mean_1, mean_2, std, n):
    """
    Rolls n samples from two distributions with the same standard deviation.
    The number of samples generated from each distribution is determined randomly.
    ----------
    :param mean_1: mean of first distribution.
    :param mean_2: mean of second distribution.
    :param std: standard deviation.
    :param n: number of samples.
    :return: random samples from two distributions.
    """
    data = np.zeros(shape=(n, 2))
    for j in range(n):
        dist = np.random.randint(0, 2)
        if dist == 0:
            num = np.random.multivariate_normal(mean_1, [[std**2, 0], [0, std**2]])
        else:
            num = np.random.multivariate_normal(mean_2, [[std**2, 0], [0, std**2]])
        data[j] = num
    return data


def binary_perceptron(data, max_steps=100):
    """
    Binary Perceptron algorithm with stopping criteria.
    --------
    :param data: A data structure.
    :param max_steps: stopping criteria.
    :return: the final weights array, number of updating iterations.
    """
    n_samples = data.shape[0]
    y = np.sign(data[:, 1]).flatten()
    y_pred = np.zeros(n_samples)
    weights = np.random.normal(0, 1)
    counter = 0
    while not np.array_equal(y, y_pred):
        for i in range(n_samples):
            y_pred[i] = np.sign(weights * data[i, 0])
            if y[i] != y_pred[i]:
                weights += y[i] * data[i, 0]
                counter += 1
        if counter >= max_steps:
            return False, weights, counter  # Return False indicating failure to converge
    return True, weights, counter  # Return True indicating successful convergence


def run_experiments(mean_1, mean_2, std_list, n, num_runs=100):
    """""
    Runs the experiments several times with different standard deviations.
    -----------------
    :param mean_1: mean of first distribution.
    :param mean_2: mean of second distribution.
    :param std_list: list of standard deviations.
    :param n: number of samples.
    :param num_runs: number of experiments. defaults to 100.
    :return: a nested dictionary with the success rate per standard deviation, mean number of steps per standard deviation.
    """
    results = {}
    for std in std_list:
        successes = 0
        total_steps = 0
        for _ in range(num_runs):
            data = roll_samples(mean_1, mean_2, std, n)
            converged, _, steps = binary_perceptron(data)
            if converged:
                successes += 1
                total_steps += steps
        success_rate = 100 * (successes / num_runs)
        mean_steps = total_steps / successes if successes > 0 else float('inf')
        results[std] = {'success_rate': success_rate, 'mean_steps': mean_steps}
        print(f"Std: {std}, Success Rate: {success_rate:.2f}, Mean Steps: {mean_steps:.2f}")
    return results


# Example usage:
mean_1 = [3, -3]
mean_2 = [-3, 3]
std_list = [0.5, 1.0, 2.0, 3.0]
n = 10
# run
results = run_experiments(mean_1, mean_2, std_list, n)

# plot the graphs
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
success_rates = [results[std]['success_rate'] for std in std_list]
plt.plot(success_rates)
plt.title("Successful Classification vs. Std")
plt.xlabel("Standard Deviation")
plt.ylabel("Success Rate")

plt.subplot(1, 2, 2)
mean_steps = [results[std]['mean_steps'] for std in std_list]
plt.plot(mean_steps)
plt.title("Mean Number of Steps vs. Std")
plt.xlabel("Standard Deviation")
plt.ylabel("Mean Number of Steps")
plt.show()
