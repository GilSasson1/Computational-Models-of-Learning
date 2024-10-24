### Gil Sasson - 208689497 ###
## Final Project: Reinforcement Learning with Punishment for Mice Ethanol Consumption Data
# In this project I implemented a reinforcement learning algorithm with punishment for mice ethanol consumption data
# from my seminar project. The data consists of 8 mice consuming 10% ethanol and 8 mice consuming 20% ethanol over
# 12 weeks. The reinforcement learning algorithm updates predictions based on the reward/punishment signal for each
# mouse and ethanol group. The reward function is based on the week, consumption, and ethanol group, with a punishment
# for quinine concentration and a reward for reducing ethanol consumption. The algorithm is run for 1000 iterations
# per session and the predictions are plotted against the actual data. The cumulative reward over time is also plotted
# for each ethanol group to visualize the learning process. Finally, the algorithm is run with different learning rates
# to compare the convergence of predictions for each group.

# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Import the data
data = pd.read_excel('/Users/gilsasson/Library/CloudStorage/OneDrive-mail.tau.ac.il/פסיכולוגיה/שנה ג/Seminar data.xlsx', header=None)
data = data.iloc[2:, :]

# Remove rows with means
for i in range(data.shape[0]):
    if i % 4 == 0 and i != 0:
        data = data.drop(i)
# Remove NaN values
data = data.dropna()
# Reset the index
data.reset_index(drop=True, inplace=True)
# Convert the data to a numpy array
data = data.to_numpy()

# Every week has 3 sessions
sessions_per_week = 3
# Define the reward function
def reward_function(week, consumption, ethanol_group):
    """
    Function to compute the reward based on the week, consumption, and ethanol group. The function applies a punishment
    based on the quinine concentration and rewards the reduction in ethanol consumption.
    for the 20 % ethanol group, the reward is multiplied by -0.2 to account for compulsiveness.
    :param week:
    :param consumption:
    :param ethanol_group:
    :return: reward
    """
    quinine_multiplier = 0
    if 4 <= week < 6:
        quinine_multiplier = -0.03  # Punishment for weeks 4-6
    elif 6 <= week < 8:
        quinine_multiplier = -0.075  # Punishment for weeks 6-8
    elif week >= 8:
        quinine_multiplier = -0.125  # Higher punishment after week 8
    elif week >= 10:
        quinine_multiplier = -0.2    # Maximum punishment after week 10

    # Positive reward for reducing ethanol consumption (inverse relationship)
    if consumption < 7.5 and week >= 4:
        reward = 0.5 + quinine_multiplier  # Positive reward for reducing consumption
    else:
        reward = quinine_multiplier # Apply punishment based on quinine concentration

    if ethanol_group == 20:
        reward *= -0.2 # Account for compulsiveness in the 20% ethanol group

    return reward

# Define the reinforcement learning algorithm with punishment
def reinforcement_learning(data, learning_rate=0.0001, iterations=1000 * data.shape[0]):
    """
    Function to implement the reinforcement learning algorithm with punishment for the ethanol consumption data.
    :param data:
    :param learning_rate:
    :param iterations:
    :return: prediction
    """
    prediction = np.ones([iterations, data.shape[1]])  # Initialize predictions
    prediction *= 7.5  # Initialize the prediction with the average consumption
    prediction[:sessions_per_week * 4] += np.random.normal(0, 0.5, (
    sessions_per_week * 4, data.shape[1]))  # Add noise for first 4 weeks
    num_mice = data.shape[1]

    # Iterating over time and updating predictions
    for i in range(1, iterations - 1):
        session = i % data.shape[0]  # Current week index
        delta = data[session] - prediction[i, :]  # Compute prediction error

        for mouse in range(num_mice):
            ethanol_group = 10 if mouse < 8 else 20  # Determine ethanol group (10% or 20%)
            consumption = data[session, mouse]  # Current consumption for this mouse

            # Compute reward (punishment) based on consumption and quinine
            reward = reward_function(session // sessions_per_week, consumption, ethanol_group)

            # Update prediction with learning rate and reward/punishment
            prediction[i + 1, mouse] = prediction[i, mouse] + learning_rate * reward * delta[mouse]

    return prediction


def plot_cumulative_rewards(data):
    cumulative_rewards_10 = []
    cumulative_rewards_20 = []
    cum_10 = 0
    cum_20 = 0

    for session in range(data.shape[0]):
        total_reward_10 = 0
        total_reward_20 = 0
        for mouse in range(data.shape[1]):
            reward = reward_function(session // sessions_per_week, data[session, mouse], mouse)
            if mouse < 8:
                total_reward_10 += reward
            else:
                total_reward_20 += reward

        cum_10 += total_reward_10 / data[:, :8].shape[1]
        cum_20 += total_reward_20 / data[:, 8:].shape[1]

        cumulative_rewards_10.append(cum_10)
        cumulative_rewards_20.append(cum_20)

    time_periods = np.arange(data.shape[0])
    plt.figure(figsize=(10, 6))
    plt.plot(time_periods, cumulative_rewards_10, label='Ethanol 10%')
    plt.plot(time_periods, cumulative_rewards_20, label='Ethanol 20%')

    plt.title('Cumulative Reward Over Time')
    plt.xlabel('Time Periods (Sessions)')
    plt.ylabel('Cumulative Reward')
    plt.axvline(x=4 * sessions_per_week, color='black', linestyle='--', label='Quinine Starts')
    plt.axvline(x=6 * sessions_per_week, color='red', linestyle='--')
    plt.axvline(x=8 * sessions_per_week, color='red', linestyle='--')
    plt.axvline(x=10 * sessions_per_week, color='red', linestyle='--')
    plt.axvspan(4 * sessions_per_week, 6 * sessions_per_week, color='gray', alpha=0.3, label='Low Quinine')
    plt.axvspan(6 * sessions_per_week, 8 * sessions_per_week, color='lightgray', alpha=0.3, label='Moderate Quinine')
    plt.axvspan(8 * sessions_per_week, 10 * sessions_per_week, color='darkgray', alpha=0.3, label='High Quinine')
    plt.axvspan(10 * sessions_per_week, 14 * sessions_per_week, color='black', alpha=0.3, label='Maximum Quinine')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot predictions for different learning rates
def plot_learning_rate_comparison(data, learning_rates, iterations=1000):
    plt.figure(figsize=(12, 8))

    for idx, lr in enumerate(learning_rates):
        predictions = reinforcement_learning(data, learning_rate=lr, iterations=iterations)
        avg_predictions_10 = np.mean(predictions[:, :8], axis=1)  # Averaging predictions across mice in 10% group
        avg_predictions_20 = np.mean(predictions[:, 8:], axis=1) # Averaging predictions across mice in 20% group

        plt.subplot(2, 3, idx + 1)
        plt.plot(avg_predictions_10, label='10% ethanol', color='blue')
        plt.plot(avg_predictions_20, label='20% ethanol', color='red')
        plt.title(f'Learning Rate = {lr}')
        plt.xlabel('Iterations')
        plt.ylabel('Average Prediction Consumption')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

# Run the reinforcement learning algorithm
predictions = reinforcement_learning(data, learning_rate=0.0005)

# Plot the Data and Predictions
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(data[:, :8].mean(axis=1), label='Ethanol 10%')
plt.plot(data[:, 8:].mean(axis=1), label='Ethanol 20%')
plt.title('Data')
plt.xlabel('Time Periods (Sessions)')
plt.ylabel('Ethanol Consumption (g/kg)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(predictions[:, :8].mean(axis=1), label='Ethanol 10%')
plt.plot(predictions[:, 8:].mean(axis=1), label='Ethanol 20%')
plt.title('Predictions')
plt.xlabel('Iterations')
plt.ylabel('Ethanol Consumption (g/kg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot the reward over time for each ethanol group
plot_cumulative_rewards(data)

# Define learning rates to compare
learning_rates = [0.1, 0.001, 0.0005, 0.0001, 0.00001, 0.000001]

# Run the function with your data and defined learning rates
plot_learning_rate_comparison(data, learning_rates, iterations=data.shape[0] * 1000)

# Observations:
# The reinforcement learning algorithm with punishment effectively reduces ethanol consumption over time for the 10%
# ethanol group but not for the 20% group due to the compulsiveness factor in the reward function, similar to the data.
# The cumulative reward plot shows the learning process with increasing punishment for quinine concentration over time.
# The learning rate comparison shows that lower learning rates converge slower but more accurately to the data.
# Higher learning rates converge faster but may overshoot the optimal predictions. The optimal learning rate depends on
# the trade-off between convergence speed and prediction accuracy. The algorithm can be further optimized with
# hyperparameter tuning and additional reward functions to improve predictions for both ethanol groups.
