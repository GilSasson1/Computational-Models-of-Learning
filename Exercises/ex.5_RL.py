import numpy as np
import matplotlib.pyplot as plt


# Define the reinforcement learning algorithm
def target_shooting(initial_aim, learning_rate, reward_function=lambda y: -(2-y)**2, noise_std=0.1, iteration=10000):
    aim = np.zeros(iteration)
    hit = np.zeros(iteration)
    aim[0] = initial_aim
    noise = np.random.normal(scale=noise_std, size=iteration)
    for i in range(iteration - 1):  # Adjust loop to avoid out-of-bounds error
        hit[i] = aim[i] + noise[i]
        eligibility = hit[i] - aim[i]  # According to the task analytical part the eligibility parameter is equal
        # to the difference between the hit and the aim divided by the noise variance.
        delta = learning_rate * reward_function(hit[i]) * 1/(noise_std**2) * eligibility
        aim[i + 1] = aim[i] + delta  # Update the next aim value
    hit[iteration - 1] = aim[iteration - 1] + noise[iteration - 1]  # Compute the last hit value
    return aim, hit


fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax = ax.flatten()

# Learning rates
etas = np.logspace(-5, -2, 6)
aim_all = np.zeros((200, 10000))
aims_on_target = np.zeros(6)
# Run the simulation for each learning rate 200 times
for idx, eta in enumerate(etas):
    for i in range(200):
        aim, hit = target_shooting(0, eta)
        aim_all[i, :] = aim
        median_aim = np.median(aim_all, axis=0)
        # Check if the aim is on target
        is_aim_on_target = np.abs(aim_all[i, -1] - 2) < 0.1
        # Count the number of times the aim is on target
        aims_on_target[idx] += is_aim_on_target
    # Plot the last 5 aim values and the median aim value
    for j in range(5):
        ax[idx].plot(aim_all[-j, :])
    ax[idx].plot(median_aim, color='black', linewidth=2, label='Median')
    ax[idx].set_title(f'Learning rate: {round(eta, 5)}')
    ax[idx].set_xlabel('Iteration')
    ax[idx].set_ylabel('Aim Location')
    ax[idx].legend()

plt.suptitle("Player's Aim Location Over Time with Different Learning Rates", fontsize=16)
plt.tight_layout()
plt.show()

# Plot the percentage of times the aim is on target for each learning rate
plt.figure(figsize=(8, 6))
plt.plot(etas, aims_on_target / 200 * 100, marker='o')
plt.xscale('log')
plt.title('Percentage of Times Aim is on Target vs. Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Percentage of Times Aim is on Target (%)')
plt.grid(True)
plt.show()

# We can see that for very small learning rates, the aim does not converge to the target. For very large learning rates,
# the aim converges to the target most of the time, but for medium to large learning rates, the aim always converges to
# the target. This is because the learning rate is proportional to the step size of the update, and a very small
# learning rate results in very small updates that do not allow the aim to converge to the target. On the other hand,a
# very large learning rate results in very large updates that overshoot the target and oscillate around it.
# A medium learning to large rate results in updates that are large enough to converge to the target but not too large
# to overshoot it.


# Redefine the reward function
@np.vectorize
def reward_function(y):
    if y <= 2:
        return -2*(y-2)**2
    else:
        return -(y-2)**2


# Std deviation of the noise
noise_std = [0.1, 0.5]
# Run the algorithm with the new reward function and noise levels
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax = ax.flatten()
for idx, std in enumerate(noise_std):
    aim_final_all = np.zeros(200)
    y = np.linspace(1, 5, 10000)
    for i in range(200):
        aim, hit = target_shooting(0, 0.0001, reward_function=reward_function, noise_std=std)
        aim_final_all[i] = aim[-1]
    median_aim = np.median(aim_final_all)
    ax[idx].plot(y, reward_function(y), alpha=0.5, linestyle='-')
    ax[idx].axvline(x=median_aim, color='black', linestyle='--', label='Median Aim Location')
    ax[idx].set_title(f'Noise Standard Deviation: {std}')
    ax[idx].set_xlabel('Hit Location')
    ax[idx].set_ylabel('Reward')
    ax[idx].grid(True)
    ax[idx].legend()
plt.suptitle("Player's Reward vs. Hit Location with Different Noise Levels", fontsize=16)
plt.show()

# We can see that for the lower noise level the median aim location is slightly below 2, while for the higher noise
# level, the median aim location is slightly above 2. This is happening because the reward function is penalizing less
# for aim locations above 2 compared to aim locations below 2. The higher noise level results in a larger spread of aim
# location, and larger update size which can cause more overshooting of the target. This overshooting is penalized less
# by the reward function so the median aim location is slightly above 2. For the lower noise level, the spread of aim
# locations is smaller and updates are smaller, which results in less overshooting the target so the aim locations are
# less likely to reach values above 2.

