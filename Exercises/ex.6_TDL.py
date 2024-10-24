import numpy as np
import matplotlib.pyplot as plt


def td_learning(p, steps, learning_rate=0.1):
    # Create the world
    world_state = np.zeros((3, 3))

    # Set reward matrix
    reward_matrix = np.ones_like(world_state) * -1
    reward_matrix[2, 2] = 10

    # Set the movement directions
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    start = (1, 1)
    position = start

    for _ in range(steps):
        if np.random.rand() <= p:
            best_value = -np.inf
            for direction in directions:
                future_position = (position[0] + direction[0], position[1] + direction[1])
                if 0 <= future_position[0] < 3 and 0 <= future_position[1] < 3:  # check bounds
                    value = reward_matrix[future_position] + world_state[future_position]
                    if value > best_value:
                        best_value = value
                        best_position = future_position
            future_position = best_position
        else:
            direction = directions[np.random.choice(range(4))]
            future_position = (position[0] + direction[0], position[1] + direction[1])
            if not (0 <= future_position[0] < 3 and 0 <= future_position[1] < 3):  # check bounds
                # Out of bounds, apply penalty and reset position
                world_state[position] += learning_rate * (-10 - world_state[position])
                position = start
                continue

        # Update the world state using TD learning rule
        gamma = reward_matrix[future_position] + world_state[future_position] - world_state[position]
        world_state[position] += learning_rate * gamma

        position = future_position

        # If the agent reaches the goal, reset to start
        if position == (2, 2):
            position = start

    return world_state + reward_matrix


# Example usage
world_state = td_learning(p=0.8, steps=30000, learning_rate=0.01)
plt.imshow(world_state, cmap='viridis', interpolation='nearest', origin='upper')
plt.colorbar()
plt.title('World State after TD Learning')
# Add the values to the plot
for i in range(3):
    for j in range(3):
        plt.text(j, i, f'{world_state[i, j]:.2f}', ha='center', va='center', color='black')
plt.show()


def calculate_expected_value(world_state, position):
    # Define possible movements
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    expected_value = 0
    num_valid_moves = 0

    for direction in directions:
        future_position = (position[0] + direction[0], position[1] + direction[1])

        # Check if the future position is within bounds
        if 0 <= future_position[0] < 3 and 0 <= future_position[1] < 3:
            future_value = world_state[future_position]
            expected_value += future_value
            num_valid_moves += 1

    if num_valid_moves > 0:
        expected_value /= num_valid_moves  # Average over possible moves

    return expected_value


reward_matrix = np.ones((3, 3)) * -1
reward_matrix[2, 2] = 10

# Calculate the expected value at of X
x_position = (1, 2)
expected_value_x = calculate_expected_value(world_state, x_position)
print(f"Expected value at position {(2, 3)}: {expected_value_x}")
