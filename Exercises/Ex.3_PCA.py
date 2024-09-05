# Section 1: imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def roll_samples(n_samples, means):
    data = np.zeros((n_samples, 3))
    lucky_numbers = np.random.randint(0, 3, size=n_samples)  # Values will be 0, 1, or 2

    for i, lucky_number in enumerate(lucky_numbers):
        data[i, :] = np.random.multivariate_normal(means[lucky_number, :], cov=np.eye(3, 3), size=1)

    return data


def angle_between(v1, v2):
    nominator = np.dot(v1, v2)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(np.abs(nominator / denominator))


def pca_oja(X, learning_rate=0.001, max_iter=1000, tol=1e-6):
    print("Oja's Rule")
    n_samples, n_features = X.shape
    w = np.random.rand(n_features)
    errors = []
    angles = []
    cov_matrix = np.cov(X.T)
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigen_values)[::-1]
    first_eigen_vector = eigen_vectors[:, sorted_indices[0]]
    indices = np.random.permutation(n_samples)

    for epoch in range(max_iter // n_samples + 1):
        for i in indices:
            x_i = X[i]
            y = np.dot(w, x_i)
            w += learning_rate * y * (x_i - y * w)
            w /= np.linalg.norm(w)
            # Reconstruction error and angle tracking
            y_all = np.dot(X, w)
            X_reconstructed = np.outer(y_all, w)
            error = np.mean((X - X_reconstructed) ** 2)
            errors.append(error)
            angle = angle_between(first_eigen_vector, w)
            angles.append(angle)

        print(f"Epoch: {epoch}, Error: {error}, Angle: {angle}")

        if epoch > 0 and np.abs(errors[-2] - errors[-1]) < tol:
            print(f"Converged after {epoch + 1} epochs\n")
            break

    return w, errors, angles, first_eigen_vector


# Example usage
means = np.array([[-10, 10, -10], [-10, -10, -10], [20, 0, 20]])
np.random.seed(77)
X_example = roll_samples(100, means)

# Run
pca_component, errors, angles, first_eigen_vector = pca_oja(X_example, learning_rate=0.0001, max_iter=100)

# Plots
scale = 20
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_example[:, 0], X_example[:, 1], X_example[:, 2])
ax.quiver(0, 0, 0, pca_component[0] * scale,
          pca_component[1] * scale,
          pca_component[2] * scale,
          color='r')
ax.set_title('Data Points and PC1')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Plot errors vs iterations
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(errors)), errors)
plt.xlabel('Iterations')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error Change')
plt.yscale('log')  # Use logarithmic scale for y-axis

plt.subplot(1, 2, 2)
plt.plot(range(len(angles)), angles)
plt.xlabel('Iterations')
plt.ylabel('Angle Between EV1 and W')
plt.title('Angle Change')
plt.tight_layout()
plt.show()

# We can see in the graph that the angle between the first eigen vector and the weights vector is decreasing
# in absolute value. This means that the weights vector is converging to the first eigen vector,
# in other words - the weights vector is converging to the optimal solution for PC1.


def pca_sanger(X, n_components=2, learning_rate=0.001, max_iter=1000, tol=1e-6):
    print("Sanger's Rule")
    n_samples, n_features = X.shape
    W = np.random.rand(n_features, n_components)
    errors = []
    angles_1 = []
    angles_2 = []
    cov_matrix = np.cov(X.T)
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigen_values)[::-1]
    eigen_vectors_sorted = eigen_vectors[:, sorted_indices]
    indices = np.random.permutation(n_samples)

    for epoch in range(max_iter // n_samples + 1):
        for i in indices:
            x_i = X[i]
            y = np.dot(x_i, W)
            gradient = np.zeros_like(W)
            for j in range(n_components):
                projection = np.dot(W[:, :j], y[:j])
                gradient[:, j] = learning_rate * (x_i - projection) * y[j]
            W += gradient
            for j in range(n_components):
                W[:, j] /= np.linalg.norm(W[:, j])  # Normalize each component separately

            Y = np.dot(X, W)
            X_reconstructed = np.dot(Y, W.T)
            error = np.mean((X - X_reconstructed) ** 2)
            errors.append(error)
            angle_1 = angle_between(eigen_vectors_sorted[:, 0], W[:, 0])
            angle_2 = angle_between(eigen_vectors_sorted[:, 1], W[:, 1])
            angles_1.append(angle_1)
            angles_2.append(angle_2)

        print(f"Epoch: {epoch}, Error: {error}, Angle 1: {angle_1} Angle 2: {angle_2}")

        if epoch > 0 and np.abs(errors[-2] - errors[-1]) < tol:
            print(f"Converged after {epoch + 1} epochs\n")
            break

    return W, errors, angles_1, angles_2, eigen_vectors_sorted


# Run
pca_components, errors, angles_1, angles_2, eigen_vectors_sorted = pca_sanger(X_example, learning_rate=0.0005, max_iter=100)

# Plots
scale = 20
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
pc_colors = ['r', 'g', 'b']
ax.scatter(X_example[:, 0], X_example[:, 1], X_example[:, 2])
for i in range(pca_components.shape[1]):
    ax.quiver(0, 0, 0,
              pca_components[0, i] * scale,
              pca_components[1, i] * scale,
              pca_components[2, i] * scale,
              color=pc_colors[i], label=f'PC{i + 1}')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Data Points and PCA Components')
# Add a legend
plt.legend()
plt.show()

# Plot errors vs iterations
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(errors)), errors)
plt.xlabel('Iterations')
plt.ylabel('Reconstruction Error')
plt.title('PCA Error vs Iterations')
plt.yscale('log')  # Use logarithmic scale for y-axis
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.plot(range(len(angles_1)), angles_1, label='Angle 1')
plt.plot(range(len(angles_2)), angles_2, label='Angle 2')
plt.xlabel('Iterations')
plt.ylabel("Angles Between Weights and 2 Ev's")
plt.title('Angles Change')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# As I mentioned in my explanation for Oja's algorithm, the weights vector is converging to the eigen vector respectively,
# Here we can see this also for the second weight vector and EV2.
