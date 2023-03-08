import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the fitness function to optimize
def fitness_func(features):
    # Extract selected features from the training and testing sets
    X_train_selected = X_train[:, features]
    X_test_selected = X_test[:, features]
    # Train a Decision Tree classifier on the selected features and calculate the accuracy on the test set
    clf = DecisionTreeClassifier()
    clf.fit(X_train_selected, y_train)
    accuracy = clf.score(X_test_selected, y_test)
    # Calculate the number of selected features
    num_selected_features = len(features)
    # Return the fitness value as a tuple
    return (-accuracy, num_selected_features)

# Define the Gravity Search Algorithm
def gravity_search_algorithm(num_particles, max_iterations, G0, alpha, beta):
    # Initialize the particles
    particles = np.random.rand(num_particles, X.shape[1])
    velocities = np.zeros((num_particles, X.shape[1]))
    fitness = np.zeros(num_particles)
    best_fitness = np.inf
    best_features = None
    # Initialize the gravitational constant
    G = G0
    # Run the algorithm for the specified number of iterations
    for t in range(max_iterations):
        # Calculate the fitness of each particle and update the best solution
        for i in range(num_particles):
            fitness[i] = fitness_func(np.where(particles[i] > 0.5)[0])[0]
            if fitness[i] < best_fitness:
                best_fitness = fitness[i]
                best_features = np.where(particles[i] > 0.5)[0]
        # Calculate the gravitational forces on each particle
        forces = np.zeros((num_particles, X.shape[1]))
        for i in range(num_particles):
            for j in range(num_particles):
                if i != j:
                    r = np.linalg.norm(particles[i] - particles[j])
                    forces[i] += G * (particles[j] - particles[i]) * fitness[j] / (r ** beta + 1e-10)
        # Update the velocities and positions of the particles
        velocities += forces * alpha
        particles += velocities
        # Clip the positions to the bounds [0, 1]
        particles = np.clip(particles, 0, 1)
        # Update the gravitational constant
        G = G0 / (1 + t)
    # Return the best solution found
    return best_features



###
best_features = gravity_search_algorithm(num_particles=50, max_iterations=100, G0=100, alpha=0.1, beta=2)

##
X_train_selected = X_train[:, best_features]
X_test_selected = X_test[:, best_features]

##
clf = DecisionTreeClassifier()
clf.fit(X_train_selected, y_train)

##
accuracy = clf.score(X_test_selected, y_test)

##
print("Selected features: {}".format(best_features))
print("Accuracy: {:.2f}%".format(accuracy*100))
