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

# Define the Bat Algorithm
def bat_algorithm(num_bats, max_iterations, A0, r0, alpha, gamma, fmin, fmax):
    # Initialize the bats
    bats = np.zeros((num_bats, X.shape[1]))
    frequencies = np.zeros(num_bats)
    velocities = np.zeros((num_bats, X.shape[1]))
    fitness = np.zeros(num_bats)
    best_fitness = np.inf
    best_features = None
    # Initialize the frequencies and velocities
    for i in range(num_bats):
        bats[i] = np.random.rand(X.shape[1])
        frequencies[i] = fmin + (fmax - fmin) * np.random.rand()
        velocities[i] = np.zeros(X.shape[1])
        fitness[i] = fitness_func(np.where(bats[i] > 0.5)[0])
        if fitness[i][0] < best_fitness:
            best_fitness = fitness[i][0]
            best_features = np.where(bats[i] > 0.5)[0]
    # Run the Bat Algorithm for the specified number of iterations
    for t in range(max_iterations):
        # Update the frequency and velocity of each bat
        for i in range(num_bats):
            # Update the frequency
            frequencies[i] = fmin + (fmax - fmin) * np.random.rand()
            # Update the velocity
            velocities[i] += (bats[i] - best_features) * frequencies[i]
            # Update the position
            bats[i] += velocities[i]
            # Apply the loudness and pulse rate
            if np.random.rand() > r0:
                bats[i] = best_features + alpha * np.random.normal(0, 1, X.shape[1])
            if np.random.rand() < gamma:
                bats[i] = np.where(np.random.rand(X.shape[1]) < bats[i], 1, 0)
            # Clip the position to the bounds [0, 1]
            bats[i] = np.clip(bats[i], 0, 1)
            # Evaluate the fitness of the new position
            new_fitness = fitness_func(np.where(bats[i] > 0.5)[0])
            # Update the best fitness and features if a better solution is found
            if new_fitness[0] < best_fitness:
                best_fitness = new_fitness[0]
                best_features = np.where(bats[i] > 0.5)[0]
        # Update the fitness of the current bat
                fitness[i] = new_fitness
    # Return the best fitness and features
    return best_fitness, best_features


#Run the Bat Algorithm with the specified parameters

num_bats = 10
max_iterations = 100
A0 = 1.0
r0 = 0.5
alpha = 0.5
gamma = 0.5
fmin = 0.0
fmax = 2.0
best_fitness, best_features = bat_algorithm(num_bats, max_iterations, A0, r0, alpha, gamma, fmin, fmax)


#Print the results

print('Best fitness:', best_fitness)
print('Selected features:', best_features)
