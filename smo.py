import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


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
    # Train a Support Vector Machine on the selected features and calculate the accuracy on the test set
    clf = SVC(kernel='linear')
    clf.fit(X_train_selected, y_train)
    accuracy = clf.score(X_test_selected, y_test)
    # Calculate the number of selected features
    num_selected_features = len(features)
    # Return the fitness value as a tuple
    return (accuracy, num_selected_features)


# Define the Spider Monkey Optimization algorithm
def smo(num_spiders, max_iterations, q, alpha, beta, gamma):
    # Initialize the spiders
    spiders = np.random.rand(num_spiders, X.shape[1])
    fitness = np.zeros(num_spiders)
    best_fitness = -np.inf
    best_features = None
    # Evaluate the fitness of each spider
    for i in range(num_spiders):
        fitness[i], _ = fitness_func(np.where(spiders[i] > 0.5)[0])
        if fitness[i] > best_fitness:
            best_fitness = fitness[i]
            best_features = np.where(spiders[i] > 0.5)[0]
    # Run the Spider Monkey Optimization algorithm for the specified number of iterations
    for t in range(max_iterations):
        # Move each spider towards a better position
        for i in range(num_spiders):
            # Compute the distance between the current spider and all other spiders
            distances = np.linalg.norm(spiders - spiders[i], axis=1)
            # Compute the attraction and repulsion forces on the spider
            attraction = np.zeros(X.shape[1])
            repulsion = np.zeros(X.shape[1])
            for j in range(num_spiders):
                if i != j:
                    if fitness[j] > fitness[i]:
                        attraction += beta * (spiders[j] - spiders[i]) / distances[j]**q
                    else:
                        repulsion += gamma * (spiders[i] - spiders[j]) / distances[j]**q
            # Compute the new velocity of the spider
            velocity = alpha * (attraction + repulsion)
            # Update the position of the spider
            spiders[i] += velocity
            # Clip the position to the bounds [0, 1]
            spiders[i] = np.clip(spiders[i], 0, 1)
            # Evaluate the fitness of the new position
            new_fitness, _ = fitness_func(np.where(spiders[i] > 0.5)[0])
            # Update the best fitness and features if a better solution is found
            if new_fitness > fitness[i]:
                fitness[i] = new_fitness
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_features = np.where(spiders[i] > 0.5)[0]
    return best_features


best_features = smo(num_spiders=50, max_iterations=100, q=1, alpha=0.1, beta=1, gamma=2)

### 
accuracy, num_selected_features = fitness_func(best_features)

##
print("Selected features:", best_features)
print("Accuracy:", accuracy)
print("Number of selected features:", num_selected_features)
