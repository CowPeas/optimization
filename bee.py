import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the fitness function to optimize
def fitness_function(features):
    # Train a Decision Tree classifier on the selected features and calculate the accuracy on the test set
    clf = DecisionTreeClassifier()
    clf.fit(X_train[:, features], y_train)
    accuracy = clf.score(X_test[:, features], y_test)
    return accuracy

# Define the Artificial Bee Colony algorithm
def abc_algorithm(pop_size, num_iterations, limit):
    # Initialize the colony
    colony = np.zeros((pop_size, limit))
    colony_fitness = np.zeros((pop_size,))
    for i in range(pop_size):
        # Initialize the bee's solution
        bee_solution = np.zeros((limit,))
        bee_solution[np.random.choice(np.arange(limit))] = 1
        # Evaluate the bee's solution
        bee_fitness = fitness_function(np.where(bee_solution == 1)[0])
        colony[i] = bee_solution
        colony_fitness[i] = bee_fitness
    
    # Initialize the best solution
    best_solution = None
    best_fitness = 0.0
    
    # Run the algorithm for the specified number of iterations
    for t in range(num_iterations):
        # Employed bees phase
        for i in range(pop_size):
            # Choose a random feature to modify
            feature_to_modify = np.random.choice(np.arange(limit))
            # Choose a random bee different from the current bee
            other_bee_indices = np.arange(pop_size) != i
            other_bee_index = np.random.choice(np.arange(pop_size)[other_bee_indices])
            other_bee_solution = colony[other_bee_index]
            # Modify the feature based on the other bee's solution
            new_bee_solution = colony[i].copy()
            if other_bee_solution[feature_to_modify] == 1:
                new_bee_solution[feature_to_modify] = 1
            else:
                new_bee_solution[feature_to_modify] = 0
            # Evaluate the new solution and update the colony
            new_bee_fitness = fitness_function(np.where(new_bee_solution == 1)[0])
            if new_bee_fitness > colony_fitness[i]:
                colony[i] = new_bee_solution
                colony_fitness[i] = new_bee_fitness
        
        # Onlooker bees phase
        # Calculate the probabilities of each bee being chosen as an onlooker bee
        probabilities = colony_fitness / np.sum(colony_fitness)
        # Choose the onlooker bees
        onlooker_bee_indices = np.random.choice(np.arange(pop_size), size=pop_size, p=probabilities)
        for i in onlooker_bee_indices:
            # Choose a random feature to modify
            feature_to_modify = np.random.choice(np.arange(limit))
            # Choose a random bee different from the current bee
            other_bee_indices = np.arange(pop_size) != i
            other_bee_index = np.random.choice(np.arange(pop_size)[other_bee_indices])
            other_bee_solution = colony[other_bee_index]
            # Modify the feature based on the other bee's solution
            new_bee_solution = colony[i].copy()
            if other_bee_solution[feature_to_modify] == 1:
                new_bee_solution[feature_to_modify] = 1
            else:
                new_bee_solution[feature_to_modify] = 0
                # Evaluate the new solution and update the colony
                new_bee_fitness = fitness_function(np.where(new_bee_solution == 1)[0])
                if new_bee_fitness > colony_fitness[i]:
                    colony[i] = new_bee_solution
                    colony_fitness[i] = new_bee_fitness
            # Onlooker bees phase
    # Calculate the probabilities of each bee being chosen as an onlooker bee
    probabilities = colony_fitness / np.sum(colony_fitness)
    # Choose the onlooker bees
    onlooker_bee_indices = np.random.choice(np.arange(pop_size), size=pop_size, p=probabilities)
    for i in onlooker_bee_indices:
        # Choose a random feature to modify
        feature_to_modify = np.random.choice(np.arange(limit))
        # Choose a random bee different from the current bee
        other_bee_indices = np.arange(pop_size) != i
        other_bee_index = np.random.choice(np.arange(pop_size)[other_bee_indices])
        other_bee_solution = colony[other_bee_index]
        # Modify the feature based on the other bee's solution
        new_bee_solution = colony[i].copy()
        if other_bee_solution[feature_to_modify] == 1:
            new_bee_solution[feature_to_modify] = 1
        else:
            new_bee_solution[feature_to_modify] = 0
        # Evaluate the new solution and update the colony
        new_bee_fitness = fitness_function(np.where(new_bee_solution == 1)[0])
        if new_bee_fitness > colony_fitness[i]:
            colony[i] = new_bee_solution
            colony_fitness[i] = new_bee_fitness
    
    # Scout bees phase
    for i in range(pop_size):
        # If a solution has not improved in a certain number of iterations, replace it with a random solution
        if np.random.random() < 0.1 and colony_fitness[i] < best_fitness:
            bee_solution = np.zeros((limit,))
            bee_solution[np.random.choice(np.arange(limit))] = 1
            bee_fitness = fitness_function(np.where(bee_solution == 1)[0])
            colony[i] = bee_solution
            colony_fitness[i] = bee_fitness
    
    # Update the best solution
    for i in range(pop_size):
        if colony_fitness[i] > best_fitness:
            best_solution = colony[i]
            best_fitness = colony_fitness[i]

    return best_solution, best_fitness

# Print the best solution and its accuracy
print("Best solution:", best_solution)
print("Accuracy:", best_fitness)


