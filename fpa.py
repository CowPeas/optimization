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

# Define the FPA algorithm
def fpa_algorithm(pop_size, num_iterations, alpha, beta, p, q, lower_bound, upper_bound, limit):
    # Initialize the population
    population = np.random.uniform(low=lower_bound, high=upper_bound, size=(pop_size, limit))
    
    # Evaluate the initial population
    fitness_values = np.array([fitness_function(np.where(p >= q)[0]) for p in population])
    
    # Find the best solution
    best_solution = population[np.argmax(fitness_values)]
    best_fitness = np.max(fitness_values)
    
    # Run the algorithm for the specified number of iterations
    for t in range(num_iterations):
        # Calculate the levy flights for each flower
        step_size = np.power(1.0 / np.sqrt(limit), beta)
        levy = np.random.normal(loc=0, scale=step_size, size=(pop_size, limit))
        step = alpha * levy * (population - best_solution)
        
        # Update the population
        new_population = population + step
        
        # Clip the population to the search space
        new_population = np.clip(new_population, lower_bound, upper_bound)
        
        # Evaluate the new population
        new_fitness_values = np.array([fitness_function(np.where(p >= q)[0]) for p in new_population])
        
        # Update the population based on the FPA rule
        for i in range(pop_size):
            j = np.random.randint(pop_size)
            if new_fitness_values[i] > fitness_values[j]:
                population[j] = new_population[i]
                fitness_values[j] = new_fitness_values[i]
        
        # Update the best solution
        if np.max(fitness_values) > best_fitness:
            best_solution = population[np.argmax(fitness_values)]
            best_fitness = np.max(fitness_values)
        
        # Print the best solution every 10 iterations
        if t % 10 == 0:
            print("Iteration {}: Best Fitness = {}".format(t, best_fitness))
    
    # Select the features of the best solution
    best_features = np.where(best_solution >= p)[0]
    
    return best_features, best_fitness

# Set the FPA algorithm parameters
pop_size = 50
num_iterations = 100
alpha = 0.01
beta = 1.5
p = 0.5
q = np.random.uniform(size=X.shape[1])
lower_bound = 0.0
upper_bound = 1.0
limit = X.shape[1]

# Run the FPA algorithm
best_features, best_fitness = fpa_algorithm(pop_size, num_iterations, alpha, beta, p, q, lower_bound, upper_bound, limit)

# Print the selected features and their fitness value
print("Selected Features:", best_features)
print("Fitness Value:", best_fitness)
