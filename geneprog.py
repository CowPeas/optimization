import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from deap import base, creator, tools

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the fitness function to optimize
def fitness_func(individual):
    # Extract selected features from the training and testing sets
    features = [i for i, f in enumerate(individual) if f]
    X_train_selected = X_train[:, features]
    X_test_selected = X_test[:, features]
    # Train a decision tree on the selected features and calculate the accuracy on the test set
    clf = DecisionTreeClassifier()
    clf.fit(X_train_selected, y_train)
    accuracy = clf.score(X_test_selected, y_test)
    # Calculate the number of selected features
    num_selected_features = sum(individual)
    # Return the fitness value as a tuple
    return (accuracy, num_selected_features)

# Create the toolbox for the genetic programming algorithm
toolbox = base.Toolbox()

# Define the fitness function and the individuals
creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define the attributes and the individual generator
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(iris.feature_names))

# Define the population generator
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Define the evaluation function
toolbox.register("evaluate", fitness_func)

# Define the main function to run the genetic programming algorithm
def main():
    # Initialize the population
    population = toolbox.population(n=100)

    # Evaluate the fitness of the initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Run the genetic programming algorithm for the specified number of generations
    for gen in range(10):
        # Select the next generation
        offspring = toolbox.select(population, len(population))

        # Apply the crossover operator
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply the mutation operator
        for mutant in offspring:
            if random.random() < 0.05:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the fitness of the offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the population with the offspring
        population[:] = offspring

    # Select the the best individual from the final population
    best_individual = tools.selBest(population, k=1)[0]

    # Extract selected features from the best individual
    features = [i for i, f in enumerate(best_individual) if f]

    # Print the selected features and their accuracy
    print("Selected features: {}".format([iris.feature_names[i] for i in features]))
    print("Accuracy: {}".format(fitness_func(best_individual)[0]))

    if name == "main":
        main()
