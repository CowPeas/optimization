import random
from deap import creator, base, tools
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define the fitness function
def evalFitness(individual):
    # Create a list of selected features
    selected_features = [i for i in range(len(individual)) if individual[i]]
    
    # If no features are selected, return a fitness of 0
    if len(selected_features) == 0:
        return 0,
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X[:, selected_features], y, test_size=0.3, random_state=0)
    
    # Train a decision tree classifier on the selected features
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    
    # Calculate the accuracy on the test set
    accuracy = clf.score(X_test, y_test)
    
    return accuracy,

# Set up the GA parameters
pop_size = 50
num_generations = 10
cx_prob = 0.5
mut_prob = 0.2

# Set up the GA toolbox
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(X[0]))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalFitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1/len(X[0]))
toolbox.register("select", tools.selTournament, tournsize=3)

# Run the GA
pop = toolbox.population(n=pop_size)
for gen in range(num_generations):
    offspring = tools.selBest(pop, k=10) # Elitism
    offspring += toolbox.select(pop, len(pop) - 10)
    random.shuffle(offspring)
    for i in range(0, len(offspring) - 1, 2):
        if random.random() < cx_prob:
            toolbox.mate(offspring[i], offspring[i+1])
        if random.random() < mut_prob:
            toolbox.mutate(offspring[i])
            toolbox.mutate(offspring[i+1])
        del offspring[i].fitness.values, offspring[i+1].fitness.values
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    pop[:] = offspring
    
# Get the best individual
best_ind = tools.selBest(pop, k=1)[0]
selected_features = [i for i in range(len(best_ind)) if best_ind[i]]
print("Selected features:", selected_features)
