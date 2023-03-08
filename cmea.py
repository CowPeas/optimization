import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.model.problem import Problem
from pymoo.optimize import minimize

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the fitness function to optimize
class FeatureSelection(Problem):
    def __init__(self):
        super().__init__(n_var=X.shape[1],
                         n_obj=2,
                         n_constr=1,
                         xl=0,
                         xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        # Apply feature selection
        selected_features = np.where(x == 1)[0]
        if len(selected_features) == 0:
            out["F"] = [np.inf, np.inf]
            out["G"] = [1]
            return
        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]
        # Train a Decision Tree classifier on the selected features and calculate the accuracy on the test set
        clf = DecisionTreeClassifier()
        clf.fit(X_train_selected, y_train)
        accuracy = clf.score(X_test_selected, y_test)
        # Calculate the number of selected features
        num_selected_features = np.sum(x)
        # Set the objectives and constraints
        out["F"] = [-accuracy, num_selected_features]
        out["G"] = [len(selected_features) - 1]

# Define the Constrained Multi-objective Evolutionary Algorithm
def cmea_algorithm(pop_size, num_iterations):
    problem = FeatureSelection()
    algorithm = GA(pop_size=pop_size,
                   sampling=get_sampling("bin_random"),
                   crossover=get_crossover("bin_hux"),
                   mutation=get_mutation("bin_bitflip"),
                   eliminate_duplicates=True)

    # Run the algorithm for the specified number of iterations
    res = minimize(problem,
                   algorithm,
                   ('n_gen', num_iterations),
                   seed=42)

    # Get the binary representation of the best solution
    x_best = res.X[0]
    selected_features = np.where(x_best == 1)[0]
    print("Selected features:", selected_features)

    return res

# Run the algorithm and print the selected features
cmea_algorithm(100, 50)
