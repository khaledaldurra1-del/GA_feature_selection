# ga_demo.py
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load the dataset
# -------------------------------
data = pd.read_csv('homework.csv')
print("First five rows of the dataset:")
print(data.head())

# Display available columns
print("\nAvailable columns in the dataset:")
for i, col in enumerate(data.columns):
    print(f"{i}: {col}")

# Select target column
target_index = int(input("\nEnter the index of the target column: "))
target_column = data.columns[target_index]

# -------------------------------
# 2. Split features and target
# -------------------------------
X = data.drop(target_column, axis=1)
y = data[target_column]
feature_names = list(X.columns)

# -------------------------------
# 3. Define fitness function
# -------------------------------
def fitness_function(selected_features):
    if len(selected_features) == 0:
        return 0
    X_selected = X[selected_features]
    model = RandomForestClassifier()
    score = cross_val_score(model, X_selected, y, cv=2).mean()
    return score

# -------------------------------
# 4. Genetic Algorithm setup
# -------------------------------
population_size = 20
num_generations = 10
mutation_rate = 0.1

def create_individual():
    return [random.choice([0, 1]) for _ in range(len(feature_names))]

def individual_fitness(individual):
    selected = [f for i, f in enumerate(feature_names) if individual[i] == 1]
    return fitness_function(selected)

def select_parents(population):
    sorted_pop = sorted(population, key=lambda ind: individual_fitness(ind), reverse=True)
    return sorted_pop[:2]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

# -------------------------------
# 5. Run the Genetic Algorithm
# -------------------------------
population = [create_individual() for _ in range(population_size)]
fitness_history = []

for gen in range(num_generations):
    population_fitness = [individual_fitness(ind) for ind in population]
    fitness_history.append(max(population_fitness))
    
    new_population = []
    for _ in range(population_size // 2):
        parents = select_parents(population)
        child1, child2 = crossover(parents[0], parents[1])
        new_population.extend([mutate(child1), mutate(child2)])
    population = new_population
    print(f"Generation {gen}: Best fitness = {max(population_fitness):.4f}")

# -------------------------------
# 6. Final results
# -------------------------------
best_individual = max(population, key=individual_fitness)
best_features = [f for i, f in enumerate(feature_names) if best_individual[i] == 1]

print("\n=== Final Result ===")
print("Selected features:", best_features)
print("Number of selected features:", len(best_features))
print("Best Fitness:", max(fitness_history))

# Save results
results_df = pd.DataFrame({
    'Selected_Features': best_features,
    'Best_Fitness': [max(fitness_history)] * len(best_features)
})
results_df.to_csv('selected_features_results.csv', index=False)
print("\nResults saved to: selected_features_results.csv")

# -------------------------------
# 7. Plot fitness progress
# -------------------------------
plt.plot(fitness_history, marker='o')
plt.title("Best Fitness Progress Over Generations")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.savefig('fitness_plot.png')
plt.show()
print("\nFitness plot saved as: fitness_plot.png")

# -------------------------------
# 8. Compare GA vs All features
# -------------------------------
print("\n=== Starting Performance Comparison ===")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model with all features
model_all = RandomForestClassifier(random_state=42)
model_all.fit(X_train, y_train)
pred_all = model_all.predict(X_test)
acc_all = accuracy_score(y_test, pred_all)

# Model with GA-selected features
X_train_sel = X_train[best_features]
X_test_sel = X_test[best_features]
model_ga = RandomForestClassifier(random_state=42)
model_ga.fit(X_train_sel, y_train)
pred_ga = model_ga.predict(X_test_sel)
acc_ga = accuracy_score(y_test, pred_ga)

# Print comparison
print("\n=== Performance Comparison ===")
print(f"Accuracy (All Features): {acc_all:.3f}")
print(f"Accuracy (GA Selected Features): {acc_ga:.3f}")

# Plot comparison
plt.figure()
plt.bar(['All Features', 'GA Features'], [acc_all, acc_ga], color=['gray', 'green'])
plt.ylabel('Accuracy')
plt.title('Performance Comparison: All Features vs GA')
plt.savefig('comparison_accuracy.png')
plt.show()
print("\nComparison plot saved as: comparison_accuracy.png")
