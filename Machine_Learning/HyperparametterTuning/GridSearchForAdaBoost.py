import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the CICID2017 dataset (Replace 'path_to_dataset.csv' with the actual path to your dataset file)
dataset = pd.read_csv('path_to_dataset.csv')

# Separate features (X) and labels (y)
X = dataset.drop('label', axis=1)
y = dataset['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid for AdaBoost
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.5, 1.0],  
}

# Create the AdaBoost classifier
ada_classifier = AdaBoostClassifier(random_state=42)

# Create the Grid Search object
grid_search = GridSearchCV(ada_classifier, param_grid, cv=5, n_jobs=-1, verbose=2)

# Perform Grid Search on the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters found by Grid Search
best_ada_params = grid_search.best_params_
print("Best AdaBoost hyperparameters:", best_ada_params)

# Plot the Grid Search results
grid_means = grid_search.cv_results_['mean_test_score']
grid_params = grid_search.cv_results_['params']

plt.figure(figsize=(10, 6))
plt.plot(grid_means, marker='o')
plt.title("Grid Search for AdaBoost")
plt.xlabel("Hyperparameter Combinations")
plt.ylabel("Mean Test Score")
plt.xticks(np.arange(len(grid_params)), grid_params, rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Evaluate the best AdaBoost model on the test set
best_ada_model = grid_search.best_estimator_
y_pred_ada = best_ada_model.predict(X_test)
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))
print("AdaBoost Classification Report:")
print(classification_report(y_test, y_pred_ada))
