import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
dataset = pd.read_csv('CICID2017.csv')

X = dataset.drop('label', axis=1)
y = dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.5, 1.0],
}

ada_classifier = AdaBoostClassifier(random_state=42)

grid_search = GridSearchCV(ada_classifier, param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

best_ada_params = grid_search.best_params_
print("Best AdaBoost hyperparameters:", best_ada_params)

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
